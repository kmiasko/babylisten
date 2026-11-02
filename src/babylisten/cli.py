"""babylisten <command> [args]: noise, breathing and cry events. stdout carries JSON events only (live, file); all logging goes to stderr.

  babylisten live <url-or-file>         what imcam runs: never exits on a missing or dropped stream
  babylisten file <clip> [--realtime]   same analysis on a file, as fast as possible by default
  babylisten calibrate <dir-or-clips>   noise floor and breathing stats -> ~/.babylisten/config.json

Settings: defaults < ~/.babylisten/config.json (or --config) < command-line options.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import av
import numpy as np

from . import paths
from .audio import RATE, NoAudio, is_stream, iter_blocks
from .detect import Analyzer, Settings

log = logging.getLogger("babylisten")

AUDIO_EXT = (".mkv", ".mp4", ".ts", ".wav", ".flac", ".ogg", ".opus", ".mp3", ".m4a")
OVERRIDES = {  # option -> Settings field
    "threshold_db": "threshold_db",
    "hysteresis_db": "hysteresis_db",
    "noise_s": "noise_s",
    "quiet_s": "quiet_s",
    "cooldown_s": "noise_cooldown_s",
    "heavy_bpm": "heavy_bpm",
    "heavy_db": "heavy_db",
    "status_s": "status_s",
}


def _settings_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=str(paths.CONFIG), help="settings file (JSON)")
    p.add_argument(
        "-v", "--verbose", action="store_true", help="log every breathing analysis"
    )
    g = p.add_argument_group("overrides")
    g.add_argument("--threshold-db", type=float, help="noise level, dBFS RMS")
    g.add_argument(
        "--hysteresis-db", type=float, help="quiet below threshold minus this"
    )
    g.add_argument("--noise-s", type=float, help="seconds above threshold for 'noise'")
    g.add_argument("--quiet-s", type=float, help="seconds below for 'quiet'")
    g.add_argument(
        "--cooldown-s", type=float, help="min seconds between pushed 'noise'"
    )
    g.add_argument(
        "--heavy-bpm", type=float, help="breaths/min above which breathing is heavy"
    )
    g.add_argument(
        "--heavy-db",
        type=float,
        help="dB above the calibrated breathing level for heavy",
    )
    g.add_argument(
        "--status-s", type=float, help="seconds between 'status' lines, 0 = off"
    )
    g.add_argument("--no-breathing", action="store_true", help="no breathing events")
    g.add_argument("--no-cry", action="store_true", help="no cry events")


def _settings(a: argparse.Namespace) -> Settings:
    s = Settings.load(a.config)
    for opt, field in OVERRIDES.items():
        v = getattr(a, opt)
        if v is not None:
            setattr(s, field, v)
    if a.no_breathing:
        s.breathing = False
    if a.no_cry:
        s.cry = False
    return s


def _run(a: argparse.Namespace, realtime: bool) -> int:
    s = _settings(a)
    stream = is_stream(a.src)
    log.info("settings %s", json.dumps(dataclasses.asdict(s)))
    an = Analyzer(s, RATE)
    try:
        for t, x, new in iter_blocks(a.src, realtime=realtime):
            for ev in an.feed(t, x, new):
                t_ev = ev.pop("t")
                ts = (
                    datetime.now().isoformat(timespec="seconds")
                    if stream
                    else round(t_ev, 1)
                )
                print(json.dumps({**ev, "ts": ts}), flush=True)
            if a.verbose and "periodicity" in an.last:
                log.info("%s", json.dumps(an.last))
    except (NoAudio, av.error.FFmpegError) as e:
        log.error("%s", e)
        return 1
    return 0


def _clips(args: list[str]) -> list[Path]:
    out = []
    for arg in args:
        p = Path(arg).expanduser()
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix.lower() in AUDIO_EXT)
        else:
            out.append(p)
    return out


def _calibrate(a: argparse.Namespace) -> int:
    s = Settings.load(a.config)
    probe = dataclasses.replace(s, breathing=True)
    dbs, breaths, n_clips = [], [], 0
    for clip in _clips(a.clips):
        an = Analyzer(probe, RATE)
        try:
            for t, x, new in iter_blocks(clip, reconnect=False):
                an.feed(t, x, new)
                dbs.append(an.last["db"])
                if "periodicity" in an.last:
                    breaths.append(an.last)
        except NoAudio as e:
            log.error("skipped: %s", e)
            continue
        except av.error.FFmpegError as e:
            log.error("skipped %s: %s", clip, e)
            continue
        n_clips += 1
    if not dbs:
        log.error("no audio found")
        return 1

    db = np.array(dbs)
    p10, p50, p90, p99 = np.percentile(db, [10, 50, 90, 99])
    print(f"clips: {n_clips}, audio: {len(db) * 0.5 / 3600:.2f} h")
    print(
        f"level dBFS: p10 {p10:.1f}  p50 {p50:.1f}  p90 {p90:.1f}  p99 {p99:.1f}  max {db.max():.1f}"
    )
    s.threshold_db = round(float(p50) + a.margin_db, 1)
    print(f"-> threshold_db {s.threshold_db} (p50 + {a.margin_db:g})")

    ok = [
        b
        for b in breaths
        if b["periodicity"] >= s.min_periodicity
        and b["depth"] >= s.min_depth
        and b.get("rate_bpm") is not None
    ]
    print(f"breathing analyses: {len(breaths)}, periodic: {len(ok)}", end="")
    if breaths:
        per = np.array([b["periodicity"] for b in breaths])
        print(
            f" ({100 * len(ok) / len(breaths):.0f}%); periodicity p50 {np.median(per):.2f} p90 {np.percentile(per, 90):.2f}"
        )
    else:
        print()
    if len(ok) >= a.min_breaths:
        rate = np.array([b["rate_bpm"] for b in ok])
        band = np.array([b["band_db"] for b in ok])
        r50, r90 = np.percentile(rate, [50, 90])
        print(
            f"breathing rate /min: p10 {np.percentile(rate, 10):.0f}  p50 {r50:.0f}  p90 {r90:.0f}"
        )
        print(
            f"breathing level dBFS (100-1000 Hz): p50 {np.median(band):.1f}  p90 {np.percentile(band, 90):.1f}"
        )
        s.baseline_db = round(float(np.median(band)), 1)
        s.heavy_bpm = round(1.25 * float(r90))
        print(f"-> baseline_db {s.baseline_db}, heavy_bpm {s.heavy_bpm:g} (1.25 x p90)")
    else:
        print(
            f"breathing: fewer than {a.min_breaths} periodic windows, breathing settings unchanged"
        )

    if a.dry_run:
        print("dry run, nothing saved")
    else:
        s.save(a.config)
        print(f"saved {a.config}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="babylisten",
        description="Baby-room audio analyzer: JSON events on stdout. Not a medical device.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser(
        "live", help="analyze a stream (or file); reconnects, never exits on a stream"
    )
    p.add_argument("src", help="stream URL (udp://...) or file")
    _settings_args(p)
    p = sub.add_parser("file", help="analyze a file, as fast as possible by default")
    p.add_argument("src", help="audio or video file")
    p.add_argument("--realtime", action="store_true", help="pace at native speed")
    _settings_args(p)
    p = sub.add_parser(
        "calibrate", help="noise floor and breathing stats -> settings file"
    )
    p.add_argument(
        "clips", nargs="+", help="clips or directories (e.g. imcam --record segments)"
    )
    p.add_argument(
        "--config", default=str(paths.CONFIG), help="settings file to update"
    )
    p.add_argument(
        "--margin-db",
        type=float,
        default=20.0,
        help="threshold_db = median level + this",
    )
    p.add_argument(
        "--min-breaths",
        type=int,
        default=30,
        help="periodic windows needed to set breathing",
    )
    p.add_argument("--dry-run", action="store_true", help="print only, do not save")
    a = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    try:
        if a.cmd == "calibrate":
            code = _calibrate(a)
        else:
            code = _run(a, realtime=a.cmd == "file" and a.realtime)
    except KeyboardInterrupt:
        code = 130
    sys.exit(code)


if __name__ == "__main__":
    main()
