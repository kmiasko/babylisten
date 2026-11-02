"""Detectors: pure functions and state machines, samples in, events out. No I/O except Settings load/save.

Events are dicts {"event", "message", "push", "t", ...extra}; t is the analyzer clock in seconds (the CLI
turns it into "ts"). Except "status", they are emitted on state changes only, and pushes are rate-limited here: imcam
delivers every line it gets.

- noise: block level (RMS dBFS) >= threshold_db for noise_s -> "noise" (push, with cooldown); below
  threshold_db - hysteresis_db for quiet_s -> "quiet".
- breathing: envelope (RMS per 0.1 s) of 100-1000 Hz audio over a breath_window_s window; its
  autocorrelation peak in 15-120/min gives the rate and the periodicity. Steady broadband noise (fan,
  white-noise machine) has a flat envelope with no periodic peak and is rejected by min_periodicity and
  min_depth. States: breathing_normal, breathing_heavy (rate above heavy_bpm, or level above the calibrated
  baseline_db + heavy_db; push, with cooldown), breathing_unsure (informational). Never an alarm: a room
  microphone cannot detect apnea.
- cry: 32 ms frames of 250-4000 Hz audio; a frame is cry-like when it is voiced (autocorrelation peak >=
  cry_harmonicity) with a pitch of 250-700 Hz (an infant cry; adult voices are lower, beeps and whistles
  higher), has overtones (not a pure tone) and is at least cry_min_db loud. When cry_fraction of the frames in the last 2 s are cry-like for
  cry_s -> "cry" (push, with cooldown); none for cry_end_s -> "cry_stopped". A "noise" event during a cry
  is not pushed (the cry is). Crying resets the breathing window.
- status: every status_s seconds of audio (0 = off) -> "status" (never pushed), a snapshot of the current
  state: level (latest block and mean over STATUS_AVG_S), noise, cry and breathing state, the latest
  breathing stats and the most recent other event. The timer restarts after a gap.
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from scipy.signal import butter, detrend, find_peaks, sosfilt

log = logging.getLogger(__name__)

ENV_RATE = 10  # envelope samples per second
BAND_HZ = (100.0, 1000.0)
RATE_HZ = (0.25, 2.0)  # breathing rates searched: 15-120 per minute
CRY_BAND_HZ = (250.0, 4000.0)
CRY_F0_HZ = (250.0, 700.0)
CRY_FRAME_S = 0.032
CRY_WINDOW_S = 2.0
CRY_MIN_OVERTONES = (
    0.1  # share of a frame's power above 1.5 x pitch; pure tones (beeps) have ~0
)
STATUS_AVG_S = 10.0  # window of the status line's db_avg


@dataclass
class Settings:
    threshold_db: float = -35.0
    hysteresis_db: float = 6.0
    noise_s: float = 2.0
    quiet_s: float = 10.0
    noise_cooldown_s: float = 300.0
    breathing: bool = True
    breath_window_s: float = 30.0
    breath_hop_s: float = 2.0
    breath_hold_s: float = 10.0
    min_periodicity: float = 0.4
    min_depth: float = 0.1
    heavy_bpm: float = 60.0
    baseline_db: float | None = None
    heavy_db: float = 6.0
    heavy_cooldown_s: float = 600.0
    cry: bool = True
    cry_min_db: float = -50.0
    cry_harmonicity: float = 0.5
    cry_fraction: float = 0.4
    cry_s: float = 3.0
    cry_end_s: float = 15.0
    cry_cooldown_s: float = 300.0
    status_s: float = 30.0

    @classmethod
    def load(cls, path: str | Path) -> Settings:
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        names = {f.name for f in fields(cls)}
        for k in sorted(set(data) - names):
            log.warning("%s: unknown setting %r ignored", path, k)
        return cls(**{k: v for k, v in data.items() if k in names})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


def event(name: str, message: str, push: bool, t: float, **extra) -> dict:
    return {"event": name, "message": message, "push": push, "t": t, **extra}


def level_db(x: np.ndarray) -> float:
    """RMS level in dBFS (full scale = 1.0)."""
    rms = math.sqrt(float(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0
    return 20 * math.log10(max(rms, 1e-10))


def breath_stats(env: np.ndarray, fs: float = ENV_RATE) -> tuple[float, float, float]:
    """(rate per minute, periodicity 0..1, modulation depth) of an amplitude envelope.

    periodicity is the normalized autocorrelation at the breathing lag (the shortest lag whose peak is
    within 85% of the highest one, so a multiple of the period is not taken for a slower rate).
    depth is std/mean of the envelope. rate is nan when there is no peak in the searched range.
    """
    env = np.asarray(env, np.float64)
    m = env.mean()
    if m <= 0:
        return math.nan, 0.0, 0.0
    depth = float(env.std() / m)
    e = detrend(env)
    n = len(e)
    spec = np.fft.rfft(e, 2 * n)
    r = np.fft.irfft(spec * np.conj(spec))[:n]
    if r[0] <= 0:
        return math.nan, 0.0, depth
    r = (
        r / r[0] * n / (n - np.arange(n))
    )  # unbiased: undo the taper of the finite window
    kmin, kmax = math.floor(fs / RATE_HZ[1]), min(math.ceil(fs / RATE_HZ[0]), n - 2)
    peaks = [p for p in find_peaks(r[: kmax + 2])[0] if kmin <= p <= kmax]
    best = max((r[p] for p in peaks), default=0.0)
    if best <= 0:
        return math.nan, 0.0, depth
    k = next(p for p in peaks if r[p] >= 0.85 * best)
    a, b, c = r[k - 1], r[k], r[k + 1]
    den = a - 2 * b + c
    lag = (k + (0.5 * (a - c) / den if den else 0.0)) / fs
    return 60.0 / lag, float(min(b, 1.0)), depth


def pitch(frames: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """(f0 Hz, harmonicity 0..1) per row of frames, from the normalized autocorrelation.

    The pitch lag is the shortest one (pitch up to 2 kHz) whose peak is within 85% of the highest peak
    with pitch >= CRY_F0_HZ[0], so a high tone is not taken for its subharmonic.
    """
    x = np.diff(frames, axis=1)  # pre-emphasis: flattens pink/brown background noise
    x = x - x.mean(axis=1, keepdims=True)
    n = x.shape[1]
    spec = np.fft.rfft(x, 2 * n, axis=1)
    r = np.fft.irfft(spec * np.conj(spec), axis=1)[:, :n]
    r0 = np.maximum(r[:, :1], 1e-20)
    r = r / r0 * n / (n - np.arange(n))
    kmin, kmax = rate // 2000, math.ceil(rate / CRY_F0_HZ[0])
    mid = r[:, kmin : kmax + 1]
    peak = (mid > r[:, kmin - 1 : kmax]) & (mid >= r[:, kmin + 1 : kmax + 2])
    best = np.where(peak, mid, -np.inf).max(axis=1)
    ok = peak & (mid >= 0.85 * best[:, None])
    k = kmin + ok.argmax(axis=1)
    h = np.where(np.isfinite(best) & (best > 0), r[np.arange(len(r)), k], 0.0)
    return rate / k, np.clip(h, 0.0, 1.0)


def overtones(frames: np.ndarray, f0: np.ndarray, rate: int) -> np.ndarray:
    """Share of each frame's power above 1.5 x f0 (Hann-windowed spectrum)."""
    p = np.abs(np.fft.rfft(frames * np.hanning(frames.shape[1]), axis=1)) ** 2
    f = np.fft.rfftfreq(frames.shape[1], 1 / rate)
    above = (p * (f[None, :] > 1.5 * f0[:, None])).sum(axis=1)
    return above / np.maximum(p.sum(axis=1), 1e-20)


class CryDetector:
    def __init__(self, s: Settings, rate: int):
        self.s = s
        self.rate = rate
        self.sos = butter(4, CRY_BAND_HZ, btype="bandpass", fs=rate, output="sos")
        self.frame = round(CRY_FRAME_S * rate)
        self.hist: deque[bool] = deque(maxlen=round(CRY_WINDOW_S / CRY_FRAME_S))
        self.f0s: deque[float] = deque(maxlen=200)  # pitch of recent cry-like frames
        self.zi: np.ndarray | None = None
        self.buf = np.zeros(0, np.float32)
        self.crying = False
        self.since: float | None = None
        self.start = 0.0
        self.last_push: float | None = None

    def reset(self) -> None:
        self.zi = None
        self.buf = np.zeros(0, np.float32)
        self.hist.clear()
        self.since = None

    @property
    def fraction(self) -> float:
        if len(self.hist) < self.hist.maxlen // 2:
            return 0.0
        return sum(self.hist) / len(self.hist)

    @property
    def like(self) -> bool:
        """The last ~2 s sound like crying."""
        return self.fraction >= self.s.cry_fraction

    def update(self, t: float, x: np.ndarray, dt: float) -> list[dict]:
        s = self.s
        if self.zi is None:
            self.zi = np.zeros((self.sos.shape[0], 2))
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        self.buf = np.concatenate([self.buf, y])
        n = len(self.buf) // self.frame * self.frame
        if n:
            frames = self.buf[:n].reshape(-1, self.frame).astype(np.float64)
            self.buf = self.buf[n:]
            f0, h = pitch(frames, self.rate)
            db = 10 * np.log10(np.maximum(np.mean(frames**2, axis=1), 1e-20))
            cry = (
                (f0 >= CRY_F0_HZ[0])
                & (f0 <= CRY_F0_HZ[1])
                & (h >= s.cry_harmonicity)
                & (overtones(frames, f0, self.rate) >= CRY_MIN_OVERTONES)
                & (db >= s.cry_min_db)
            )
            self.hist.extend(cry.tolist())
            self.f0s.extend(f0[cry].tolist())
        like = self.like
        if not self.crying:
            if not like:
                self.since = None
                return []
            if self.since is None:
                self.since = t - dt
            if t - self.since < s.cry_s:
                return []
            self.crying, self.start, self.since = True, self.since, None
            push = self.last_push is None or t - self.last_push >= s.cry_cooldown_s
            if push:
                self.last_push = t
            f0 = float(np.median(self.f0s)) if self.f0s else math.nan
            msg = f"baby crying for {t - self.start:.0f} s (pitch ~{f0:.0f} Hz)"
            return [
                event(
                    "cry",
                    msg if push else msg + ", within cooldown",
                    push,
                    t,
                    f0_hz=round(f0),
                )
            ]
        if like:
            self.since = None
            return []
        if self.since is None:
            self.since = t - dt
        if t - self.since < s.cry_end_s:
            return []
        self.crying = False
        dur, self.since = self.since - self.start, None
        return [event("cry_stopped", f"crying stopped after {dur:.0f} s", False, t)]


class NoiseDetector:
    def __init__(self, s: Settings):
        self.s = s
        self.loud = False
        self.since: float | None = None  # start of the candidate transition
        self.start = 0.0
        self.peak = -math.inf
        self.last_push: float | None = None

    def reset(self) -> None:
        self.since = None

    def update(
        self, t: float, db: float, dt: float, crying: bool = False
    ) -> list[dict]:
        """crying: a cry is (being) detected; the noise event is then reported, not pushed."""
        s = self.s
        if not self.loud:
            if db < s.threshold_db:
                self.since = None
                return []
            if self.since is None:
                self.since, self.peak = t - dt, db
            self.peak = max(self.peak, db)
            if t - self.since < s.noise_s:
                return []
            self.loud, self.start, self.since = True, self.since, None
            push = not crying and (
                self.last_push is None or t - self.last_push >= s.noise_cooldown_s
            )
            if push:
                self.last_push = t
            msg = f"loud sound for {t - self.start:.0f} s ({self.peak:.0f} dBFS)"
            if crying:
                msg += ", crying"
            elif not push:
                msg += ", within cooldown"
            return [
                event(
                    "noise",
                    msg,
                    push,
                    t,
                    db=round(self.peak, 1),
                )
            ]
        self.peak = max(self.peak, db)
        if db >= s.threshold_db - s.hysteresis_db:
            self.since = None
            return []
        if self.since is None:
            self.since = t - dt
        if t - self.since < s.quiet_s:
            return []
        self.loud = False
        dur, self.since = self.since - self.start, None
        msg = f"quiet again after {dur:.0f} s of noise (peak {self.peak:.0f} dBFS)"
        return [event("quiet", msg, False, t, db=round(db, 1))]


class BreathingDetector:
    def __init__(self, s: Settings, rate: int):
        self.s = s
        self.sos = butter(4, BAND_HZ, btype="bandpass", fs=rate, output="sos")
        self.frame = rate // ENV_RATE
        self.env: deque[float] = deque(maxlen=round(s.breath_window_s * ENV_RATE))
        self.zi: np.ndarray | None = None
        self.next_t: float | None = None
        self.state: str | None = None
        self.cand: str | None = None
        self.cand_since = 0.0
        self.last_push: float | None = None
        self.stats: dict = {}  # of the latest analysis

    def reset(self) -> None:
        """Forget the signal (gap or loud noise); the reported state stays."""
        self.zi = None
        self.env.clear()
        self.next_t = None
        self.cand = None

    def classify(self, st: dict) -> str:
        s = self.s
        if st["periodicity"] < s.min_periodicity or st["depth"] < s.min_depth:
            return "breathing_unsure"
        if st["rate_bpm"] > s.heavy_bpm:
            return "breathing_heavy"
        if s.baseline_db is not None and st["band_db"] > s.baseline_db + s.heavy_db:
            return "breathing_heavy"
        return "breathing_normal"

    def update(self, t: float, x: np.ndarray) -> tuple[list[dict], dict | None]:
        """Events and, when an analysis ran on this block, its stats."""
        if self.zi is None:
            self.zi = np.zeros((self.sos.shape[0], 2))
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        n = len(y) // self.frame * self.frame
        self.env.extend(np.sqrt(np.mean(y[:n].reshape(-1, self.frame) ** 2, axis=1)))
        if len(self.env) < self.env.maxlen or (
            self.next_t is not None and t < self.next_t
        ):
            return [], None
        self.next_t = t + self.s.breath_hop_s
        env = np.fromiter(self.env, np.float64)
        rate, per, depth = breath_stats(env)
        st = {
            "rate_bpm": round(rate, 1) if not math.isnan(rate) else None,
            "periodicity": round(per, 3),
            "depth": round(depth, 3),
            "band_db": round(
                20 * math.log10(max(math.sqrt(np.mean(env**2)), 1e-10)), 1
            ),
        }
        self.stats = st
        return self._hold(t, self.classify(st), st), st

    def _hold(self, t: float, state: str, st: dict) -> list[dict]:
        if state == self.state:
            self.cand = None
            return []
        if state != self.cand:
            self.cand, self.cand_since = state, t
        if t - self.cand_since < self.s.breath_hold_s:
            return []
        self.state, self.cand = state, None
        extra = {k: v for k, v in st.items() if v is not None}
        if state == "breathing_unsure":
            msg = "breathing not detected (informational, not a safety alarm)"
            return [event(state, msg, False, t, **extra)]
        if state == "breathing_normal":
            return [
                event(state, f"breathing ~{st['rate_bpm']:.0f}/min", False, t, **extra)
            ]
        push = self.last_push is None or t - self.last_push >= self.s.heavy_cooldown_s
        if push:
            self.last_push = t
        msg = f"heavy breathing ~{st['rate_bpm']:.0f}/min ({st['band_db']:.0f} dBFS)"
        return [
            event(state, msg if push else msg + ", within cooldown", push, t, **extra)
        ]


class Analyzer:
    """Feed consecutive blocks; get events. Loud or crying blocks reset the breathing window."""

    def __init__(self, s: Settings, rate: int):
        self.s = s
        self.rate = rate
        self.noise = NoiseDetector(s)
        self.breath = BreathingDetector(s, rate) if s.breathing else None
        self.cry = CryDetector(s, rate) if s.cry else None
        self.last: dict = {}  # diagnostics of the latest block: t, db, and breathing stats if any
        self.levels: deque[tuple[float, float]] = deque()  # (t, db) over STATUS_AVG_S
        self.next_status: float | None = None
        self.last_event: tuple[str, float] | None = None

    def feed(self, t: float, x: np.ndarray, new_segment: bool = False) -> list[dict]:
        if new_segment:
            self.levels.clear()
            self.next_status = None
            self.noise.reset()
            if self.breath:
                self.breath.reset()
            if self.cry:
                self.cry.reset()
        db = level_db(x)
        dt = len(x) / self.rate
        self.last = {"t": t, "db": round(db, 1)}
        events = self.cry.update(t, x, dt) if self.cry else []
        crying = bool(self.cry) and (self.cry.like or self.cry.crying)
        if self.cry:
            self.last["cry_fraction"] = round(self.cry.fraction, 2)
        events = self.noise.update(t, db, dt, crying) + events
        if self.breath:
            if db >= self.s.threshold_db or self.noise.loud or crying:
                self.breath.reset()
            else:
                ev, st = self.breath.update(t, x)
                events += ev
                if st:
                    self.last.update(st)
        if events:
            self.last_event = (events[-1]["event"], events[-1]["t"])
        self.levels.append((t, db))
        while t - self.levels[0][0] >= STATUS_AVG_S:
            self.levels.popleft()
        if self.s.status_s > 0:
            if self.next_status is None:
                self.next_status = t + self.s.status_s
            elif t >= self.next_status:
                self.next_status = t + self.s.status_s
                events.append(self.status(t))
        return events

    def status(self, t: float) -> dict:
        """Snapshot of the current state as a "status" event."""
        db = self.last["db"]
        power = np.mean([10 ** (d / 10) for _, d in self.levels])
        db_avg = round(10 * math.log10(max(float(power), 1e-20)), 1)
        crying = self.cry.crying if self.cry else None
        breathing = None
        if self.breath and self.breath.state:
            breathing = self.breath.state.removeprefix("breathing_")
        stats = self.breath.stats if self.breath else {}
        parts = [f"{db_avg:.0f} dBFS", "loud" if self.noise.loud else "quiet"]
        if crying:
            parts.append("crying")
        elif breathing == "unsure":
            parts.append("breathing unsure")
        elif breathing and stats.get("rate_bpm") is not None:
            parts.append(f"breathing {breathing} {stats['rate_bpm']:.0f}/min")
        elif breathing:
            parts.append(f"breathing {breathing}")
        name, t_ev = self.last_event or (None, None)
        return event(
            "status",
            ", ".join(parts),
            False,
            t,
            db=db,
            db_avg=db_avg,
            loud=self.noise.loud,
            crying=crying,
            cry_fraction=self.last.get("cry_fraction"),
            breathing=breathing,
            rate_bpm=stats.get("rate_bpm"),
            periodicity=stats.get("periodicity"),
            depth=stats.get("depth"),
            band_db=stats.get("band_db"),
            last_event=name,
            last_event_age_s=None if t_ev is None else round(t - t_ev, 1),
        )
