# babylisten

Listens to the baby camera's audio and prints JSON events on stdout. [imcam](../imcam) runs it as a child
process and turns its events into alerts. It is the audio counterpart of babywatch (video).

**babylisten is not a medical device.** A room microphone cannot reliably detect apnea, so babylisten never
raises a "breathing stopped" alarm. `breathing_unsure` is informational only. Do not rely on it for safety.

## Install

```fish
uv tool install .            # babylisten lands in ~/.local/bin
```

For development: `uv venv; uv pip install -e '.[test,dev]'; .venv/bin/pytest -q`.

## Commands

```fish
babylisten live udp://127.0.0.1:47314       # what imcam runs; waits and reconnects, never exits on a stream
babylisten file clip.mkv [--realtime]       # same analysis on a file, as fast as possible by default
babylisten calibrate ~/ImCam/record         # noise floor + breathing stats -> ~/.babylisten/config.json
```

`live` and `file` take `-v` (log every breathing analysis to stderr) and these overrides:
`--threshold-db --hysteresis-db --noise-s --quiet-s --cooldown-s --heavy-bpm --heavy-db --status-s --no-breathing
--no-cry --config`. The source is always the last argument, as imcam expects:

```fish
set -x IMCAM_AUDIO_ANALYZER "babylisten live --threshold-db -35"
imcam monitor <serial> <permit>
```

## Events

One JSON object per line on stdout, flushed at once. All logging goes to stderr.

```json
{"event": "noise", "message": "loud sound for 2 s (-28 dBFS)", "push": true, "db": -27.6, "ts": "2026-11-02T03:12:45"}
```

| event              | push                    | when                                                                   |
|--------------------|-------------------------|------------------------------------------------------------------------|
| `noise`            | yes, once per cooldown  | level >= `threshold_db` for `noise_s` (not pushed during a cry)        |
| `quiet`            | no                      | level < `threshold_db - hysteresis_db` for `quiet_s` after `noise`     |
| `breathing_normal` | no                      | periodic breathing found, rate <= `heavy_bpm`                          |
| `breathing_heavy`  | yes, once per cooldown  | rate > `heavy_bpm`, or level > calibrated `baseline_db + heavy_db`     |
| `breathing_unsure` | no                      | no periodic breathing found (informational)                            |
| `cry`              | yes, once per cooldown  | cry-like sound for `cry_s`                                             |
| `cry_stopped`      | no                      | no cry-like sound for `cry_end_s` after `cry`                          |
| `status`           | no                      | every `status_s` seconds of audio: a snapshot of the current state     |

Events other than `status` are emitted on state changes only. `ts` is wall-clock time for streams and media time (s) for files.
Breathing events also carry `rate_bpm`, `periodicity`, `depth` and `band_db`; `cry` carries `f0_hz`.

`status` lets imcam's event log show the audio state at any moment, e.g. next to a babywatch event:

```json
{"event": "status", "message": "-42 dBFS, quiet, breathing normal 32/min", "push": false, "db": -43.1, "db_avg": -42.0, "loud": false, "crying": false, "cry_fraction": 0.0, "breathing": "normal", "rate_bpm": 32.1, "periodicity": 0.61, "depth": 0.24, "band_db": -48.3, "last_event": "breathing_normal", "last_event_age_s": 412.5, "ts": "2026-11-02T03:12:45"}
```

`db` is the latest 0.5 s block, `db_avg` the mean power over the last 10 s. `crying` / `breathing` are `null`
when that detector is off, `breathing` is `normal`, `heavy`, `unsure` or `null` before the first report, and
the breathing stats are those of the latest analysis. The first `status` comes `status_s` after the audio
starts, and again after a stream gap; there are none while no audio arrives.

## How it works

Audio is decoded with PyAV, downmixed to mono and resampled to 16 kHz, then analyzed in 0.5 s blocks.

- **Noise**: RMS level of each block in dBFS, with a hold time, hysteresis and push cooldown.
- **Breathing**: 100-1000 Hz band-pass, envelope = RMS per 0.1 s over a 30 s window, autocorrelation of
  the envelope. The peak in the 15-120/min range gives the rate; its height is the `periodicity`; the
  envelope's std/mean is the `depth`. Steady noise (fan, white-noise machine) has a flat envelope without
  a periodic peak, so it gives `breathing_unsure` rather than a false rate. A new state must hold for
  `breath_hold_s` before it is reported. Loud blocks, crying and stream gaps reset the window.
- **Cry**: 250-4000 Hz band-pass, 32 ms frames. A frame is cry-like when it is voiced (autocorrelation
  peak >= `cry_harmonicity`), its pitch is 250-700 Hz (infant cry; adult voices are lower, whistles
  higher), it has overtones (rejects pure-tone beeps) and it is at least `cry_min_db` loud. A cry is
  reported when `cry_fraction` of the frames in the last 2 s are cry-like for `cry_s`, so it also works
  below the noise threshold. A loud cry gives a `noise` event too, but only the `cry` is pushed.
  Limits: a child singing or talking in a high voice, or music with a tonal melody in that range, can
  look like crying.

## Settings

Defaults < `~/.babylisten/config.json` (override the directory with `set -x BABYLISTEN_HOME /path`, or the
file with `--config`) < command-line options. The file holds the fields of `detect.Settings`:

| key                 | default | meaning                                              |
|---------------------|---------|------------------------------------------------------|
| `threshold_db`      | -35     | noise level, dBFS RMS                                |
| `hysteresis_db`     | 6       | `quiet` below threshold minus this                   |
| `noise_s`           | 2       | seconds above threshold for `noise`                  |
| `quiet_s`           | 10      | seconds below for `quiet`                            |
| `noise_cooldown_s`  | 300     | min seconds between pushed `noise` events            |
| `breathing`         | true    | run breathing analysis                               |
| `breath_window_s`   | 30      | envelope window                                      |
| `breath_hop_s`      | 2       | seconds between analyses                             |
| `breath_hold_s`     | 10      | a new breathing state must hold this long            |
| `min_periodicity`   | 0.4     | autocorrelation peak needed for breathing            |
| `min_depth`         | 0.1     | envelope std/mean needed for breathing               |
| `heavy_bpm`         | 60      | breaths/min above which breathing is heavy           |
| `baseline_db`       | null    | typical breathing level (set by `calibrate`)         |
| `heavy_db`          | 6       | dB above `baseline_db` for heavy                     |
| `heavy_cooldown_s`  | 600     | min seconds between pushed `breathing_heavy` events  |
| `cry`               | true    | run cry detection                                    |
| `cry_min_db`        | -50     | min frame level (250-4000 Hz) for a cry-like frame   |
| `cry_harmonicity`   | 0.5     | min voicing (autocorrelation peak)                   |
| `cry_fraction`      | 0.4     | share of cry-like frames in 2 s needed               |
| `cry_s`             | 3       | seconds of cry-like sound for `cry`                  |
| `cry_end_s`         | 15      | seconds without for `cry_stopped`                    |
| `cry_cooldown_s`    | 300     | min seconds between pushed `cry` events              |
| `status_s`          | 30      | seconds between `status` lines, 0 = off              |

`calibrate` runs over `imcam watch --record` clips (or any audio files; files without audio are skipped),
prints level percentiles and breathing statistics, and saves `threshold_db = median + 20 dB` (`--margin-db`)
and, when enough periodic windows were found, `baseline_db` and `heavy_bpm = 1.25 x p90 rate`. Use
`--dry-run` to only print.

## Testing without the camera

```fish
.venv/bin/pytest -q          # synthetic signals; the file test needs ffmpeg with libopus
# stream path, the way imcam publishes it:
ffmpeg -re -f lavfi -i "anoisesrc=c=pink:a=0.1" -c:a libopus -f mpegts "udp://127.0.0.1:47314?pkt_size=1316"
babylisten live -v udp://127.0.0.1:47314
```

Only one process can read the UDP port, so `ffplay` and babylisten cannot listen at the same time.

## License

[MIT](LICENSE)
