import json

import numpy as np
import pytest

from babylisten.detect import Analyzer, Settings, breath_stats, level_db, pitch

RATE = 16000
BLOCK = RATE // 2


def pink(seconds, db, seed=0):
    """Pink noise at an RMS level of db dBFS."""
    n = int(seconds * RATE)
    spec = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / RATE)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    return (x / np.sqrt(np.mean(x**2)) * 10 ** (db / 20)).astype(np.float32)


def breathing(seconds, hz, db=-50, seed=0):
    t = np.arange(int(seconds * RATE)) / RATE
    return pink(seconds, db, seed) * (
        0.3 + 0.7 * 0.5 * (1 + np.sin(2 * np.pi * hz * t))
    )


def run(x, s=None, t0=0.0, an=None, status=False):
    """Events of feeding x in blocks; "status" events only when status is set."""
    an = an or Analyzer(s or Settings(), RATE)
    events = []
    for i in range(0, len(x) - BLOCK + 1, BLOCK):
        events += an.feed(t0 + (i + BLOCK) / RATE, x[i : i + BLOCK])
    if not status:
        events = [e for e in events if e["event"] != "status"]
    return events, an


def names(events):
    return [e["event"] for e in events]


def envelope(x):
    return np.sqrt(np.mean(x[: len(x) // 1600 * 1600].reshape(-1, 1600) ** 2, axis=1))


def test_level_db():
    assert level_db(np.full(100, 0.1)) == pytest.approx(-20)
    assert level_db(np.zeros(100)) == pytest.approx(-200)
    assert level_db(pink(1, -40)) == pytest.approx(-40, abs=0.01)


@pytest.mark.parametrize("hz", [0.4, 0.5, 1.0, 1.5])
def test_breath_stats_rate(hz):
    rate, per, depth = breath_stats(envelope(breathing(30, hz)))
    assert rate == pytest.approx(60 * hz, rel=0.05)
    assert per > 0.5
    assert depth > 0.2


@pytest.mark.parametrize("seed", range(5))
def test_breath_stats_steady_noise_not_periodic(seed):
    _, per, _ = breath_stats(envelope(pink(30, -40, seed)))
    assert per < Settings().min_periodicity


def test_breath_stats_silence():
    rate, per, depth = breath_stats(np.zeros(300))
    assert np.isnan(rate) and per == 0 and depth == 0


def test_normal_breathing():
    events, _ = run(breathing(60, 0.5))
    assert names(events) == ["breathing_normal"]
    ev = events[0]
    assert ev["push"] is False
    assert ev["rate_bpm"] == pytest.approx(30, rel=0.05)
    json.dumps(ev)


def test_heavy_breathing_pushes_once_per_cooldown():
    events, an = run(breathing(60, 1.5))
    assert names(events) == ["breathing_heavy"]
    assert events[0]["push"] is True
    assert events[0]["rate_bpm"] == pytest.approx(90, rel=0.05)
    # back to normal, then heavy again within the cooldown: reported, not pushed
    events, _ = run(breathing(60, 0.5, seed=1), t0=60, an=an)
    assert names(events) == ["breathing_normal"]
    events, _ = run(breathing(60, 1.5, seed=2), t0=120, an=an)
    assert names(events) == ["breathing_heavy"]
    assert events[0]["push"] is False


def test_heavy_by_level_needs_baseline():
    base = run(breathing(60, 0.5, db=-60))[0][0]["band_db"]
    s = Settings(baseline_db=base, heavy_db=6)
    assert names(run(breathing(60, 0.5, db=-58, seed=1), s)[0]) == ["breathing_normal"]
    assert names(run(breathing(60, 0.5, db=-50, seed=1), s)[0]) == ["breathing_heavy"]


def test_steady_noise_is_unsure_not_breathing():
    events, _ = run(pink(90, -45))
    assert names(events) == ["breathing_unsure"]
    assert events[0]["push"] is False


def test_noise_burst_then_quiet():
    x = np.concatenate([pink(10, -60, 1), pink(3, -20, 2), pink(20, -60, 3)])
    events, an = run(x, Settings(breathing=False))
    assert names(events) == ["noise", "quiet"]
    noise, quiet = events
    assert noise["push"] is True
    assert noise["t"] == pytest.approx(12.0, abs=0.6)
    assert noise["db"] == pytest.approx(-20, abs=3)
    assert "dBFS" in noise["message"]
    assert quiet["push"] is False
    # a second burst within the cooldown is reported but not pushed; after it, pushed again
    events, _ = run(x, t0=40, an=an)
    assert names(events) == ["noise", "quiet"] and events[0]["push"] is False
    events, _ = run(x, t0=400, an=an)
    assert events[0]["push"] is True


def test_short_burst_is_ignored():
    x = np.concatenate([pink(5, -60, 1), pink(1, -20, 2), pink(5, -60, 3)])
    events, _ = run(x, Settings(breathing=False))
    assert events == []


def test_hysteresis_keeps_noise_state():
    # 4 dB below the threshold is inside the 6 dB hysteresis: still loud, no 'quiet'
    x = np.concatenate([pink(3, -30, 1), pink(20, -39, 2)])
    events, _ = run(x, Settings(breathing=False, threshold_db=-35))
    assert names(events) == ["noise"]


def test_loud_noise_resets_breathing_window():
    x = np.concatenate([breathing(60, 0.5), pink(3, -20, 5)])
    events, an = run(x)
    assert names(events) == ["breathing_normal", "noise"]
    assert len(an.breath.env) == 0


def test_new_segment_resets_timers():
    an = Analyzer(Settings(breathing=False), RATE)
    loud = pink(1.5, -20)
    run(loud, an=an)
    an.feed(100.5, loud[:BLOCK], new_segment=True)
    events, _ = run(loud[: BLOCK * 2], t0=100.5, an=an)
    assert events == []  # 1.5 s before the gap + 1.5 s after is not 2 s of noise


def test_settings_load_save(tmp_path, caplog):
    p = tmp_path / "sub" / "config.json"
    assert Settings.load(p) == Settings()
    Settings(threshold_db=-42, baseline_db=-55.5).save(p)
    assert Settings.load(p).threshold_db == -42
    data = json.loads(p.read_text())
    data["bogus"] = 1
    p.write_text(json.dumps(data))
    assert Settings.load(p).baseline_db == -55.5
    assert "bogus" in caplog.text


def cry(seconds, db, f0=450.0, harmonics=10):
    """Harmonic tone with a gliding pitch in 1 s units separated by 0.3 s inhalation pauses."""
    t = np.arange(int(seconds * RATE)) / RATE
    ph = 2 * np.pi * np.cumsum(f0 * (1 + 0.08 * np.sin(2 * np.pi * 0.7 * t))) / RATE
    x = sum(np.sin(h * ph) / h for h in range(1, harmonics + 1) if h * f0 < 7000)
    unit = t % 1.3
    x = x * np.where(unit < 1.0, np.sin(np.pi * unit), 0.0)
    return (x / np.sqrt(np.mean(x[x != 0] ** 2)) * 10 ** (db / 20)).astype(np.float32)


def test_pitch_of_cry():
    frames = cry(1, -30)[: 15 * 512].reshape(15, 512).astype(np.float64)
    f0, h = pitch(frames[3:12], RATE)  # middle of the first unit
    assert np.median(f0) == pytest.approx(450, rel=0.1)
    assert np.all(h > 0.8)


def test_loud_cry():
    x = np.concatenate(
        [pink(5, -60, 1), cry(10, -25) + pink(10, -60, 2), pink(30, -60, 3)]
    )
    events, _ = run(x, Settings(breathing=False))
    assert names(events) == ["noise", "cry", "quiet", "cry_stopped"]
    noise, c, _, stopped = events
    assert noise["push"] is False and "crying" in noise["message"]
    assert c["push"] is True
    assert c["t"] == pytest.approx(8, abs=1.5)
    assert c["f0_hz"] == pytest.approx(450, rel=0.1)
    assert stopped["push"] is False
    assert stopped["t"] == pytest.approx(15 + 15, abs=2)


def test_quiet_cry_without_noise_and_cooldown():
    x = np.concatenate([cry(10, -45) + pink(10, -50, 1), pink(20, -50, 2)])
    events, an = run(x, Settings(breathing=False))
    assert names(events) == ["cry", "cry_stopped"]
    assert events[0]["push"] is True
    events, _ = run(x, t0=30, an=an)
    assert names(events) == ["cry", "cry_stopped"]
    assert events[0]["push"] is False and "cooldown" in events[0]["message"]


@pytest.mark.parametrize(
    "x",
    [
        pink(30, -30),
        np.random.default_rng(0).standard_normal(30 * RATE).astype(np.float32) * 0.03,
        cry(30, -30, f0=600, harmonics=1),  # beep: a pure tone in the cry pitch range
        cry(30, -30, f0=1000),  # whistle
        cry(30, -30, f0=130),  # adult voice
        cry(30, -30, f0=210),
        breathing(30, 0.5),
    ],
    ids=["pink", "white", "beep", "whistle", "adult130", "adult210", "breathing"],
)
def test_not_cry(x):
    events, an = run(x, Settings(breathing=False))
    assert "cry" not in names(events)
    assert an.cry.fraction < 0.1


def test_crying_resets_breathing_window():
    x = np.concatenate([breathing(40, 0.5), cry(5, -50) + breathing(5, 0.5)])
    _, an = run(x)
    assert len(an.breath.env) == 0


def statuses(events):
    return [e for e in events if e["event"] == "status"]


def test_status_every_status_s():
    events, _ = run(pink(95, -60), Settings(breathing=False, cry=False), status=True)
    st = statuses(events)
    assert [e["t"] for e in st] == [30.5, 60.5, 90.5]
    assert all(e["push"] is False for e in st)
    assert st[0]["db"] == pytest.approx(-60, abs=5)  # one block of pink noise
    assert st[0]["db_avg"] == pytest.approx(-60, abs=1)
    assert st[0]["loud"] is False
    assert st[0]["crying"] is None and st[0]["breathing"] is None
    assert st[0]["last_event"] is None and st[0]["last_event_age_s"] is None
    assert st[0]["message"] == "%.0f dBFS, quiet" % st[0]["db_avg"]


def test_status_off():
    events, _ = run(pink(65, -60), Settings(status_s=0), status=True)
    assert statuses(events) == []


def test_status_reports_breathing_and_last_event():
    events, _ = run(breathing(95, 0.5), status=True)
    st = statuses(events)[-1]
    assert st["breathing"] == "normal"
    assert st["rate_bpm"] == pytest.approx(30, rel=0.1)
    assert {"periodicity", "depth", "band_db", "cry_fraction"} <= st.keys()
    assert st["crying"] is False
    assert st["last_event"] == "breathing_normal"
    normal = next(e for e in events if e["event"] == "breathing_normal")
    assert st["last_event_age_s"] == pytest.approx(st["t"] - normal["t"])
    assert st["message"].endswith(", quiet, breathing normal 30/min")


def test_status_loud_and_db_avg_window():
    x = np.concatenate([pink(20, -60, 1), pink(11, -20, 2)])
    events, _ = run(x, Settings(breathing=False, cry=False), status=True)
    (st,) = statuses(events)
    assert st["loud"] is True
    assert st["db_avg"] == pytest.approx(
        -20, abs=1
    )  # the quiet part is older than 10 s
    assert st["message"].endswith("dBFS, loud")


def test_status_crying_message():
    events, _ = run(cry(31, -30), Settings(breathing=False), status=True)
    (st,) = statuses(events)
    assert st["crying"] is True
    assert st["message"].endswith(", crying")


def test_status_timer_restarts_after_gap():
    an = Analyzer(Settings(breathing=False, cry=False), RATE)
    q = pink(25, -60)
    run(q, an=an)
    an.feed(100.5, q[:BLOCK], new_segment=True)
    events, _ = run(pink(35, -60), t0=100.5, an=an, status=True)
    assert [e["t"] for e in statuses(events)] == [130.5]
