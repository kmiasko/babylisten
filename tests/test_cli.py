import json
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def babylisten(*args, config):
    return subprocess.run(
        [sys.executable, "-m", "babylisten", *args, "--config", str(config)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """60 s of Opus/MKV stereo: breathing at 0.5 Hz, then a 4 s loud burst at 45 s."""
    p = tmp_path_factory.mktemp("clips") / "clip.mkv"
    gain = "if(between(t,45,49),1,0.02*(0.3+0.35*(1+sin(2*PI*0.5*t))))"
    ffmpeg(
        "-f", "lavfi", "-i", "anoisesrc=d=60:c=pink:a=0.5:r=48000",
        "-af", f"volume='{gain}':eval=frame,aformat=channel_layouts=stereo",
        "-c:a", "libopus", str(p),
    )  # fmt: skip
    return p


def lines(out, status=False):
    """Events on stdout; "status" lines only when status is set."""
    evs = [json.loads(line) for line in out.splitlines()]
    assert all(isinstance(e["event"], str) for e in evs)
    return [e for e in evs if status or e["event"] != "status"]


def test_file(clip, tmp_path):
    r = babylisten("file", str(clip), config=tmp_path / "none.json")
    assert r.returncode == 0, r.stderr
    evs = lines(r.stdout)
    names = [e["event"] for e in evs]
    assert names[:2] == ["breathing_normal", "noise"], names
    assert evs[0]["rate_bpm"] == pytest.approx(30, rel=0.1)
    assert evs[1]["push"] is True
    assert evs[1]["ts"] == pytest.approx(47, abs=1.5)


def test_file_status(clip, tmp_path):
    r = babylisten("file", str(clip), "--status-s", "20", config=tmp_path / "none.json")
    assert r.returncode == 0, r.stderr
    st = [e for e in lines(r.stdout, status=True) if e["event"] == "status"]
    assert len(st) in (2, 3), st
    assert st[0]["ts"] == pytest.approx(20, abs=1)
    assert st[0]["push"] is False and "dBFS" in st[0]["message"]


def test_file_without_audio(tmp_path):
    p = tmp_path / "video.mkv"
    ffmpeg("-f", "lavfi", "-i", "testsrc=d=1:s=64x64", "-c:v", "mpeg4", str(p))
    r = babylisten("file", str(p), config=tmp_path / "none.json")
    assert r.returncode == 1
    assert r.stdout == ""
    assert "no audio stream" in r.stderr


def test_calibrate(clip, tmp_path):
    cfg = tmp_path / "config.json"
    r = subprocess.run(
        [sys.executable, "-m", "babylisten", "calibrate", str(clip), "--config", str(cfg),
         "--min-breaths", "5"],
        check=False, capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert "saved" in r.stdout
    s = json.loads(cfg.read_text())
    assert -70 < s["threshold_db"] < -20
    assert s["baseline_db"] is not None
    assert s["heavy_bpm"] == pytest.approx(1.25 * 30, rel=0.1)


def test_file_cry(tmp_path):
    """Opus-encoded cry (450 Hz, 6 harmonics, 1 s units) over quiet pink noise."""
    p = tmp_path / "cry.mkv"
    tone = "+".join(f"sin({h}*2*PI*450*t)/{h}" for h in range(1, 7))
    env = "if(lt(mod(t,1.3),1),sin(PI*mod(t,1.3)),0)*between(t,5,15)"
    ffmpeg(
        "-f", "lavfi", "-i", f"aevalsrc='0.02*({tone})*{env}':s=48000:d=40",
        "-f", "lavfi", "-i", "anoisesrc=d=40:c=pink:a=0.003:r=48000",
        "-filter_complex", "amix=inputs=2:normalize=0", "-c:a", "libopus", str(p),
    )  # fmt: skip
    r = babylisten("file", str(p), "--no-breathing", config=tmp_path / "none.json")
    assert r.returncode == 0, r.stderr
    evs = lines(r.stdout)
    assert [e["event"] for e in evs] == ["cry", "cry_stopped"], evs
    assert evs[0]["push"] is True
    assert evs[0]["f0_hz"] == pytest.approx(450, rel=0.1)
