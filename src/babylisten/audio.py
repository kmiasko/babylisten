"""Audio reader for files and UDP/RTSP streams (PyAV): mono float32 blocks at RATE, with reconnect.

A thin layer around the detector: it decodes, downmixes, resamples and cuts fixed-size blocks. It never
exits on a stream that is not there yet or that stops; it waits and opens it again.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np

log = logging.getLogger(__name__)

RATE = 16000  # analysis sample rate; the breathing band ends at 1 kHz
BLOCK_S = 0.5
STREAM_OPTIONS = {
    "overrun_nonfatal": "1",  # as imcam gives babywatch
    "fifo_size": "50000",  # in 188-byte units, ~9 MB
    "timeout": "5000000",  # us without data before a read fails -> reconnect
}


class NoAudio(Exception):
    """The input opened but has no audio stream."""


def is_stream(src: str) -> bool:
    return "://" in src


def _open(src: str, stream: bool):
    if stream:
        c = av.open(src, options=STREAM_OPTIONS, timeout=(10.0, 10.0))
    else:
        c = av.open(src)
    if not c.streams.audio:
        c.close()
        raise NoAudio(f"{src}: no audio stream")
    return c, c.streams.audio[0]


def _decode(container, stream) -> Iterator[np.ndarray]:
    """Mono float32 sample arrays at RATE (channels are averaged)."""
    rs = av.AudioResampler(format="fltp", rate=RATE)
    for frame in container.decode(stream):
        for out in rs.resample(frame):
            yield out.to_ndarray().mean(axis=0, dtype=np.float32)
    for out in rs.resample(None):
        yield out.to_ndarray().mean(axis=0, dtype=np.float32)


def iter_blocks(
    src: str | Path,
    block_s: float = BLOCK_S,
    realtime: bool = False,
    reconnect: bool = True,
) -> Iterator[tuple[float, np.ndarray, bool]]:
    """Yield (t, samples, new_segment) for consecutive blocks of block_s seconds.

    t is the time of the block's end in seconds. For files it is media time. For streams it is monotonic
    time: within a connection it advances by block_s per block, and a reconnect starts at the current
    clock, so gaps count. new_segment is True on the first block after each (re)open.
    realtime=True paces a file at its native speed.
    """
    src = str(src)
    stream = is_stream(src)
    n_block = round(block_s * RATE)
    backoff = 1.0
    while True:
        try:
            container, astream = _open(src, stream)
        except (av.error.FFmpegError, NoAudio) as e:
            if not (stream and reconnect):
                raise
            log.warning("open failed (%s); retry in %.0fs", e, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue
        log.info("reading %s (%s)", src, astream.codec_context.name)
        backoff = 1.0
        t0 = time.monotonic() if stream else None
        new = True
        n = 0  # samples yielded in this segment
        buf = np.zeros(0, np.float32)
        wall0 = time.monotonic()
        try:
            for x in _decode(container, astream):
                if t0 is None:  # file: start at the first frame's media time
                    st = astream.start_time
                    t0 = float(st * astream.time_base) if st is not None else 0.0
                buf = np.concatenate([buf, x]) if len(buf) else x
                while len(buf) >= n_block:
                    block, buf = buf[:n_block], buf[n_block:]
                    n += n_block
                    t = t0 + n / RATE
                    if realtime and not stream:
                        lag = (t - t0) - (time.monotonic() - wall0)
                        if lag > 0:
                            time.sleep(lag)
                    yield t, block, new
                    new = False
        except av.error.FFmpegError as e:
            if not (stream and reconnect):
                raise
            log.warning("stream error (%s); reconnecting", e)
        finally:
            container.close()
        if not (stream and reconnect):
            return
        log.warning("stream ended; reconnecting in %.0fs", backoff)
        time.sleep(backoff)
