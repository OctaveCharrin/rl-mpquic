"""
WebRTC-like real-time video source model.

Describes how the encoder turns a *target bitrate* into a stream of variable-size
frames at a fixed frame rate. This mirrors the C++ generator in
``ns3/realtime_mpquic.cc`` (``RealtimeController::GenerateFrame``) so the mock and
NS-3 backends produce comparable frame sizes:

* a base size of ``bitrate / fps`` bytes per frame,
* periodic **I-frames** (every ``keyframe_interval`` frames) that are larger
  (intra-coded, no temporal prediction), and
* per-frame **size jitter** (P-frame content variability).

Real-time means frames are emitted on the wall clock (every ``1/fps`` seconds)
regardless of whether the previous frame finished delivering — so when the send
rate exceeds path capacity, frames queue and latency grows (the bufferbloat the
Transport agent must avoid).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["VideoSourceConfig", "frame_bytes"]

# Matches the C++ I-frame multiplier in RealtimeController::GenerateFrame.
KEYFRAME_MULTIPLIER: float = 2.5


@dataclass
class VideoSourceConfig:
    """Encoder parameters for the real-time source."""

    fps: float = 30.0
    min_bitrate_kbps: float = 300.0
    max_bitrate_kbps: float = 6000.0
    init_bitrate_kbps: float = 1500.0
    frame_size_jitter: float = 0.25  # relative +/- per-frame variability
    keyframe_interval: int = 30      # every Nth frame is an I-frame

    def clamp_bitrate(self, kbps: float) -> float:
        return float(min(self.max_bitrate_kbps, max(self.min_bitrate_kbps, kbps)))


def frame_bytes(
    bitrate_kbps: float,
    frame_idx: int,
    config: VideoSourceConfig,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Encoded size (bytes) of frame ``frame_idx`` at ``bitrate_kbps``.

    ``base = bitrate / fps`` bytes, scaled by the I-frame multiplier on keyframes
    and by a uniform jitter factor in ``[1 - j, 1 + j]``. Always >= 1 byte.
    """
    base = (bitrate_kbps * 1000.0 / 8.0) / config.fps
    kf = (
        KEYFRAME_MULTIPLIER
        if config.keyframe_interval > 0 and frame_idx % config.keyframe_interval == 0
        else 1.0
    )
    if rng is not None and config.frame_size_jitter > 0.0:
        jit = 1.0 + config.frame_size_jitter * (2.0 * rng.random() - 1.0)
    else:
        jit = 1.0
    return int(max(1, round(base * kf * jit)))
