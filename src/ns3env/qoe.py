"""
Real-time video Quality-of-Experience (QoE) reward.

Quality is measured with **VMAF** (0-100), not raw bitrate, because the
rate->quality relationship is concave/saturating: equal bitrate steps are *not*
equal perceptual steps. We map bitrate to VMAF with a logarithmic curve anchored
to the endpoints Netflix describes (low-quality encode ~25, high-quality ~92+).

For real-time conferencing the reward then trades quality against the live
network penalties — end-to-end latency, jitter (inter-frame delay variation) and
loss (here, frames that miss their playout deadline):

    QoE = a*VMAF(bitrate) - b*latency - c*jitter - d*loss              (App agent)

with latency/jitter normalized to [0, ~1] by configurable scales and loss in
[0, 1]. The VMAF term is pluggable (``compute_qoe_reward(..., vmaf_fn=…)``): with
a *learned* QoS->VMAF model that already folds loss into the quality score, the
explicit ``- d*loss`` term is dropped to avoid double-counting, leaving
``a*VMAF(bitrate, loss, …) - b*latency - c*jitter`` (see ``compute_qoe_reward``).

The Transport agent, which only moves bytes across paths (it does not set the
bitrate), is rewarded for *delivering* frames cheaply:

    R_transport = (1 - loss) - b*latency - c*jitter                    (Transport)

Reference: Zhi Li et al., "VMAF: The Journey Continues," Netflix Tech Blog, 2018.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

__all__ = [
    "VMAF_MAX",
    "vmaf_for_kbps",
    "VmafFn",
    "QoEWeights",
    "compute_qoe_reward",
    "compute_transport_reward",
    "qoe_components",
]

# A pluggable VMAF scorer. The default maps bitrate alone via the log curve
# (``vmaf_for_kbps``); a learned model (see ``learned_vmaf.load_learned_vmaf_fn``)
# additionally consumes the live latency/jitter/loss. Keyword-only so callers can
# pass exactly the signals a given model needs; returns VMAF in [0, 100].
VmafFn = Callable[..., float]


def _score_vmaf(
    vmaf_fn: Optional[VmafFn],
    *,
    bitrate_kbps: float,
    latency_ms: float,
    jitter_ms: float,
    loss: float,
) -> float:
    """VMAF via the supplied scorer, or the default bitrate-only log curve."""
    if vmaf_fn is None:
        return vmaf_for_kbps(bitrate_kbps)
    return float(
        vmaf_fn(
            bitrate_kbps=bitrate_kbps,
            latency_ms=latency_ms,
            jitter_ms=jitter_ms,
            loss=loss,
        )
    )

VMAF_MAX: float = 100.0
# (bitrate_kbps, VMAF) anchors defining the log rate-quality curve. A documented
# stand-in for measured per-content VMAF, following the Netflix description.
_ANCHOR_LOW: Tuple[float, float] = (300.0, 25.0)
_ANCHOR_HIGH: Tuple[float, float] = (4300.0, 92.0)


def vmaf_for_kbps(
    bitrate_kbps: float,
    *,
    anchor_low: Tuple[float, float] = _ANCHOR_LOW,
    anchor_high: Tuple[float, float] = _ANCHOR_HIGH,
) -> float:
    """Map a bitrate (kbps) to a VMAF score in ``[0, 100]`` (concave/log curve).

    ``VMAF(R) = a*ln(R) + b`` fit through the two anchors, clipped to ``[0, 100]``.
    Monotonically increasing and concave, capturing diminishing returns at high
    bitrate (saturation toward 100).
    """
    (r0, v0), (r1, v1) = anchor_low, anchor_high
    a = (v1 - v0) / (math.log(r1) - math.log(r0))
    b = v0 - a * math.log(r0)
    v = a * math.log(max(bitrate_kbps, 1e-6)) + b
    return float(min(VMAF_MAX, max(0.0, v)))


@dataclass
class QoEWeights:
    """Weights and normalizers for the QoE reward.

    ``a_quality`` scales perceptual quality (VMAF/100); ``b_latency`` and
    ``c_jitter`` penalize latency/jitter after normalizing by ``*_norm_ms``;
    ``d_loss`` penalizes the deadline-miss (loss) fraction. Defaults are balanced
    so a fully congested frame (high latency + loss) roughly cancels top quality.

    ``e_util`` adds a *utilization* reward for delivered bitrate, normalized by
    ``util_norm_kbps`` (the quality knee) and gated by ``(1 - loss)`` so it only
    pays for bits that land on time. It is **only applied with a learned VMAF
    scorer**, whose grid is nearly bitrate-flat; it restores the rate-control
    gradient (more delivered bits is good) that the surrogate erases, so the agent
    must aggregate paths to earn it without eating latency/loss. The default
    log-curve VMAF is already bitrate-sensitive, so the term is skipped there.
    Default ``e_util = 0.0`` keeps it off unless explicitly enabled in config.
    """

    a_quality: float = 1.0
    b_latency: float = 0.5
    c_jitter: float = 0.5
    d_loss: float = 1.0
    latency_norm_ms: float = 200.0
    jitter_norm_ms: float = 50.0
    e_util: float = 0.0
    util_norm_kbps: float = 3300.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]] = None) -> "QoEWeights":
        base = cls()
        if not data:
            return base
        for key in (
            "a_quality",
            "b_latency",
            "c_jitter",
            "d_loss",
            "latency_norm_ms",
            "jitter_norm_ms",
            "e_util",
            "util_norm_kbps",
        ):
            if key in data:
                setattr(base, key, float(data[key]))
        return base


def compute_qoe_reward(
    *,
    bitrate_kbps: float,
    latency_ms: float,
    jitter_ms: float,
    loss: float,
    weights: Optional[QoEWeights] = None,
    vmaf_fn: Optional[VmafFn] = None,
) -> float:
    """App-agent QoE reward.

    Default (log-curve) form::

        a*VMAF(bitrate) - b*latency - c*jitter - d*loss

    Quality is ``VMAF/100`` in ``[0, 1]``; latency/jitter are normalized by the
    weight's ``*_norm_ms`` (then soft-capped at 2x to bound the penalty); ``loss``
    is a fraction in ``[0, 1]``. Result is clipped to ``[-2, 1]``.

    ``vmaf_fn`` selects the quality model:

    * **None (default)** — the bitrate-only log curve ``vmaf_for_kbps``; the full
      ``- d*loss`` penalty applies, since this VMAF ignores loss.
    * **Learned scorer** — a QoS->VMAF model that *already folds loss into the
      quality score*. The explicit ``- d*loss`` penalty is therefore **dropped**
      to avoid double-counting loss; the reward becomes::

          a*VMAF(bitrate, loss, …) - b*latency - c*jitter

      The latency/jitter penalties are intentionally kept: the current learned
      model does not separate those out (its delay/jitter grid axes are
      degenerate), so they must remain priced explicitly. A ``e_util`` utilization
      reward (delivered-bitrate, see :class:`QoEWeights`) is also added here to
      restore the bitrate gradient the surrogate erases.
    """
    w = weights or QoEWeights()
    q = _score_vmaf(
        vmaf_fn,
        bitrate_kbps=bitrate_kbps,
        latency_ms=latency_ms,
        jitter_ms=jitter_ms,
        loss=loss,
    ) / VMAF_MAX
    lat = min(2.0, max(0.0, latency_ms) / max(w.latency_norm_ms, 1e-6))
    jit = min(2.0, max(0.0, jitter_ms) / max(w.jitter_norm_ms, 1e-6))
    # A learned VMAF already reflects loss, so its explicit penalty is dropped
    # (double-count); the default bitrate-only curve does not, so it is kept.
    los = min(1.0, max(0.0, loss))
    loss_penalty = 0.0 if vmaf_fn is not None else w.d_loss * los
    # Utilization reward: only with a (bitrate-flat) learned VMAF, and gated by
    # (1 - loss) so it credits delivered, on-time bits — not over-sending.
    util_reward = 0.0
    if vmaf_fn is not None and w.e_util > 0.0:
        util = min(1.0, max(0.0, bitrate_kbps) / max(w.util_norm_kbps, 1e-6))
        util_reward = w.e_util * (1.0 - los) * util
    r = w.a_quality * q - w.b_latency * lat - w.c_jitter * jit - loss_penalty + util_reward
    return float(max(-2.0, min(1.0, r)))


def compute_transport_reward(
    *,
    latency_ms: float,
    jitter_ms: float,
    loss: float,
    weights: Optional[QoEWeights] = None,
) -> float:
    """Transport-agent reward: deliver frames on time and cheaply.

    ``(1 - loss) - b*latency - c*jitter`` — high when the chosen split lands the
    frame quickly and intact, negative when it is late or lost. The bitrate
    (quality) lever belongs to the App agent, so it is intentionally excluded.
    Clipped to ``[-2, 1]``.
    """
    w = weights or QoEWeights()
    lat = min(2.0, max(0.0, latency_ms) / max(w.latency_norm_ms, 1e-6))
    jit = min(2.0, max(0.0, jitter_ms) / max(w.jitter_norm_ms, 1e-6))
    los = min(1.0, max(0.0, loss))
    r = (1.0 - los) - w.b_latency * lat - w.c_jitter * jit
    return float(max(-2.0, min(1.0, r)))


def qoe_components(
    *,
    bitrate_kbps: float,
    latency_ms: float,
    jitter_ms: float,
    loss: float,
    vmaf_fn: Optional[VmafFn] = None,
) -> Dict[str, float]:
    """Unweighted QoE terms for logging/inspection.

    The logged ``vmaf`` uses the same scorer (``vmaf_fn``) as the reward so logs
    match the objective being optimized.
    """
    return {
        "vmaf": _score_vmaf(
            vmaf_fn,
            bitrate_kbps=bitrate_kbps,
            latency_ms=latency_ms,
            jitter_ms=jitter_ms,
            loss=loss,
        ),
        "latency_ms": float(latency_ms),
        "jitter_ms": float(jitter_ms),
        "loss": float(loss),
    }
