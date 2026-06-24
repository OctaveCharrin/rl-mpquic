"""
Adapter from the WebRTC-grounded learned VMAF surrogate to the rl-mpquic reward.

The model itself (:mod:`src.ns3env.qos_vmaf_reward`, vendored verbatim from the
``WebRTC-QoE-Data-Generator`` sibling project together with ``reward_model.npz``)
is a multilinear interpolant over **real** WebRTC VMAF measurements mapping

    (bitrate_kbps, loss_pct, delay_ms, jitter_ms) -> VMAF in [0, 100].

This module exposes :func:`load_learned_vmaf_fn`, which returns a callable with
the signature the QoE reward expects -- ``vmaf_fn(*, bitrate_kbps, latency_ms,
jitter_ms, loss) -> float`` -- handling the unit translation between rl-mpquic's
internal conventions and the model's grid axes:

  * ``loss`` is a **fraction** in [0, 1] here; the model wants **percent** -> x100.
  * ``latency_ms`` is rl-mpquic's per-frame **one-way** completion latency, which
    maps directly onto the model's one-way ``delay_ms`` netem axis (no /2).
  * ``jitter_ms`` passes through unchanged.

Inputs outside the fitted grid box are clamped by the model (it saturates at the
nearest measured boundary rather than extrapolating). Note the shipped model's
bitrate axis tops out around 2500 kbps and its delay/jitter axes may be
degenerate (single measured point), so with the current ``reward_model.npz`` the
score effectively varies over bitrate and loss; a richer refit activates the
delay/jitter axes without any change here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union

# A VMAF scorer: keyword-only, returns VMAF in [0, 100]. Mirrors the optional
# ``vmaf_fn`` hook in :mod:`src.ns3env.qoe`.
VmafFn = Callable[..., float]


def load_learned_vmaf_fn(model_path: Optional[Union[str, Path]] = None) -> VmafFn:
    """Build a ``vmaf_fn`` backed by the learned QoS->VMAF surrogate.

    Args:
        model_path: Optional explicit path to ``reward_model.npz``/``.pkl``. If
            None, the model resolves next to :mod:`src.ns3env.qos_vmaf_reward`
            (the vendored ``reward_model.npz``) or via ``$QOS_VMAF_MODEL``.

    Returns:
        A callable ``vmaf_fn(*, bitrate_kbps, latency_ms, jitter_ms, loss)`` that
        returns a VMAF score in ``[0, 100]``.
    """
    from .qos_vmaf_reward import QoSVmafReward

    model = QoSVmafReward(model_path)

    def vmaf_fn(
        *,
        bitrate_kbps: float,
        latency_ms: float = 0.0,
        jitter_ms: float = 0.0,
        loss: float = 0.0,
    ) -> float:
        return model.vmaf(
            bitrate_kbps=float(bitrate_kbps),
            loss_pct=100.0 * max(0.0, min(1.0, float(loss))),  # fraction -> percent
            delay_ms=max(0.0, float(latency_ms)),              # one-way; maps directly
            jitter_ms=max(0.0, float(jitter_ms)),
        )

    return vmaf_fn
