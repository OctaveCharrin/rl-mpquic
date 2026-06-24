"""
QoS -> VMAF reward, grounded in real WebRTC VMAF measurements.

This is a **self-contained, copy-paste-portable** module: its only dependency is
NumPy. Drop this file (plus its ``reward_model.npz``) into any project — e.g.
``rl-mpquic`` — and import :class:`QoSVmafReward` directly. It has no dependency
on the WebRTC-QoE-Data-Generator package that produced the model.

The model is a multilinear interpolant over a regular grid of measured
experiments, fitted by ``pipeline/surrogate.py``. It maps

    (bitrate_kbps, loss_pct, delay_ms, jitter_ms) -> VMAF in [0, 100]

Inputs are clamped to the measured grid box before evaluation, so out-of-range
queries saturate at the nearest measured boundary rather than extrapolating
wildly.

Usage (matching the rl-mpquic call site exactly):

    from qos_vmaf_reward import QoSVmafReward

    reward = QoSVmafReward()  # loads reward_model.npz next to this file
    score = reward.vmaf(
        bitrate_kbps=1200.0,   # App agent's target/encoding send rate
        loss_pct=2.5,          # effective loss at the reassembly point, percent
        delay_ms=30.0,         # one-way delay: pass aggregate_RTT / 2
        jitter_ms=8.0,         # interarrival jitter / latency std-dev, ms
    )

Units / argument notes for callers in rl-mpquic:
  - ``bitrate_kbps``: App action ``target_bitrate_kbps``.
  - ``loss_pct``: App state ``packet_loss`` in **percent [0, 100]**. If your
    simulator reports a fraction in [0, 1], multiply by 100 before calling.
  - ``delay_ms``: ONE-WAY delay. The grid's delay axis is one-way (netem), while
    rl-mpquic's transport state reports an aggregate RTT, so pass ``rtt / 2``.
  - ``jitter_ms``: App state ``jitter_ms`` (RFC 3550-style interarrival jitter).

For fast training loops use :meth:`vmaf_batch`, which takes an ``(n, 4)`` array
with columns in the fixed order ``[bitrate_kbps, loss_pct, delay_ms, jitter_ms]``.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np

__all__ = ["QoSVmafReward", "FEATURE_COLUMNS"]

# Fixed column order for the batch API and the serialized model.
FEATURE_COLUMNS = ["bitrate_kbps", "loss_pct", "delay_ms", "jitter_ms"]

# Default model file shipped alongside this module.
_DEFAULT_MODEL_NAME = "reward_model.npz"
# Environment variable to override the model path without code changes.
_MODEL_ENV_VAR = "QOS_VMAF_MODEL"


class QoSVmafReward:
    """Real-VMAF-grounded QoS reward backed by a multilinear grid surrogate."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Args:
            model_path: Path to ``reward_model.npz`` or ``reward_model.pkl``.
                If None, resolves in order: ``$QOS_VMAF_MODEL`` env var,
                ``reward_model.npz`` next to this file, then
                ``../output/reward_model.npz`` (the producing repo's output dir).
        """
        path = self._resolve_path(model_path)
        self._params = self._load(path)
        self.model_path = path
        self._columns = list(self._params["columns"])
        self._axes = self._params["axes"]
        self._table = self._params["table"]
        self._bmin = self._params["bounds_min"]
        self._bmax = self._params["bounds_max"]

    # ---- Public API --------------------------------------------------------

    def vmaf(
        self,
        *,
        bitrate_kbps: float,
        loss_pct: float,
        delay_ms: float,
        jitter_ms: float,
    ) -> float:
        """Predict VMAF (0-100) for a single condition. Keyword args only.

        ``delay_ms`` is ONE-WAY delay; callers reporting RTT must pass ``rtt/2``.
        Inputs are clamped to the fitted grid box.
        """
        x = np.array(
            [[bitrate_kbps, loss_pct, delay_ms, jitter_ms]], dtype=np.float64
        )
        return float(self._predict(x)[0])

    def vmaf_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorized prediction for many conditions.

        Args:
            X: ``(n, 4)`` array with columns
               ``[bitrate_kbps, loss_pct, delay_ms, jitter_ms]``.

        Returns:
            ``(n,)`` array of VMAF values in [0, 100].
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[1] != 4:
            raise ValueError(
                f"Expected (n, 4) with columns {FEATURE_COLUMNS}, got {X.shape}"
            )
        return self._predict(X)

    @property
    def bounds(self) -> dict:
        """Measured grid box per feature: {name: (min, max)} (pre-clamp range)."""
        return {
            name: (float(self._params["bounds_min"][i]),
                   float(self._params["bounds_max"][i]))
            for i, name in enumerate(self._columns)
        }

    # ---- Multilinear evaluation (pure numpy; mirrors pipeline/surrogate.py) -

    def _predict(self, X: np.ndarray) -> np.ndarray:
        axes = self._axes
        table = self._table
        d = len(axes)
        m = X.shape[0]

        lo = np.zeros((m, d), dtype=np.intp)
        frac = np.zeros((m, d), dtype=np.float64)
        has_upper = np.zeros(d, dtype=bool)
        for j, axis in enumerate(axes):
            x = np.clip(X[:, j], axis[0], axis[-1])
            if axis.size < 2:
                continue
            has_upper[j] = True
            idx = np.clip(np.searchsorted(axis, x, side="right") - 1, 0, axis.size - 2)
            lo[:, j] = idx
            frac[:, j] = (x - axis[idx]) / (axis[idx + 1] - axis[idx])

        out = np.zeros(m, dtype=np.float64)
        for corner in range(1 << d):
            w = np.ones(m, dtype=np.float64)
            index = []
            valid = True
            for j in range(d):
                bit = (corner >> j) & 1
                if bit and not has_upper[j]:
                    valid = False
                    break
                w *= frac[:, j] if bit else (1.0 - frac[:, j])
                index.append(lo[:, j] + bit)
            if valid:
                out += w * table[tuple(index)]
        return np.clip(out, 0.0, 100.0)

    # ---- Loading -----------------------------------------------------------

    @staticmethod
    def _resolve_path(model_path: Optional[Union[str, Path]]) -> Path:
        if model_path is not None:
            return Path(model_path)
        env = os.environ.get(_MODEL_ENV_VAR)
        if env:
            return Path(env)
        here = Path(__file__).resolve().parent
        candidates = [
            here / _DEFAULT_MODEL_NAME,
            here / "reward_model.pkl",
            here.parent / "output" / _DEFAULT_MODEL_NAME,
        ]
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(
            f"No reward model found. Looked for {_MODEL_ENV_VAR}, "
            f"{candidates[0]}, {candidates[2]}. Fit one with "
            f"`python run.py fit-reward` and copy reward_model.npz next to this "
            f"module."
        )

    @staticmethod
    def _load(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Reward model not found: {path}")
        if path.suffix == ".pkl":
            with open(path, "rb") as fh:
                params = pickle.load(fh)
            params["axes"] = [np.asarray(a, dtype=np.float64) for a in params["axes"]]
            params["table"] = np.asarray(params["table"], dtype=np.float64)
        else:
            data = np.load(path, allow_pickle=False)
            n_dims = int(data["n_dims"])
            params = {
                "method": str(data["method"]),
                "columns": [str(c) for c in data["columns"]],
                "axes": [data[f"axis_{j}"].astype(np.float64) for j in range(n_dims)],
                "table": data["table"].astype(np.float64),
                "bounds_min": data["bounds_min"].astype(np.float64),
                "bounds_max": data["bounds_max"].astype(np.float64),
            }
        if params.get("method") != "multilinear":
            raise ValueError(f"Unsupported model method: {params.get('method')}")
        return params


if __name__ == "__main__":
    # Tiny smoke test / demo when a model is present.
    r = QoSVmafReward()
    print(f"Loaded model: {r.model_path}")
    print(f"Grid bounds: {r.bounds}")
    demo = r.vmaf(bitrate_kbps=1000, loss_pct=5, delay_ms=50, jitter_ms=10)
    print(f"VMAF(1000kbps, 5% loss, 50ms, 10ms jitter) = {demo:.1f}")
