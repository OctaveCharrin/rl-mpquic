"""
ITU-T G.1070 opinion model as a QoE **oracle** for reward calibration.

G.1070 ("Opinion model for video-telephony applications") estimates a Mean
Opinion Score (MOS, 1-5) for interactive videophone from network/application
parameters. We use it as the *ground-truth* oracle to calibrate the provisional
reward weights in :mod:`src.ns3env.qoe` (see ``scripts/calibrate_reward.py`` and
``docs/REWARD_TUNING.md`` §5B): the same regression-to-subjective-MOS procedure
that produced G.1070's own coefficients.

Scope actually used here (YAGNI):

* **Video quality ``Vq``** (Rec. clause 11.3, Eqs 11-16…11-21) — a MOS component
  driven by video bit rate, frame rate and packet loss. This is the scientifically
  strong part (cross-correlation ~0.95 vs. subjective MOS in the standard) and it
  carries the **bitrate/quality shape** and the **loss** sensitivity.
* **Multimedia integration ``MMq``** (clause 11.4, Eqs 11-22…11-27, Annex C
  coefficients) — folds in the **one-way delay** impairment, giving the oracle a
  latency sensitivity. G.1070 prices delay *only* here (clause 7), so it is needed
  to calibrate the latency weight. Audio is not modelled by this pipeline, so the
  speech quality ``Sq`` is held at a clean-audio constant (default 4.5, the max of
  Eq 11-6) and ``TS = TV = latency`` (media-sync term ``MS → 0`` when synchronized).

**Jitter is not an input anywhere in G.1070.** The oracle therefore accepts a
``jitter_ms`` argument but ignores it unless a non-degenerate jitter donor is
supplied (see :func:`build_composite_oracle`); the calibration pins ``c`` at the
standards ratio instead. The shipped learned surrogate ``reward_model.npz`` is
degenerate on its jitter axis, so the composite stays dormant until it is refit.

Since ``video_source`` fixes fps=30, ``Vq`` reduces to ``Vq(bitrate, loss)`` in
practice; frame rate is kept as a parameter for completeness.

Reference: Recommendation ITU-T G.1070 (06/2018),
``docs/T-REC-G.1070-201806.pdf``. Delay companion: ITU-T G.107 (referenced by
G.1070 clause 7 for pure-delay degradation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

__all__ = [
    "VideoCoeffs",
    "IntegrationCoeffs",
    "VIDEO_COEFF_PRESETS",
    "INTEGRATION_COEFF_PRESETS",
    "G1070Config",
    "G1070Oracle",
    "build_composite_oracle",
]

_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Coefficient tables (Rec. ITU-T G.1070 Annex B video, Annex C integration).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VideoCoeffs:
    """v1..v12 for the video quality estimation function (Annex B)."""

    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float


@dataclass(frozen=True)
class IntegrationCoeffs:
    """m1..m14 for the multimedia quality integration function (Annex C)."""

    m1: float
    m2: float
    m3: float
    m4: float
    m5: float
    m6: float
    m7: float
    m8: float
    m9: float
    m10: float
    m11: float
    m12: float
    m13: float
    m14: float


# Annex B video coefficients. Default targets a videoconferencing endpoint:
# H.264 baseline-profile VGA on a small (6-inch) screen (Table B.4, column #1) —
# calibrated for 8/15/30 fps and 0-3% loss. Other columns are provided as presets.
VIDEO_COEFF_PRESETS: Dict[str, VideoCoeffs] = {
    # Table B.4 #1 — H.264 BP, VGA, 6-inch screen (default).
    "h264_bp_vga_6in": VideoCoeffs(
        v1=6.743, v2=0.9998e-2, v3=3.051, v4=168.1, v5=1.766,
        v6=1.130, v7=18.340e-4, v8=1.232, v9=53.25,
        v10=3.353, v11=6.025, v12=80.752,
    ),
    # Table B.6 #1 — H.264 BP, VGA, 65-inch (large) screen.
    "h264_bp_vga_65in": VideoCoeffs(
        v1=5.643, v2=1.042e-2, v3=2.862, v4=178.2, v5=1.972,
        v6=1.263, v7=11.026e-4, v8=1.125, v9=49.34,
        v10=3.047, v11=5.824, v12=92.465,
    ),
    # Table B.2 #5 — H.264, VGA, 9.2-inch screen (2012 data; 400 kbit/s-2 Mbit/s).
    "h264_vga_9in": VideoCoeffs(
        v1=5.517, v2=1.29e-2, v3=3.459, v4=178.53, v5=1.02,
        v6=1.15, v7=3.55e-4, v8=0.114, v9=513.77,
        v10=0.736, v11=-6.451, v12=13.684,
    ),
}

# Annex C integration coefficients, keyed by video display size.
INTEGRATION_COEFF_PRESETS: Dict[str, IntegrationCoeffs] = {
    "4.2in": IntegrationCoeffs(
        m1=-4.457e-1, m2=-6.638e-1, m3=4.042e-1, m4=2.321,
        m5=-3.255e-1, m6=3.309e-1, m7=1.494e-1, m8=5.457e-1,
        m9=-3.235e-4, m10=3.915, m11=-1.377e-3, m12=0.000,
        m13=-1.095e-3, m14=0.000,
    ),
    "2.1in": IntegrationCoeffs(
        m1=-6.966e-1, m2=-8.127e-1, m3=4.562e-1, m4=3.003,
        m5=-1.638e-1, m6=3.626e-1, m7=1.291e-1, m8=5.456e-1,
        m9=-1.251e-4, m10=3.763, m11=-1.065e-3, m12=1.465e-2,
        m13=-1.002e-3, m14=0.000,
    ),
}


@dataclass
class G1070Config:
    """Selects the coefficient sets and the fixed-audio assumptions.

    ``video_preset`` / ``integration_preset`` name entries in the preset tables.
    ``fps`` is the encoder frame rate (30 in this pipeline). ``sq`` is the held
    clean-audio speech-quality MOS (4.5 = perfect narrowband, Eq 11-6 max), since
    this pipeline does not model an audio channel.
    """

    video_preset: str = "h264_bp_vga_6in"
    integration_preset: str = "4.2in"
    fps: float = 30.0
    sq: float = 4.5

    @property
    def video_coeffs(self) -> VideoCoeffs:
        return VIDEO_COEFF_PRESETS[self.video_preset]

    @property
    def integration_coeffs(self) -> IntegrationCoeffs:
        return INTEGRATION_COEFF_PRESETS[self.integration_preset]


# --------------------------------------------------------------------------- #
# Model equations.
# --------------------------------------------------------------------------- #


def video_quality(
    *, bitrate_kbps: float, fps: float, loss_frac: float, coeffs: VideoCoeffs
) -> float:
    """G.1070 video quality ``Vq`` (Eqs 11-16…11-21), MOS in ``[1, 5]``.

    ``loss_frac`` is a fraction in ``[0, 1]`` (converted to the model's percent
    ``PplV`` internally). Frame rate ``fps`` in fps, bit rate in kbit/s.
    """
    br = max(bitrate_kbps, _EPS)
    fr = max(fps, _EPS)
    pplv = 100.0 * min(1.0, max(0.0, loss_frac))  # fraction -> percent
    c = coeffs

    ofr = c.v1 + c.v2 * br                                    # optimal frame rate
    i_ofr = c.v3 * (1.0 - 1.0 / (1.0 + (br / c.v4) ** c.v5))  # max quality @ Br
    d_frv = max(c.v6 + c.v7 * br, _EPS)                       # frame-rate robustness
    i_coding = i_ofr * math.exp(
        -((math.log(fr) - math.log(ofr)) ** 2) / (2.0 * d_frv ** 2)
    )
    d_pplv = max(
        c.v10 + c.v11 * math.exp(-fr / c.v8) + c.v12 * math.exp(-br / c.v9), _EPS
    )                                                         # loss robustness
    vq = 1.0 + i_coding * math.exp(-pplv / d_pplv)
    return float(min(5.0, max(1.0, vq)))


def _media_sync(ts_ms: float, tv_ms: float, c: IntegrationCoeffs) -> float:
    """Audio-visual media-synchronization impairment ``MS`` (Eqs 11-26/11-27).

    Zero when speech and video delay are equal (our default TS == TV).
    """
    if ts_ms >= tv_ms:
        return min(c.m11 * (ts_ms - tv_ms) + c.m12, 0.0)
    return min(c.m13 * (ts_ms - tv_ms) + c.m14, 0.0)


def multimedia_quality(
    *, vq: float, sq: float, ts_ms: float, tv_ms: float, coeffs: IntegrationCoeffs
) -> float:
    """G.1070 multimedia quality ``MMq`` (Eqs 11-22…11-27), MOS in ``[1, 5]``.

    Note the term grouping (linear terms then the interaction term) — the naive
    reading of the OCR'd equations inverts monotonicity. Verified: MMq decreases
    monotonically with delay for the clean-audio operating regime used here.
    """
    c = coeffs
    mm_sv = c.m5 * sq + c.m6 * vq + c.m7 * (sq * vq) + c.m8       # audio-visual quality
    mm_sv = min(5.0, max(1.0, mm_sv))
    ad = c.m9 * (ts_ms + tv_ms) + c.m10                          # absolute delay
    ms = _media_sync(ts_ms, tv_ms, c)
    mm_t = max(ad + ms, 1.0)                                     # delay impairment factor
    mm_q = c.m1 * mm_sv + c.m2 * mm_t + c.m3 * (mm_sv * mm_t) + c.m4
    return float(min(5.0, max(1.0, mm_q)))


class G1070Oracle:
    """G.1070 MOS oracle: ``(bitrate, latency, loss) -> MOS in [1, 5]``.

    ``jitter_ms`` is accepted for a uniform oracle signature but **ignored** —
    G.1070 has no jitter input — unless a non-degenerate ``jitter_donor`` is wired
    via :func:`build_composite_oracle`.
    """

    def __init__(
        self,
        config: Optional[G1070Config] = None,
        *,
        jitter_donor: Optional[Callable[..., float]] = None,
    ):
        self.config = config or G1070Config()
        self._jitter_donor = jitter_donor

    def mos(
        self,
        *,
        bitrate_kbps: float,
        latency_ms: float,
        loss: float,
        jitter_ms: float = 0.0,
    ) -> float:
        cfg = self.config
        vq = video_quality(
            bitrate_kbps=bitrate_kbps, fps=cfg.fps, loss_frac=loss,
            coeffs=cfg.video_coeffs,
        )
        lat = max(0.0, latency_ms)
        mm = multimedia_quality(
            vq=vq, sq=cfg.sq, ts_ms=lat, tv_ms=lat, coeffs=cfg.integration_coeffs
        )
        if self._jitter_donor is not None:
            mm = _apply_jitter_impairment(
                mm, self._jitter_donor,
                bitrate_kbps=bitrate_kbps, latency_ms=lat, loss=loss, jitter_ms=jitter_ms,
            )
        return float(mm)


# --------------------------------------------------------------------------- #
# Dormant jitter-donor composite (auto-activates only on a non-degenerate model).
# --------------------------------------------------------------------------- #


def _apply_jitter_impairment(
    mos: float,
    donor: Callable[..., float],
    *,
    bitrate_kbps: float,
    latency_ms: float,
    loss: float,
    jitter_ms: float,
) -> float:
    """Graft the donor's *isolated* jitter degradation onto the G.1070 MOS.

    ``Δ = [VMAF(…, jitter=0) - VMAF(…, jitter=J)] / 100`` (a normalized, isolated
    fractional drop); ``MOS' = 1 + (MOS - 1)·(1 - Δ)``. Only the jitter *difference*
    is borrowed, never the donor's absolute level — so its (degenerate) bitrate/loss
    axes never touch the oracle.
    """
    v0 = donor(bitrate_kbps=bitrate_kbps, latency_ms=latency_ms, jitter_ms=0.0, loss=loss)
    vj = donor(bitrate_kbps=bitrate_kbps, latency_ms=latency_ms, jitter_ms=jitter_ms, loss=loss)
    delta = max(0.0, min(1.0, (v0 - vj) / 100.0))
    return 1.0 + (mos - 1.0) * (1.0 - delta)


def _donor_has_jitter_signal(
    donor: Callable[..., float],
    *,
    probes: Optional[List[float]] = None,
) -> bool:
    """True iff the donor's VMAF actually varies across jitter (non-degenerate)."""
    probes = probes or [0.0, 10.0, 40.0, 80.0]
    for b, l, d in ((1200.0, 0.0, 30.0), (800.0, 0.05, 80.0)):
        vals = [
            donor(bitrate_kbps=b, latency_ms=d, jitter_ms=j, loss=l) for j in probes
        ]
        if max(vals) - min(vals) > 1e-6:
            return True
    return False


def build_composite_oracle(
    config: Optional[G1070Config] = None,
    *,
    jitter_donor: Optional[Callable[..., float]] = None,
    load_default_donor: bool = False,
) -> G1070Oracle:
    """Build a G.1070 oracle, activating the jitter term only if a donor resolves it.

    Pass an explicit ``jitter_donor`` (a ``vmaf_fn``), or ``load_default_donor=True``
    to wire the repo's learned surrogate. If the chosen donor is degenerate on its
    jitter axis (the shipped ``reward_model.npz`` is), the jitter term stays dormant
    and the oracle behaves as plain G.1070 — so it "auto-activates on refit" with no
    code change.
    """
    donor = jitter_donor
    if donor is None and load_default_donor:
        from .learned_vmaf import load_learned_vmaf_fn

        donor = load_learned_vmaf_fn()
    if donor is not None and not _donor_has_jitter_signal(donor):
        donor = None  # dormant: no usable jitter signal
    return G1070Oracle(config, jitter_donor=donor)
