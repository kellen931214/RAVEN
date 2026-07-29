"""Frozen official HSQR (SFWMark) reference used by the parity tests.

Independent transcription of the official HSQR specification frozen in Issue #5
and in the official repository

    https://github.com/thomas11809/SFWMark
    commit 78666128b44614a0cc471993649e3132d5dddfcb
    (``src/generate.py`` / ``src/detect.py`` / ``src/utils.py``)

Nothing here imports ``utils.wm.hsqr_provider``: the point is to re-derive the
official behaviour from the specification so an element-wise comparison against
the provider is meaningful. Where the specification fixes a concrete value, the
expected SHA-256 digests below freeze it so a change in the ``qrcode`` library,
in the key derivation or in the provider is caught rather than silently
absorbed.

Scope of the parity claim
-------------------------
These fixtures cover the *static* watermark construction and the detector math
(key book, payload, QR pattern, injection, center FFT, complex L1, score sign).
They are derived from the frozen written specification and the ``qrcode``
package, **not** from tensors downloaded from the official repository, and they
say nothing about diffusion-model inversion parity. See
``utils/wm/sfw_inversion.SFW_INVERSION_PARITY_STATUS``.
"""

from __future__ import annotations

import hashlib

import numpy as np
import qrcode
import torch


OFFICIAL_SFWMARK_COMMIT = "78666128b44614a0cc471993649e3132d5dddfcb"

OFFICIAL_BASE_KEY_SEED = 7433
OFFICIAL_CAPACITY = 2048
OFFICIAL_QR_VERSION = 1
OFFICIAL_BOX_SIZE = 2
OFFICIAL_BORDER = 0
OFFICIAL_ERROR_CORRECTION = qrcode.constants.ERROR_CORRECT_H
OFFICIAL_QR_SIZE = 42
OFFICIAL_CENTER_SLICE = (10, 54)
OFFICIAL_WM_CHANNEL = 3
OFFICIAL_DELTA = 0
OFFICIAL_TARGET_MAGNITUDE = 45.0

#: Frozen expected values for the four required fixture keys (first, second, a
#: middle key and the last). ``pattern_sha256`` is the digest of the boolean
#: ``(1, 42, 42)`` pattern under the repository's canonical tensor hashing.
OFFICIAL_KEY_FIXTURES = {
    0: {
        "key_seed": 7433,
        "payload": "HSQR7433",
        "pattern_sha256": "4fb8b70ec30dd568ca829cc623a4755860658e2a53a82070d0707df33d678d07",
    },
    1: {
        "key_seed": 7434,
        "payload": "HSQR7434",
        "pattern_sha256": "a57eb68242291614e510b8d99172df9338045a2dda06e3ac212d756f50888ce7",
    },
    1024: {
        "key_seed": 8457,
        "payload": "HSQR8457",
        "pattern_sha256": "404842532c72f6b55c554ca77378fdfbdddc0ae78fa130d6f8c70a93e5b71fbb",
    },
    2047: {
        "key_seed": 9480,
        "payload": "HSQR9480",
        "pattern_sha256": "a37f0bed123aa1ea44773f146a576dc5f37206cefc176e43f68f7feb37b5e0ce",
    },
}


def official_key_seed(key_index: int, base_key_seed: int = OFFICIAL_BASE_KEY_SEED) -> int:
    """Official key seed: ``base + i``."""
    return base_key_seed + key_index


def official_payload(key_index: int, base_key_seed: int = OFFICIAL_BASE_KEY_SEED) -> str:
    """Official QR payload: ``HSQR{key_seed % 10000}``."""
    return f"HSQR{official_key_seed(key_index, base_key_seed) % 10000}"


def official_qr_bool(data: str) -> torch.Tensor:
    """Official QR profile: version 1, box size 2, border 0, EC H -> 42x42 bool."""
    code = qrcode.QRCode(
        version=OFFICIAL_QR_VERSION,
        box_size=OFFICIAL_BOX_SIZE,
        border=OFFICIAL_BORDER,
        error_correction=OFFICIAL_ERROR_CORRECTION,
    )
    code.add_data(data)
    code.make(fit=True)
    assert code.version == OFFICIAL_QR_VERSION, (
        f"payload {data!r} does not fit official QR version {OFFICIAL_QR_VERSION}"
    )
    array = np.array(code.make_image(fill_color="black", back_color="white"))
    assert array.shape == (OFFICIAL_QR_SIZE, OFFICIAL_QR_SIZE), array.shape
    return torch.from_numpy(array).clone().detach()


def official_pattern(key_index: int, base_key_seed: int = OFFICIAL_BASE_KEY_SEED) -> torch.Tensor:
    """Official ``(1, 42, 42)`` boolean pattern for one key index."""
    return official_qr_bool(official_payload(key_index, base_key_seed)).unsqueeze(0)


def sha256_tensor(tensor: torch.Tensor) -> str:
    """Same canonical tensor digest the repository uses (dtype + shape + bytes)."""
    tensor = tensor.detach().cpu().contiguous()
    header = f"torch|{tensor.dtype}|{tuple(tensor.shape)}|".encode("utf-8")
    return hashlib.sha256(header + tensor.numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Official injection / detection math
# ---------------------------------------------------------------------------

def _rfft(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.rfft2(x, dim=(-2, -1)), dim=-2)


def _irfft(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.irfft2(
        torch.fft.ifftshift(x, dim=-2), dim=(-2, -1), s=(x.shape[-2], x.shape[-2])
    )


def _fft(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fft2(x), dim=(-1, -2))


def official_inject(latent: torch.Tensor, pattern: torch.Tensor,
                    delta: int = OFFICIAL_DELTA) -> torch.Tensor:
    """Official center-region injection.

    The 42x42 QR splits into two 42x21 halves; the left half controls the sign of
    the real RFFT coefficients of the 44x44 center latent and the right half the
    imaginary ones, using ``abs(value) + delta`` / ``-abs(value) - delta``.
    """
    start, end = OFFICIAL_CENTER_SLICE
    center_slice = (slice(None), slice(None), slice(start, end), slice(start, end))
    qr = pattern.unsqueeze(0)  # (1, c_wm, 42, 42)
    half = (OFFICIAL_QR_SIZE + 1) // 2
    left, right = qr[..., :half], qr[..., half:]

    latent = latent.clone()
    center_rfft = _rfft(latent[center_slice])
    real, imag = center_rfft.real, center_rfft.imag
    target_slice = (
        slice(None), [OFFICIAL_WM_CHANNEL],
        slice(1, 1 + OFFICIAL_QR_SIZE), slice(1, 1 + half),
    )
    real[target_slice] = torch.where(
        left, real[target_slice].abs() + delta, -real[target_slice].abs() - delta
    )
    imag[target_slice] = torch.where(
        right, imag[target_slice].abs() + delta, -imag[target_slice].abs() - delta
    )
    latent[center_slice] = _irfft(torch.complex(real, imag))
    return latent


def official_target(pattern: torch.Tensor) -> torch.Tensor:
    """Official complex 42x21 detector target (booleans -> ±45)."""
    half = (OFFICIAL_QR_SIZE + 1) // 2
    values = torch.where(
        pattern,
        torch.tensor(OFFICIAL_TARGET_MAGNITUDE),
        torch.tensor(-OFFICIAL_TARGET_MAGNITUDE),
    ).to(torch.float32)
    return torch.complex(values[0, :, :half], values[0, :, half:])


def official_l1_distance(latent: torch.Tensor, pattern: torch.Tensor) -> float:
    """Official mean complex L1 distance for a single-item latent."""
    start, end = OFFICIAL_CENTER_SLICE
    center_slice = (slice(None), slice(None), slice(start, end), slice(start, end))
    spectrum = torch.zeros_like(latent, dtype=torch.complex64)
    spectrum[center_slice] = _fft(latent[center_slice])

    half = (OFFICIAL_QR_SIZE + 1) // 2
    center_row = spectrum.shape[-2] // 2
    row_start = start + 1
    col_start = center_row + 1
    window = spectrum[
        0, OFFICIAL_WM_CHANNEL,
        row_start:row_start + OFFICIAL_QR_SIZE,
        col_start:col_start + half,
    ]
    return float(torch.abs(official_target(pattern) - window).mean())


def official_score(distance: float) -> float:
    """Official canonical ROC score."""
    return -float(distance)
