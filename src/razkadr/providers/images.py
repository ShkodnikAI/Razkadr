"""Image preparation for embedding/VLM adapters (naryad 038).

Images are downscaled to a configurable max dimension and re-encoded as
JPEG with a hard byte limit before they leave the machine: the provider
bills for transmitted pixels, so oversized sources must never reach the
API (auditor criterion: "изображения не отправляются в избыточном
разрешении"). Deterministic: same input -> same output bytes.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from razkadr.providers.base import ProviderError

DEFAULT_MAX_DIM = 1024  # longest side, px
DEFAULT_MAX_BYTES = 4_000_000  # hard request-size cap per image
QUALITY_LADDER = (85, 70, 55, 40)  # JPEG quality steps when over the byte cap


def _encode(frame: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ProviderError("image re-encoding failed")
    return buf.tobytes()


def prepare_image_jpeg(
    path: Path,
    max_dim: int = DEFAULT_MAX_DIM,
    max_bytes: int = DEFAULT_MAX_BYTES,
    quality: int = QUALITY_LADDER[0],
) -> bytes:
    """Read an image, downscale to max_dim (longest side) and JPEG-encode.

    When the encoded size exceeds ``max_bytes`` the quality is lowered in
    steps and the longest side is halved, in that order, until the image
    fits. ``ProviderError`` only when even the smallest step cannot fit.
    """
    if max_dim < 32 or max_bytes < 1024:
        raise ProviderError("image limits are unreasonably small")
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ProviderError(f"cannot read image {Path(path).name}")
    height, width = frame.shape[:2]
    if max(height, width) > max_dim:
        scale = max_dim / max(height, width)
        frame = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    qualities = [quality] + [q for q in QUALITY_LADDER if q < quality]
    current_dim = max_dim
    for q in qualities:
        data = _encode(frame, q)
        if len(data) <= max_bytes:
            return data
    while True:  # quality exhausted: keep halving the longest side
        current_dim //= 2
        if current_dim < 32:
            raise ProviderError(
                f"image {Path(path).name} cannot fit {max_bytes} bytes at any supported size"
            )
        height, width = frame.shape[:2]
        if max(height, width) <= current_dim:
            continue
        scale = current_dim / max(height, width)
        frame = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        for q in qualities:
            data = _encode(frame, q)
            if len(data) <= max_bytes:
                return data
