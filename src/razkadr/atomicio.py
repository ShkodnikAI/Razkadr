"""Atomic file writes (naryad 029): a crash never leaves a truncated artifact.

Every JSON artifact a stage reads back (and the deck files of the build
package) lands on disk via tmp-file + ``os.replace`` in the same
directory: a reader sees either the previous file or the complete new
one, never a half-written one. This is what makes "обрыв посреди стадии
→ повторный запуск продолжает с места сбоя" safe: the killed stage
leaves no cache marker (markers are written after the body), so the
resume re-runs it and the atomic replace overwrites whatever partial
output the kill left behind.

The tmp name carries the pid, so two concurrent runs in one directory
cannot collide; a SIGKILL may leave a stray ``.<name>.tmp<pid>`` behind
- it is inert and overwritten/replaced by the next successful write.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path


def _tmp_name(path: Path, suffix: str = "") -> Path:
    return path.with_name(f".{path.name}.tmp{os.getpid()}{suffix}")


def write_json_atomic(path: str | Path, data: object, *, indent: int = 2, newline: bool = True) -> None:
    """Serialize ``data`` and move it into place atomically."""
    path = Path(path)
    text = json.dumps(data, ensure_ascii=False, indent=indent) + ("\n" if newline else "")
    tmp = _tmp_name(path)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_text_atomic(path: str | Path, text: str) -> None:
    """Move ``text`` into place atomically (same contract as write_json_atomic)."""
    path = Path(path)
    tmp = _tmp_name(path)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def build_atomic(path: str | Path, produce: Callable[[Path], None], *, tmp_suffix: str = "") -> None:
    """Run ``produce(tmp_path)`` and only then move the result into place.

    For writers that take a destination path (python-pptx save, an
    external converter): a crash inside ``produce`` leaves the caller's
    path untouched. ``tmp_suffix`` is for writers that insist on a
    particular extension (e.g. LibreOffice wants ``.pdf``).
    """
    path = Path(path)
    tmp = _tmp_name(path, tmp_suffix)
    try:
        produce(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
