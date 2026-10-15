"""Shared pytest fixtures: synthetic media with known ground truth (naryad 034)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

FIXTURE_SCRIPT = Path(__file__).parent / "fixtures" / "make_fixtures.py"
GENERATION_BUDGET_S = 30.0  # naryad 034 acceptance: generation must take <= 30 s


@dataclass(frozen=True)
class Fixtures:
    dir: Path
    generation_s: float

    @property
    def ground_truth(self) -> dict:
        import json

        return json.loads((self.dir / "ground_truth.json").read_text(encoding="utf-8"))

    def path(self, name: str) -> Path:
        return self.dir / name


@pytest.fixture(scope="session")
def synthetic_fixtures(tmp_path_factory: pytest.TempPathFactory) -> Fixtures:
    """Generate all synthetic fixtures once per test run; skip without ffmpeg."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found in PATH; media fixture tests are skipped")
    out = tmp_path_factory.mktemp("razkadr_fixtures")
    started = time.monotonic()
    subprocess.run(
        [sys.executable, str(FIXTURE_SCRIPT), str(out)],
        check=True,
        timeout=120,
        capture_output=True,
    )
    return Fixtures(dir=out, generation_s=time.monotonic() - started)
