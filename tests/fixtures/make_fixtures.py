#!/usr/bin/env python3
"""Generate synthetic Razkadr fixtures with known ground truth.

Usage:
    python tests/fixtures/make_fixtures.py <dir>

Produces (no network, no external files, only ffmpeg lavfi sources):
    <dir>/scenes.mp4         30 s, 5 scenes, hard cuts at 6/12/18/24 s,
                             one blurred fragment, one near-duplicate fragment
    <dir>/titles.mp4         20 s, text strings on screen at known moments
    <dir>/silence.wav        10 s of silence with one tonal marker
    <dir>/ground_truth.json  cuts, blur interval, duplicate pair, texts, marker

Frames and samples are deterministic; the exact bytes may differ between
ffmpeg versions, so tests must rely on times and properties, not on hashes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

W, H, FPS = 320, 240, 10

# Scenes: (hex_color, box(x, y, w, h), kind, duplicate_of)
SCENES = [
    ("#CC4444", (16, 16, 128, 96), "sharp", None),
    ("#44CC44", (24, 24, 112, 84), "sharp", None),
    ("#4444CC", (24, 24, 112, 84), "blurry", None),
    ("#46CE46", (26, 24, 112, 84), "sharp", 1),
    ("#CCCC44", (96, 120, 176, 72), "sharp", None),
]
SCENE_DURATION_S = 6.0
BLUR_SIGMA = 10

# Texts on screen in titles.mp4. ASCII and Cyrillic; gaps between windows.
TEXTS = [
    ("FLOUR 500 G", 3.0, 7.0),
    ("МУКА 500 Г", 9.0, 13.0),
    ("BAKE 25 MIN", 15.0, 19.0),
]
TITLES_DURATION_S = 20.0
TITLES_BG = "#202020"

# Tonal marker in silence.wav.
TONE_FREQ_HZ = 440
TONE_START_S, TONE_END_S = 5.0, 6.0
AUDIO_DURATION_S, AUDIO_RATE_HZ = 10.0, 8000

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]


def find_font() -> str | None:
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            return path
    return None


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            "ffmpeg failed:\n"
            + " ".join(cmd[:8])
            + " ...\n"
            + (proc.stderr or "")[-2000:]
        )


def build_scenes(out: Path) -> list[str]:
    """5 solid-color scenes with a white box each; concat; one branch blurred."""
    inputs: list[str] = []
    branches: list[str] = []
    labels: list[str] = []
    for i, (color, box, kind, _dup) in enumerate(SCENES):
        inputs += ["-f", "lavfi", "-i", f"color=c={color}:s={W}x{H}:d={SCENE_DURATION_S}:r={FPS}"]
        x, y, w, h = box
        chain = f"[{i}:v]drawbox=x={x}:y={y}:w={w}:h={h}:color=white:t=fill"
        if kind == "blurry":
            chain += f",gblur=sigma={BLUR_SIGMA}"
        chain += f",format=yuv420p,setsar=1[v{i}]"
        branches.append(chain)
        labels.append(f"[v{i}]")
    graph = ";".join(branches) + ";" + "".join(labels) + f"concat=n={len(SCENES)}:v=1:a=0[out]"
    dst = out / "scenes.mp4"
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", graph, "-map", "[out]",
           "-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p", str(dst)]
    run(cmd)
    return cmd


def build_titles(out: Path) -> list[str]:
    """Dark background; white text lines shown in known time windows."""
    font = find_font()
    if font is None:
        raise SystemExit("drawtext needs a font file; none of the known candidates found")
    chains = []
    for text, t0, t1 in TEXTS:
        chains.append(
            f"drawtext=fontfile={font}:text='{text}':fontsize=40:fontcolor=white:"
            f"x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,{t0:g},{t1:g})'"
        )
    dst = out / "titles.mp4"
    cmd = ["ffmpeg", "-y",
           "-f", "lavfi", "-i", f"color=c={TITLES_BG}:s={W}x{H}:d={TITLES_DURATION_S}:r={FPS}",
           "-vf", ",".join(chains),
           "-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p", str(dst)]
    run(cmd)
    return cmd


def build_silence(out: Path) -> list[str]:
    """1 s of 440 Hz sine delayed to the marker window, padded to full length."""
    dst = out / "silence.wav"
    cmd = ["ffmpeg", "-y",
           "-f", "lavfi", "-i",
           f"sine=frequency={TONE_FREQ_HZ}:sample_rate={AUDIO_RATE_HZ}:duration=1",
           "-af", f"adelay={int(TONE_START_S * 1000)}:all=1,apad",
           "-t", f"{AUDIO_DURATION_S:g}",
           "-c:a", "pcm_s16le", str(dst)]
    run(cmd)
    return cmd


def ground_truth(commands: list[list[str]]) -> dict:
    segments = []
    for i, (color, box, kind, dup) in enumerate(SCENES):
        seg = {
            "index": i,
            "t_start": i * SCENE_DURATION_S,
            "t_end": (i + 1) * SCENE_DURATION_S,
            "color": color,
            "box_xywh": list(box),
            "kind": kind,
        }
        if dup is not None:
            seg["duplicate_of"] = dup
        segments.append(seg)
    blur_idx = next(i for i, s in enumerate(SCENES) if s[2] == "blurry")
    dup_scene = next(i for i, s in enumerate(SCENES) if s[3] is not None)
    return {
        "generator": "tests/fixtures/make_fixtures.py",
        "note": "bytes vary between ffmpeg versions; rely on times and properties, not hashes",
        "video": {"width": W, "height": H, "fps": FPS},
        "scenes": {
            "file": "scenes.mp4",
            "duration_s": len(SCENES) * SCENE_DURATION_S,
            "cuts_s": [i * SCENE_DURATION_S for i in range(1, len(SCENES))],
            "segments": segments,
            "blur_interval_s": [blur_idx * SCENE_DURATION_S, (blur_idx + 1) * SCENE_DURATION_S],
            "blur_sigma": BLUR_SIGMA,
            "duplicate_pair": [dup_scene, SCENES[dup_scene][3]],
        },
        "titles": {
            "file": "titles.mp4",
            "duration_s": TITLES_DURATION_S,
            "font": find_font(),
            "texts": [{"text": t, "t_start": a, "t_end": b} for t, a, b in TEXTS],
        },
        "audio": {
            "file": "silence.wav",
            "duration_s": AUDIO_DURATION_S,
            "sample_rate_hz": AUDIO_RATE_HZ,
            "marker": {"kind": "sine", "frequency_hz": TONE_FREQ_HZ,
                       "t_start": TONE_START_S, "t_end": TONE_END_S},
        },
        "commands": [" ".join(c) for c in commands],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic Razkadr fixtures.")
    parser.add_argument("dir", help="output directory (created if missing)")
    args = parser.parse_args(argv)

    if shutil.which("ffmpeg") is None:
        print("ffmpeg binary not found in PATH", file=sys.stderr)
        return 2
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)

    commands = [build_scenes(out), build_titles(out), build_silence(out)]
    (out / "ground_truth.json").write_text(
        json.dumps(ground_truth(commands), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for f in ("scenes.mp4", "titles.mp4", "silence.wav", "ground_truth.json"):
        print(out / f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
