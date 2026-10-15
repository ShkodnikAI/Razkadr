"""Synthetic fixture checks: ground truth must match actual media content.

Thresholds are taken from real runs (ffmpeg 7.1.5, 2026-10-15, see
tests/fixtures/README.md); margins around measured values are >= 1.9x.
Frames are decoded to raw RGB and compared in pure Python: fixtures are
tiny (320x240 @ 10 fps), no extra dependencies needed.
"""

from __future__ import annotations

import subprocess

import pytest

pytestmark = pytest.mark.ffmpeg

W, H = 320, 240


def frame(path, t: float) -> bytes:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path),
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        check=False,
        capture_output=True,
    ).stdout
    assert len(out) == W * H * 3, f"unexpected frame size at t={t}: {len(out)}"
    return out


def corner_mean(buf: bytes, n: int = 10) -> tuple[float, float, float]:
    px = [buf[(y * W + x) * 3:(y * W + x) * 3 + 3] for y in range(n) for x in range(n)]
    return tuple(round(sum(p[c] for p in px) / len(px), 1) for c in range(3))  # type: ignore[return-value]


def mean_abs_diff(a: bytes, b: bytes) -> float:
    return round(sum(abs(a[i] - b[i]) for i in range(len(a))) / len(a), 2)


def reblur_diff(buf: bytes) -> float:
    """Change caused by blurring again: small for already-blurry frames."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-i", "pipe:0", "-vf", "gblur=sigma=8", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        input=buf,
        check=False,
        capture_output=True,
    ).stdout
    return mean_abs_diff(buf, out)


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def duration_s(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    return float(out.stdout.strip())


def pcm_samples(path, rate: int) -> list[int]:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"],
        check=False,
        capture_output=True,
    ).stdout
    return [int.from_bytes(raw[i:i + 2], "little", signed=True) for i in range(0, len(raw), 2)]


# --- generation --------------------------------------------------------------

def test_generation_within_budget(synthetic_fixtures):
    assert synthetic_fixtures.generation_s <= 30.0, (
        f"fixture generation took {synthetic_fixtures.generation_s:.2f} s (budget 30 s)"
    )


def test_ground_truth_files_declared(synthetic_fixtures):
    gt = synthetic_fixtures.ground_truth
    for name in (gt["scenes"]["file"], gt["titles"]["file"], gt["audio"]["file"]):
        assert synthetic_fixtures.path(name).is_file()


# --- scenes.mp4 --------------------------------------------------------------

def test_scenes_duration_and_cuts(synthetic_fixtures):
    gt = synthetic_fixtures.ground_truth["scenes"]
    assert abs(duration_s(synthetic_fixtures.path(gt["file"])) - gt["duration_s"]) < 0.2
    assert gt["cuts_s"] == [6.0, 12.0, 18.0, 24.0]


@pytest.mark.parametrize("seg_index", range(5))
def test_scene_color_matches_ground_truth(synthetic_fixtures, seg_index):
    """Frame-by-frame color check at scene middles and at cut moments."""
    gt = synthetic_fixtures.ground_truth["scenes"]
    seg = gt["segments"][seg_index]
    expected = hex_to_rgb(seg["color"])
    mid_t = seg["t_start"] + 3.0
    got = corner_mean(frame(synthetic_fixtures.path(gt["file"]), mid_t))
    for got_c, exp_c in zip(got, expected):
        assert abs(got_c - exp_c) <= 10, f"scene {seg_index} at t={mid_t}: {got} vs {seg['color']}"


@pytest.mark.parametrize("cut", [6.0, 12.0, 18.0, 24.0])
def test_hard_cut_changes_color(synthetic_fixtures, cut):
    """At each cut moment color switches: -0.1 s is previous scene, +0.1 s next."""
    gt = synthetic_fixtures.ground_truth["scenes"]
    path = synthetic_fixtures.path(gt["file"])
    before = corner_mean(frame(path, cut - 0.1))
    after = corner_mean(frame(path, cut + 0.1))
    assert mean_abs_diff(bytes(map(int, before)), bytes(map(int, after))) > 40, (
        f"cut at {cut} s did not switch scene color: {before} -> {after}"
    )


def test_near_duplicate_pair_is_similar(synthetic_fixtures):
    gt = synthetic_fixtures.ground_truth["scenes"]
    i, j = gt["duplicate_pair"]
    path = synthetic_fixtures.path(gt["file"])
    fi = frame(path, gt["segments"][i]["t_start"] + 2.0)
    fj = frame(path, gt["segments"][j]["t_start"] + 2.0)
    assert mean_abs_diff(fi, fj) < 15  # measured 3.26


@pytest.mark.parametrize("pair", [(0, 1), (1, 4), (0, 4)])
def test_different_scenes_are_far_apart(synthetic_fixtures, pair):
    gt = synthetic_fixtures.ground_truth["scenes"]
    path = synthetic_fixtures.path(gt["file"])
    fa = frame(path, gt["segments"][pair[0]]["t_start"] + 2.0)
    fb = frame(path, gt["segments"][pair[1]]["t_start"] + 2.0)
    assert mean_abs_diff(fa, fb) > 40  # measured 67.9-81.8


def test_blurry_fragment_is_actually_blurry(synthetic_fixtures):
    """Re-blurring changes sharp frames a lot and blurry frames little."""
    gt = synthetic_fixtures.ground_truth["scenes"]
    path = synthetic_fixtures.path(gt["file"])
    b0, b1 = gt["blur_interval_s"]
    assert reblur_diff(frame(path, (b0 + b1) / 2)) < 3.0  # measured 1.78
    for t in (2.0, 8.0, 27.0):  # sharp scenes; measured 3.97-4.50
        assert reblur_diff(frame(path, t)) > 3.0, f"frame at {t} s looks blurry"


# --- titles.mp4 --------------------------------------------------------------

def test_titles_duration(synthetic_fixtures):
    gt = synthetic_fixtures.ground_truth["titles"]
    assert abs(duration_s(synthetic_fixtures.path(gt["file"])) - gt["duration_s"]) < 0.2


def test_titles_appear_in_known_windows(synthetic_fixtures):
    gt = synthetic_fixtures.ground_truth["titles"]
    path = synthetic_fixtures.path(gt["file"])
    empty = frame(path, 1.0)
    for entry in gt["texts"]:
        t_mid = (entry["t_start"] + entry["t_end"]) / 2
        with_text = frame(path, t_mid)
        assert mean_abs_diff(empty, with_text) > 3.0, (
            f"text {entry['text']!r} not visible at t={t_mid}"
        )
    gap_after = gt["texts"][0]["t_end"] + 1.0
    assert mean_abs_diff(empty, frame(path, gap_after)) < 1.0, "unexpected content in gap"


# --- silence.wav -------------------------------------------------------------

def test_silence_duration_and_rate(synthetic_fixtures):
    gt = synthetic_fixtures.ground_truth["audio"]
    assert abs(duration_s(synthetic_fixtures.path(gt["file"])) - gt["duration_s"]) < 0.2
    rate = gt["sample_rate_hz"]
    samples = pcm_samples(synthetic_fixtures.path(gt["file"]), rate)
    assert abs(len(samples) / rate - gt["duration_s"]) < 0.2


def test_tonal_marker_and_silence_levels(synthetic_fixtures):
    gt = synthetic_fixtures.ground_truth["audio"]
    rate = gt["sample_rate_hz"]
    samples = pcm_samples(synthetic_fixtures.path(gt["file"]), rate)

    def rms(t0: float, t1: float) -> float:
        seg = samples[int(t0 * rate):int(t1 * rate)]
        return (sum(s * s for s in seg) / len(seg)) ** 0.5 / 32768

    marker = gt["marker"]
    assert rms(marker["t_start"], marker["t_end"]) > 0.05  # measured 0.088
    assert rms(1.0, 2.0) < 0.005  # measured 0.0
    assert rms(marker["t_end"] + 1.0, marker["t_end"] + 2.0) < 0.005

    window = samples[int(marker["t_start"] * rate):int(marker["t_end"] * rate)]
    crossings = sum(1 for i in range(1, len(window)) if (window[i - 1] < 0) != (window[i] < 0))
    expected = marker["frequency_hz"] * 2 * (marker["t_end"] - marker["t_start"])
    assert abs(crossings - expected) <= expected * 0.1  # measured 879 vs 880
