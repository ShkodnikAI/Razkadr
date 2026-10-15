"""Measure frames-stage metrics on the synthetic fixture (naryad 006 tuning).

Run: .venv/bin/python scripts/measure_frames.py <path/to/scenes.mp4>
Prints per-candidate: t, scene_id, laplacian var, sharpness, phash, exposure.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "src")
from razkadr.stages.frames import _candidates, _exposure, _phash, _sharpness, detect_scenes


def main() -> None:
    import cv2

    src = Path(sys.argv[1])
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(src)],
        capture_output=True, text=True, check=True).stdout)
    vstream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    fps = eval(vstream["avg_frame_rate"])  # "10/1"
    duration = float(probe["format"]["duration"])

    params = {"scene_threshold": 27.0, "min_scene_len_s": 0.5, "step_s": 2.5}
    scenes = detect_scenes(src, fps, params)
    print(f"fps={fps} duration={duration:.3f}")
    print("scenes:", [(round(a, 3), round(b, 3)) for a, b in scenes])

    cands = _candidates(scenes, duration, params)
    print(f"candidates: {len(cands)}")
    print(f"{'t':>8} {'scene':>5} {'var':>9} {'sharp':>7} {'expo':>6}  phash")
    for t, scene_id, kind in cands:
        cap = cv2.VideoCapture(str(src))
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(t * fps))
        _ok, frame = cap.read()
        cap.release()
        if not _ok:
            print(f"{t:8.3f} DECODE FAIL")
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        print(f"{t:8.3f} {scene_id:5d} {var:9.3f} {_sharpness(gray):7.4f} "
              f"{_exposure(gray):6.4f}  {_phash(frame)}")


if __name__ == "__main__":
    main()
