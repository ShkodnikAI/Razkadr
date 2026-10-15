"""Measure pHash distances between frame candidates (naryad 006 dup_threshold tuning)."""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, "src")
from razkadr.stages.frames import _candidates, _phash, detect_scenes


def ham(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def main() -> None:
    src = Path(sys.argv[1])
    fps = 10.0
    params = {"scene_threshold": 12.0, "min_scene_len_s": 0.5, "step_s": 2.5}
    scenes = detect_scenes(src, fps, params)
    print("scenes:", [(round(a, 3), round(b, 3)) for a, b in scenes])
    cands = _candidates(scenes, 30.0, params)
    hashes = []
    for t, scene_id, kind in cands:
        cap = cv2.VideoCapture(str(src))
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(t * fps))
        _ok, frame = cap.read()
        cap.release()
        hashes.append((t, scene_id, kind, _phash(frame)))
    kind_name = {0: "mid", 1: "uni"}
    print("candidates:", " ".join(f"{t:.1f}({kind_name[k]})" for t, _s, k, _h in hashes))
    print()
    print("pairwise distances (same or neighboring scene in bold with *):")
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            d = ham(hashes[i][3], hashes[j][3])
            tag = ""
            if hashes[j][1] - hashes[i][1] <= 1:
                tag = " *" if d <= 8 else ""
            if d <= 12 or tag:
                print(f"  t={hashes[i][0]:5.1f} scene{hashes[i][1]} <-> t={hashes[j][0]:5.1f} "
                      f"scene{hashes[j][1]}: dist={d:2d}{tag}")


if __name__ == "__main__":
    main()
