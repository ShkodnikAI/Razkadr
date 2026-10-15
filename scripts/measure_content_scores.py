"""Measure ContentDetector frame scores for every frame pair (naryad 006 tuning)."""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, "src")


def main() -> None:
    path = Path(sys.argv[1])
    cap = cv2.VideoCapture(str(path))
    prev_hsv = None
    scores = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if prev_hsv is None:
            scores.append(0.0)
        else:
            ph, ps, pv = prev_hsv
            dh = float(cv2.absdiff(h, ph).mean())
            ds = float(cv2.absdiff(s, ps).mean())
            dv = float(cv2.absdiff(v, pv).mean())
            scores.append((dh + ds + dv) / 3.0)
        prev_hsv = (h, s, v)
        i += 1
    cap.release()
    # print top scores around known cuts and overall stats
    order = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    print("top-15 scores (frame, t, score):")
    for k in order[:15]:
        print(f"  frame {k:4d}  t={k / 10.0:6.2f}s  score={scores[k]:7.2f}")
    cuts = [6, 12, 18, 24]
    for c in cuts:
        window = scores[(c - 1) * 10:(c + 1) * 10 + 1]
        print(f"cut {c:>2}s: max score in ±1s window = {max(window):.2f}")
    inside = [scores[k] for k in range(len(scores)) if all(abs(k / 10.0 - c) > 1.0 for c in cuts)]
    print(f"inside-scene noise: max={max(inside):.2f} median={sorted(inside)[len(inside)//2]:.2f}")


if __name__ == "__main__":
    main()
