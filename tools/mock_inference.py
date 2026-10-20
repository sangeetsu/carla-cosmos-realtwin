#!/usr/bin/env python3
"""Stand-in for Cosmos examples/inference.py, for testing the bundle's plumbing without a GPU (gen.py --mock).
Writes outputs with the same names, sizes and formats Cosmos writes. MOCK_BLOCK=<substring> drops that sample's output at seed 1
(simulates a guardrail rejection, so the seed-2 retry path is exercised)."""
import argparse, json, os, time
import numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("-i", nargs="+"); ap.add_argument("-o"); ap.add_argument("--offload-guardrail-models", action="store_true")
ap.add_argument("--model", default=None); a = ap.parse_args()
for p in a.i:
    s = json.load(open(p)); name = s["name"]; src = s["video_path"]
    if os.environ.get("MOCK_BLOCK") and os.environ["MOCK_BLOCK"] in name and s["seed"] == 1: print("mock guardrail block", name); continue
    if src.endswith(".png"): frames = [cv2.imread(src)]
    else:
        cap = cv2.VideoCapture(src); frames = []
        while True:
            ok, f = cap.read()
            if not ok: break
            frames.append(f)
    frames = [cv2.addWeighted(f, 0.8, np.full_like(f, (40, 20, 0)), 1.0, 0) for f in frames]   # visibly "restyled"
    if s.get("max_frames") == 1:
        cv2.imwrite(os.path.join(a.o, name + ".jpg"), frames[0])
        for k in [c for c in ("edge", "depth", "seg") if c in s]: cv2.imwrite(os.path.join(a.o, f"{name}_control_{k}.jpg"), cv2.Canny(frames[0], 100, 200))
    else:
        h, w = frames[0].shape[:2]; vw = cv2.VideoWriter(os.path.join(a.o, name + ".mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 16, (w, h))
        for f in frames: vw.write(f)
        vw.release()
    time.sleep(0.02)
print("mock done")
