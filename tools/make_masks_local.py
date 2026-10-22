#!/usr/bin/env python3
"""Provenance: guided-generation masks for Cosmos (data/inputs/masks/<Town>/frame_XXXXXX.npz), built on the source workstation.
Each mask is 1 x 684 x 1242 uint8 in letterbox coordinates (frame rows 154..528): 1 inside every labeled box (Car, Pedestrian, Cyclist),
padded by 4 px, 0 elsewhere. Cosmos anchors the masked regions to the simulator frame for the first N denoising steps."""
import os, sys
import numpy as np
SRC = "/mnt/features/carla_doses"; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "inputs", "masks"); PAD = 4
for t in ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"]:
    d = f"{SRC}/{t}_d00_ref"; o = os.path.join(OUT, t); os.makedirs(o, exist_ok=True); n = 0
    for f in sorted(x for x in os.listdir(d) if x.endswith(".txt")):
        m = np.zeros((1, 684, 1242), np.uint8)
        for ln in open(os.path.join(d, f)):
            p = ln.split()
            if not p or p[0] not in ("Car", "Pedestrian", "Cyclist"): continue
            x1, y1, x2, y2 = (float(v) for v in p[4:8])
            m[0, max(0, int(y1) - PAD) + 154: min(375, int(np.ceil(y2)) + PAD) + 154, max(0, int(x1) - PAD): min(1242, int(np.ceil(x2)) + PAD)] = 1
        np.savez_compressed(os.path.join(o, f[:-4] + ".npz"), m); n += 1
    print(t, n, "masks")
