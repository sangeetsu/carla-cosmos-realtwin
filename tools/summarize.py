#!/usr/bin/env python3
"""Morning summary: counts, speed, reproducibility vs the local A6000, simple sanity stats, and a contact sheet.
Writes results/summary.json, results/summary.txt and results/contact_sheet.jpg."""
import csv, json, os, sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).parent)); os.environ.setdefault("BUNDLE", str(Path(__file__).resolve().parent.parent))
import gen
RES, INP, B = gen.RES, gen.INP, gen.B
def first_frame(src):
    if src.endswith(".png"): f = cv2.imread(src)
    else: ok, f = cv2.VideoCapture(src).read()
    return f[154:529] if f is not None and f.shape[0] == 684 else f
def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2); return float("inf") if m == 0 else 10 * np.log10(255 ** 2 / m)
S = {}; J = gen.plan(); lines = []
for j, its in J.items():
    by = {}
    for it in its: k = f"{it['variant']}/{it['town']}"; by.setdefault(k, [0, 0]); by[k][0] += 1; by[k][1] += Path(it["out"]).exists()
    S[j] = {k: dict(expected=e, done=d) for k, (e, d) in by.items()}
    lines.append(f"{j}: " + "; ".join(f"{k} {d}/{e}" for k, (e, d) in by.items()))
t = list(csv.DictReader(open(RES / "logs" / "timing.csv"))) if (RES / "logs" / "timing.csv").exists() else []
for j in ("J1", "J4", "J2", "J3", "J5"):
    r = [x for x in t if x["job"] == j and int(x["n_new_outputs"]) > 0]
    if r:
        n = sum(int(x["n_new_outputs"]) for x in r); sec = sum(float(x["seconds"]) for x in r)
        S.setdefault("speed", {})[j] = dict(outputs=n, process_seconds=round(sec), seconds_per_output_per_slot=round(sec / n, 1))
        lines.append(f"speed {j}: {n} outputs, {sec / n:.1f} s per output per worker slot (incl. model loads)")
rep = {}
for v in ("shipped_mp4", "png_input"):
    vals = []
    for ref in sorted((B / "data" / "reference" / "Town03_local_A6000").glob("*.jpg")):
        h = RES / "J1_repro" / v / "Town03" / ref.name
        if h.exists(): a, b = cv2.imread(str(ref)), cv2.imread(str(h)); vals.append((psnr(a, b), float(np.abs(a.astype(int) - b.astype(int)).mean())))
    if vals: rep[v] = dict(n=len(vals), psnr_vs_local_median=round(float(np.median([x[0] for x in vals])), 2), mean_abs_diff_median=round(float(np.median([x[1] for x in vals])), 2))
if rep: S["repro_vs_local_A6000"] = rep; lines.append(f"reproducibility vs local A6000: {rep}")
san = {}
for town in gen.J3_TOWNS:
    fs = sorted((RES / "J3_realtwin" / f"{town}_d50_realtwin").glob("*.jpg"))[::50]
    if fs:
        ims = [cv2.imread(str(f)) for f in fs]; san[town] = dict(n_sampled=len(ims), mean=round(float(np.mean([i.mean() for i in ims])), 1),
                                                                   share_near_black=round(float(np.mean([(i.max(2) < 10).mean() for i in ims])), 3))
if san: S["sanity_J3"] = san; lines.append(f"J3 sanity (every 50th frame): {san}")
if (RES / "logs" / "check.json").exists(): S["check"] = json.load(open(RES / "logs" / "check.json"))
json.dump(S, open(RES / "summary.json", "w"), indent=1); open(RES / "summary.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
rows = []                                                                    # contact sheet: CARLA input | generated, a few per job
pick = [it for it in J["J1"][:2]] + [it for it in J["J2"] if it["frame"] in ("frame_000940",)][:4] + [it for t_ in gen.J3_TOWNS for it in [x for x in J["J3"] if x["town"] == t_][800:801]]
for it in pick:
    if Path(it["out"]).exists():
        a = first_frame(it["src"]); b = cv2.imread(it["out"])
        if a is not None and b is not None:
            row = np.hstack([cv2.resize(a, (621, 188)), cv2.resize(b, (621, 188))])
            cv2.putText(row, f"{it['job']} {it['variant']} {it['town']} {it['frame']}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            rows.append(row)
if rows: cv2.imwrite(str(RES / "contact_sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]); print(f"contact sheet: {len(rows)} rows")
