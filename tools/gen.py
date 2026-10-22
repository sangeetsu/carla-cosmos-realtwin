#!/usr/bin/env python3
"""Generation driver for the CARLA real-twin bundle (Cosmos-Transfer2.5-2B).

  gen.py --check            GPU pre-flight: two real one-frame generations, picks the number of worker slots
  gen.py --run [--only J3]  all jobs in priority order, resumable (anything already in results/ is skipped)
  add --mock to either for a dry run of the whole pipeline without Cosmos (tools/mock_inference.py)

Each chunk of samples is one `examples/inference.py` process (model loaded once per chunk). Worker slots run chunks in
parallel (several per GPU if memory allows). Outputs are cropped back from the 1242x684 letterbox to the 1242x375 frame.
Missing outputs are retried: once with the same seed (crash), then with seeds 2 and 3 (e.g. a guardrail rejection).
Guardrails are never disabled (NVIDIA Open Model License).
"""
import argparse, csv, json, os, queue, shutil, subprocess, sys, threading, time
from pathlib import Path

B = Path(os.environ["BUNDLE"]); WORK = Path(os.environ["WORK"]); RES = Path(os.environ["RESULTS"]); COSMOS = Path(os.environ["COSMOS"])
PY = os.environ["PY"]; INP = B / "data" / "inputs"; LOGS = RES / "logs"; SCR = Path(os.environ.get("GEN_SCRATCH", WORK)); OUTRAW = SCR / "out"; SPECS = SCR / "specs"
PROMPT = ("A real dashcam photograph taken from a car driving on a suburban road on a clear sunny day. Real-world asphalt, real cars, "
          "real pedestrians, natural lighting, realistic textures, sharp details, shot on a car-mounted camera.")
PROMPT_VIDEO = PROMPT.replace("A real dashcam photograph", "A real dashcam video")
SEG_PROMPT = "car. truck. bus. person. bicycle. motorcycle. road. sidewalk. building. tree. pole. traffic sign. traffic light. sky."
EDGE = {"edge": {"preset_edge_threshold": "medium"}}                         # set-up D (identical to the local pilot)
M1 = {"edge": {"control_weight": 1.0, "preset_edge_threshold": "medium"}, "depth": {"control_weight": 1.0},
      "seg": {"control_weight": 1.0, "control_prompt": SEG_PROMPT}}
NV4 = {"edge": {"control_weight": 1.0, "preset_edge_threshold": "medium"}, "depth": {"control_weight": 1.0},   # NVIDIA's car multi-control example:
       "seg": {"control_weight": 1.0, "control_prompt": SEG_PROMPT}, "vis": {"control_weight": 1.0}}              # all four controls at 1.0
V25 = {"edge": {"control_weight": 1.0, "preset_edge_threshold": "medium"}, "vis": {"control_weight": 0.25}}     # light blur anchors colours
IMG = {"num_video_frames_per_chunk": 1, "max_frames": 1}
CROP = (0, 154, 1242, 529)                                                   # letterbox -> original 1242x375 frame
SMOKE = [("Town07", k) for k in range(820, 1021, 40)] + [("Town10HD", k) for k in range(820, 1021, 40)]

def log(m):
    line = f"[gen {time.strftime('%F %T')}] {m}"; print(line, flush=True)
    with open(LOGS / "gen.log", "a") as f: f.write(line + "\n")

def item(job, variant, town, frame, src, out, controls, kind="img", prompt=PROMPT, cli=(), seed0=1):
    return dict(job=job, variant=variant, town=town, frame=frame, src=str(src), out=str(out), controls=controls, kind=kind, prompt=prompt, cli=list(cli), seed0=seed0)

def png_letterbox(jpg):
    """Lossless input for the 'uncompressed' variant: the original JPEG frame on a black 1242x684 canvas, saved as PNG."""
    from PIL import Image
    p = SCR / "png" / jpg.parent.name / (jpg.stem + ".png")
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True); im = Image.open(jpg).convert("RGB"); assert im.size == (1242, 375), im.size
        c = Image.new("RGB", (1242, 684)); c.paste(im, (0, 154)); c.save(p)
    return p

def guided(town, frame, steps):
    """Edge control + guided generation: labeled boxes (data/inputs/masks) stay anchored to the CARLA frame for the first `steps` of 35."""
    return dict(EDGE, guided_generation_mask=str(INP / "masks" / town / f"{frame}.npz"), guided_generation_step_threshold=steps)

# set-up name -> (cell suffix, controls for a given town/frame). D = the local pilot's set-up (edge only).
SETUPS = {"D": ("d50_realtwin", lambda t, f: EDGE),
          "G25": ("d52_guided25", lambda t, f: guided(t, f, 25)),           # tested on the workstation: no survival gain over D
          "M1": ("d54_edge_depth_seg", lambda t, f: M1),
          "V25": ("d55_edge_blur25", lambda t, f: V25),
          "NV4": ("d56_four_controls", lambda t, f: NV4),
          "G35": ("d57_guided35_pasteback", lambda t, f: guided(t, f, 35))}   # boxes anchored for all 35 steps: background-only restyle
COMPARE = ["M1", "NV4", "V25", "G35"]                                      # J2: label-preserving candidates
BULK_TOWNS = os.environ.get("BULK_TOWNS", "Town03 Town05 Town10HD Town07 Town01 Town02 Town04 Town06").split()

def setup_status():
    """Which set-ups passed check.sh, and the bulk set-up after any fallback."""
    info = json.load(open(LOGS / "check.json")) if (LOGS / "check.json").exists() else {}
    ok = {k: info.get(k, {}).get("ok", True) for k in ("D", "M1", "NV4", "G35")}
    usable = {"D": ok["D"], "M1": ok["M1"], "NV4": ok["NV4"], "V25": ok["NV4"], "G25": ok["G35"], "G35": ok["G35"]}
    bulk = os.environ.get("BULK_SETUP", "M1")                            # chosen on the workstation, 2026-10-21: best label survival
    if not usable.get(bulk, False): bulk = "D"
    return usable, bulk

def cell(setup, town, frame, root="realtwin", tag=""):
    return RES / root / setup / f"{town}_{SETUPS[setup][0]}{tag}" / f"{frame}.jpg"

def plan():
    """All jobs, highest priority first. Outputs are keyed by (set-up, town, frame), so a frame shared by two jobs is generated once."""
    usable, bulk = setup_status()
    J = {}
    J["J1"] = []                                                             # reproducibility vs the local A6000 outputs (30 frames)
    for mp4 in sorted((INP / "repro" / "Town03").glob("*.mp4")):
        f = mp4.stem
        J["J1"].append(item("J1", "shipped_mp4", "Town03", f, mp4, RES / "J1_repro" / "shipped_mp4" / "Town03" / f"{f}.jpg", EDGE))
        J["J1"].append(item("J1", "png_input", "Town03", f, png_letterbox(mp4.with_suffix(".jpg")), RES / "J1_repro" / "png_input" / "Town03" / f"{f}.jpg", EDGE))
    J["J2"] = []                                                             # label-preserving comparison on frames the pilot already scored
    frames = [(t, f"frame_{k:06d}") for t in ("Town03", "Town05") for k in range(800, 1050, 2)] + [(t, f"frame_{k:06d}") for t, k in SMOKE]
    for v in COMPARE:
        if not usable[v]: continue
        for t, f in frames: J["J2"].append(item("J2", v, t, f, INP / "D" / t / f"{f}.mp4", cell(v, t, f), SETUPS[v][1](t, f)))
    J["J4"] = []                                                             # video mode on two 35-record drives (base and distilled edge)
    for mp4 in sorted((INP / "video").glob("*.mp4")):
        J["J4"].append(item("J4", "video_base", mp4.stem.split("_")[0], mp4.stem, mp4, RES / "J4_video" / "base" / f"{mp4.stem}.mp4", EDGE, "video", PROMPT_VIDEO))
        J["J4"].append(item("J4", "video_distilled", mp4.stem.split("_")[0], mp4.stem, mp4, RES / "J4_video" / "distilled" / f"{mp4.stem}.mp4",
                            dict(EDGE, num_steps=4), "video", PROMPT_VIDEO, ("--model", "edge/distilled")))
    J["J3"] = []                                                             # bulk real twin with the chosen set-up, all frames, Town03/Town05 first
    for t in BULK_TOWNS:
        if bulk == "D" and t in ("Town03", "Town05"): continue              # already made by the local pilot
        for p in sorted((INP / "D" / t).glob("*.mp4")):
            J["J3"].append(item("J3", bulk, t, p.stem, p, cell(bulk, t, p.stem), SETUPS[bulk][1](t, p.stem)))
    J["J5"] = []                                                             # filler: held-out records again with generation seed 11
    for t in BULK_TOWNS:
        for p in sorted((INP / "D" / t).glob("*.mp4")):
            if int(p.stem.split("_")[1]) >= 800:
                J["J5"].append(item("J5", bulk + "_seed11", t, p.stem, p, cell(bulk, t, p.stem, "realtwin_seed11", "_s11"), SETUPS[bulk][1](t, p.stem), seed0=11))
    return J

CHUNK = {"J1": 30, "J4": 2, "J2": 50, "J3": 150, "J5": 150}

def spec(it, seed, name):
    s = {"name": name, "prompt": it["prompt"], "video_path": it["src"], "seed": it.get("seed0", 1) + seed - 1,   # retries: seed0, seed0+1, seed0+2
         "resolution": "720", "keep_input_resolution": True}
    if it["kind"] == "img": s.update(IMG)
    s.update(it["controls"]); return s

def harvest(chunk_dir):
    """Crop finished raw outputs of one chunk into results/. Returns the number of new results."""
    from PIL import Image
    meta = chunk_dir / "items.json"
    if not meta.exists(): return 0
    n = 0
    for name, it in json.load(open(meta)).items():
        out = Path(it["out"])
        if out.exists(): continue
        raw = OUTRAW / chunk_dir.name / (name + (".mp4" if it["kind"] == "video" else ".jpg"))
        if not raw.exists(): continue
        try: n += _harvest_one(name, it, raw, out)
        except Exception as e: log(f"WARNING could not post-process {name}: {type(e).__name__}: {e}")
    return n

def _harvest_one(name, it, raw, out):
    from PIL import Image
    if True:
        out.parent.mkdir(parents=True, exist_ok=True)
        if it["kind"] == "img":
            im = Image.open(raw).convert("RGB")
            if im.size != (1242, 684): log(f"WARNING unexpected output size {im.size} for {name}"); return 0
            im.crop(CROP).save(out.with_suffix(".tmp.jpg"), quality=95); os.replace(out.with_suffix(".tmp.jpg"), out)
            if it["job"] == "J2":                                            # keep the control maps the model actually used
                cdir = out.parent / "controls"; cdir.mkdir(exist_ok=True)
                for c in raw.parent.glob(name + "_control_*.jpg"):
                    Image.open(c).convert("RGB").crop(CROP).save(cdir / f"{it['frame']}_{c.stem.split('_control_')[1]}.jpg", quality=90)
        else:
            tmp = out.with_suffix(".tmp.mp4")
            # H.264 needs even sizes: the 1242x375 frame gets one black row (1242x376)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-vf", "crop=1242:375:0:154,pad=1242:376", "-c:v", "libx264", "-crf", "16",
                            "-pix_fmt", "yuv420p", str(tmp)], check=True)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", it["src"], "-i", str(tmp), "-filter_complex",
                            "[0:v]crop=1242:375:0:154,pad=1242:376[a];[a][1:v]vstack", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                            str(out.with_name(out.stem + "_vs_carla.mp4"))], check=True)
            fdir = out.with_name(out.stem + "_frames"); fdir.mkdir(exist_ok=True)   # every generated frame, for analysis
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-vf", "crop=1242:375:0:154", "-q:v", "2",
                            str(fdir / "frame_%03d.jpg")], check=True)
            os.replace(tmp, out)
        return 1

def harvest_all():
    if not SPECS.exists(): return 0
    return sum(harvest(d) for d in sorted(SPECS.iterdir()) if d.is_dir())

def gpus():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
        return [dict(index=int(a), name=b.strip(), total=int(c), used=int(d)) for a, b, c, d in (l.split(",") for l in q.strip().splitlines())]
    except Exception: return []

def gpu_ids():
    """GPUs this job may use. Under Slurm (or any scheduler) CUDA_VISIBLE_DEVICES lists the allocated devices, in the numbering CUDA
    expects inside the job; nvidia-smi's own indices can differ there, so they are only the fallback."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd and cvd not in ("NoDevFiles", "-1"): return [x.strip() for x in cvd.split(",") if x.strip()]
    return [str(x["index"]) for x in gpus()] or ["0"]

def ram_limit_gb():
    """Host RAM this job may use: the node's available memory, capped by the Slurm allocation when there is one."""
    m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if l.split(":")[0] in ("MemTotal", "MemAvailable")}
    avail = m["MemAvailable"] // 1048576
    for k in ("SLURM_MEM_PER_NODE",):
        if os.environ.get(k, "").isdigit(): avail = min(avail, int(os.environ[k]) // 1024)
    if os.environ.get("SLURM_MEM_PER_CPU", "").isdigit() and os.environ.get("SLURM_CPUS_ON_NODE", "").isdigit():
        avail = min(avail, int(os.environ["SLURM_MEM_PER_CPU"]) * int(os.environ["SLURM_CPUS_ON_NODE"]) // 1024)
    return m["MemTotal"] // 1048576, avail

class Peak(threading.Thread):
    """Polls GPU memory and host RAM while a test runs."""
    def __init__(s): super().__init__(daemon=True); s.vram = 0; s.ram = 0; s.stop = False
    def run(s):
        while not s.stop:
            g = gpus(); s.vram = max([s.vram] + [x["used"] for x in g])
            m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if l.split(":")[0] in ("MemTotal", "MemAvailable")}
            s.ram = max(s.ram, (m["MemTotal"] - m["MemAvailable"]) // 1024); time.sleep(2)

def run_chunk(cid, items, seed, gpu, mock, offload):
    d = SPECS / cid; o = OUTRAW / cid; d.mkdir(parents=True, exist_ok=True); o.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.json"): old.unlink()                               # never re-run stale specs
    meta = {}
    for it in items:
        name = f"{it['job']}_{it['variant']}_{it['town']}_{it['frame']}"; meta[name] = it
        json.dump(spec(it, seed, name), open(d / f"{name}.json", "w"), indent=1)
    json.dump(meta, open(d / "items.json", "w"), indent=1)
    cli = items[0]["cli"]
    if mock: cmd = [sys.executable, str(B / "tools" / "mock_inference.py")]
    else: cmd = [PY, "examples/inference.py"]
    cmd += ["-i", *sorted(str(p) for p in d.glob("*.json") if p.name != "items.json"), "-o", str(o)] + (["--offload-guardrail-models"] if offload else []) + cli
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    t0 = time.time()
    with open(LOGS / f"chunk_{cid}.log", "w") as lf: rc = subprocess.run(cmd, cwd=COSMOS if not mock else B, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
    dt = time.time() - t0; n = harvest(d)
    return rc, dt, n

def check(mock):
    LOGS.mkdir(parents=True, exist_ok=True); g = gpus()
    tot, avail = ram_limit_gb(); du = shutil.disk_usage(WORK); ids = gpu_ids()
    info = dict(host=os.uname().nodename, gpus=g, gpu_ids=ids, ram_total_gb=tot, ram_available_gb=avail, disk_free_gb=du.free // 2**30)
    log(f"host {info['host']}; GPUs {[(x['name'], x['total']) for x in g]}; using GPU ids {ids}; RAM usable {avail} GB of {tot}; disk free {info['disk_free_gb']} GB")
    if not g and not mock: log("ERROR: no NVIDIA GPU visible (nvidia-smi)"); sys.exit(1)
    src = INP / "D" / "Town10HD" / "frame_000940.mp4"
    tests = [("D", item("CHECK", "D_edge", "Town10HD", "frame_000940", src, RES / "check" / "D_edge.jpg", EDGE)),
             ("M1", item("CHECK", "M1_edge_depth_seg", "Town10HD", "frame_000940", src, RES / "check" / "M1_edge_depth_seg.jpg", M1)),
             ("NV4", item("CHECK", "NV4_four_controls", "Town10HD", "frame_000940", src, RES / "check" / "NV4_four_controls.jpg", NV4)),
             ("G35", item("CHECK", "G35_guided", "Town10HD", "frame_000940", src, RES / "check" / "G35_guided.jpg", guided("Town10HD", "frame_000940", 35)))]
    for k, it in tests:
        if Path(it["out"]).exists(): Path(it["out"]).unlink()
        p = Peak(); p.start(); rc, dt, n = run_chunk(f"check_{k}", [it], 1, ids[0], mock, True); p.stop = True; p.join()
        info[k] = dict(ok=bool(n), rc=rc, seconds=round(dt, 1), peak_vram_mib=p.vram, peak_ram_mib=p.ram)
        log(f"check {k}: {'OK' if n else 'FAILED'} in {dt:.0f} s (incl. model load); peak GPU memory {p.vram} MiB, peak used RAM {p.ram} MiB")
        if not n: log(f"  see {LOGS}/chunk_check_{k}.log")
    if not info["D"]["ok"]: json.dump(info, open(LOGS / "check.json", "w"), indent=1); log("CHECK FAILED (the main job cannot run)"); sys.exit(1)
    per = max(info[k]["peak_vram_mib"] for k in ("D", "M1", "NV4", "G35") if info[k]["ok"])
    total = min(x["total"] for x in g) if g else 81920
    cap = int(os.environ.get("MAX_SLOTS_PER_GPU", 1))                       # default: one generation at a time per GPU
    slots_gpu = max(1, min(cap, int(total * 0.92 // max(per, 1) if per else 1)))
    if total < 44000: log(f"WARNING: this GPU has {total // 1024} GB; multi-control needs about 40 GB. Use an 80 GB A100 or larger.")
    slots_ram = max(1, int(info["ram_available_gb"] // 32))
    info["workers"] = min(slots_gpu * len(ids), slots_ram)
    json.dump(info, open(LOGS / "check.json", "w"), indent=1)
    usable, bulk = setup_status()
    log(f"CHECK PASSED. Worker slots: {info['workers']} ({slots_gpu} per GPU by memory, {slots_ram} by RAM). "
        f"Set-ups usable: {[k for k, v in usable.items() if v]}; bulk set-up: {bulk}")

def run(only, workers, mock, offload, stagger):
    info = json.load(open(LOGS / "check.json")) if (LOGS / "check.json").exists() else {}
    ids = gpu_ids(); ngpu = len(ids)
    workers = workers or min(info.get("workers", 1) // max(1, len(info.get("gpu_ids", ids))) * ngpu or 1, max(1, ram_limit_gb()[1] // 32))
    # (slots per GPU from check.sh x GPUs of THIS job, so a later job with a different GPU count adapts)
    J = plan(); order = [j for j in ("J1", "J2", "J4", "J3", "J5") if (j in only if only else j != "J5")]   # J5 only on request
    usable, bulk = setup_status(); log(f"bulk set-up {bulk}; comparison set-ups {[v for v in COMPARE if usable[v]]}")
    n_h = harvest_all(); log(f"resume: harvested {n_h} finished outputs; jobs {order}; {workers} worker slots on {ngpu} GPU(s)")
    tcsv = LOGS / "timing.csv"; new = not tcsv.exists(); lock = threading.Lock()
    with open(tcsv, "a", newline="") as f:
        if new: csv.writer(f).writerow(["chunk", "job", "variant", "seed", "n_samples", "n_new_outputs", "seconds", "slot", "gpu", "rc", "start"])
    runid = time.strftime("%m%d-%H%M%S")                                     # chunk folders are unique per invocation
    rounds = [1, 1, 2, 3]                                                    # seed per round: 1, retry 1 (crash), then 2, then 3
    for r, seed in enumerate(rounds):
        Q = queue.Queue(); total = 0; queued = set()
        for j in order:
            pend = [it for it in J[j] if not Path(it["out"]).exists() and it["out"] not in queued]
            queued.update(it["out"] for it in pend)
            groups = {}
            for it in pend: groups.setdefault(it["variant"], []).append(it)
            for v, its in groups.items():
                for i in range(0, len(its), CHUNK[j]):
                    Q.put((f"{runid}_{j}_{v}_r{r}_s{seed}_{i:05d}", its[i:i + CHUNK[j]], seed)); total += len(its[i:i + CHUNK[j]])
        if total == 0: break
        log(f"round {r + 1} (seed {seed}): {total} samples in {Q.qsize()} chunks")
        def worker(slot):
            time.sleep(slot * stagger if r == 0 else 0)                      # stagger the first model loads (host-RAM peaks)
            while True:
                try: cid, its, sd = Q.get_nowait()
                except queue.Empty: return
                gpu = ids[slot % ngpu]; t0 = time.strftime("%F %T")
                try: rc, dt, n = run_chunk(cid, its, sd, gpu, mock, offload)
                except Exception as e: log(f"ERROR in chunk {cid}: {type(e).__name__}: {e}"); continue   # keep the slot alive
                with lock:
                    with open(tcsv, "a", newline="") as f: csv.writer(f).writerow([cid, its[0]["job"], its[0]["variant"], sd, len(its), n, round(dt, 1), slot, gpu, rc, t0])
                log(f"{cid}: {n}/{len(its)} outputs in {dt:.0f} s (slot {slot}, GPU {gpu}, rc {rc})")
        T = [threading.Thread(target=worker, args=(s,)) for s in range(workers)]
        for t in T: t.start()
        for t in T: t.join()
    left = {j: sum(not Path(it["out"]).exists() for it in J[j]) for j in order}
    log(f"RUN FINISHED. Missing outputs per job: {left}")
    return 0 if not any(left.values()) else 3

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--run", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of jobs, e.g. J1,J3"); ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--mock", action="store_true"); ap.add_argument("--no-offload", action="store_true", help="keep guardrail models on the GPU (faster, more memory)")
    ap.add_argument("--stagger", type=int, default=int(os.environ.get("STAGGER", 150)))
    a = ap.parse_args(); LOGS.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True); SCR.mkdir(parents=True, exist_ok=True)
    if a.check: check(a.mock)
    if a.run: sys.exit(run(a.only.split(",") if a.only else None, a.workers, a.mock, not a.no_offload, a.stagger))
