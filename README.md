# CARLA real-twin generation bundle (Cosmos-Transfer2.5)

This repository restyles simulator frames (CARLA) toward a photographic look with NVIDIA
[Cosmos-Transfer2.5-2B](https://github.com/nvidia-cosmos/cosmos-transfer2.5), and runs three small comparison jobs.
Everything is scripted, resumable and logged. You run four commands; nothing needs editing.

**Needs:** one NVIDIA GPU with 80 GB or more (H100 / H200), about 100 GB free disk, internet for the setup step,
a Hugging Face account, and `git`. No root access is needed.

## Steps

**0. Hugging Face access (5 min, once).** Sign in at huggingface.co and open each page below. Accept the NVIDIA license if asked.

- https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B
- https://huggingface.co/nvidia/Cosmos-Guardrail1
- https://huggingface.co/nvidia/Cosmos-Reason1-7B
- https://huggingface.co/nvidia/Cosmos-Predict2.5-2B

Then create a read token (Settings → Access Tokens) and make it available in the shell you use:

```bash
export HF_TOKEN=hf_your_read_token
```

**1. Get the bundle.**

```bash
git clone https://github.com/sangeetsu/carla-cosmos-realtwin.git realtwin && cd realtwin
```

On a cluster, put the large working folder on scratch storage first (optional):

```bash
export WORK=/scratch/$USER/realtwin_work
```

**2. Setup (30–60 min, no GPU needed).** Builds the Python environment and downloads about 60 GB of models.

```bash
bash setup.sh
```

**3. GPU check (10–15 min, on the GPU node).** Generates two test frames and chooses how many runs share the GPU.
Wait for `CHECK PASSED`.

```bash
bash check.sh
```

**4. The overnight run.** Leave it running; it stops by itself.

```bash
nohup bash run_overnight.sh > overnight.out 2>&1 &
```

On Slurm, edit the three marked lines in `slurm/overnight.sbatch`, then submit it with `sbatch slurm/overnight.sbatch`.
Progress shows in `results/logs/gen.log`. If the run is stopped for any reason, start the same command again;
finished frames are kept and skipped.

**5. Send the results back.** The run ends by writing `realtwin_results_<host>_<date>.tar` (about 2–4 GB) in this folder.
To pack whatever is finished at any time, run `bash pack_results.sh`. Send the `.tar` file back with any file-sharing link.

## What runs, in order

| Job | What | Samples | Rough time |
|---|---|---|---|
| J1 | Reproducibility: 30 frames a reference GPU already made, from the identical input file and from a lossless PNG | 60 | 10 min |
| J2 | Three label-preserving set-ups on 262 frames: guided generation (labeled boxes anchored for 25 or 15 of 35 steps), and edge + depth + segmentation | 786 | 2–4 h |
| J4 | Video mode on two 35-frame drives, standard model and the distilled 4-step model | 4 clips | 20–40 min |
| J3 | The main job: every frame of eight towns (8,350) with the bulk set-up, Town03 and Town05 first | 8,350 | rest of the night |
| J5 | Filler if time remains: the last 250 frames of each town again with another seed | 1,950 | whatever is left |

The main job will probably not finish in one night. That is expected: rerun `run_overnight.sh` on another night and it continues.
Times are estimates; `results/logs/timing.csv` records the real ones.
All settings are fixed in `tools/gen.py`: 720p, 35 steps, guidance 3, fixed seeds, fixed prompt.
The bulk set-up is guided generation with 25 anchored steps (`BULK_SETUP=G25`); `check.sh` tests it and falls back to edge-only (`D`) if it fails.

## Good to know

- **Guardrails stay on.** Cosmos's safety guardrails run on every prompt and output. These scripts never disable them,
  and the NVIDIA Open Model License does not allow bypassing them.
- **Memory.** `check.sh` measures GPU memory and RAM and picks 1–3 parallel runs per GPU. To force a number, use
  `WORKERS=1 bash run_overnight.sh`. Several GPUs on one node are used automatically.
- **GPU nodes without internet.** Run `setup.sh` on a login node with internet, using the same `WORK` folder, then run
  `export HF_HUB_OFFLINE=1` before `check.sh` and `run_overnight.sh`.
- **Low-RAM machines.** On machines with less than 64 GB RAM, `setup.sh` applies a small patch, `patches/text_encoder_lowram.patch`,
  that streams the text-encoder weights to the GPU. It changes no weights.
- **Only some jobs.** For example, `JOBS=J3 bash run_overnight.sh`, or `BULK_TOWNS="Town03 Town05" ...` for some towns.
  Keep the same `BULK_SETUP` on every night of a multi-night run.
- **Dry run.** After `setup.sh`, `bash run_overnight.sh --mock` exercises all the plumbing in a few minutes with a stand-in generator,
  no GPU needed. It writes only to `results_mock/` and never affects the real run.

## Layout

```
setup.sh  check.sh  run_overnight.sh  pack_results.sh   the four commands
env.sh                                                  paths and versions (Cosmos commit 2ff49d07, uv 0.13.4, Python 3.10)
tools/gen.py                                            job definitions, scheduling, cropping, retries
tools/summarize.py                                      morning summary + contact sheet (results/summary.txt)
tools/predownload.py                                    Hugging Face access check and model downloads
tools/make_inputs_local.sh                              how data/ was made (provenance only)
data/inputs/                                            input frames (letterboxed one-frame H.264) and guided-generation masks
tools/make_masks_local.py                               how the masks were made (provenance only)
data/reference/                                         outputs of the reference GPU for the reproducibility job
```

## License

Scripts: MIT (see `LICENSE`). Input frames: rendered with the CARLA simulator, whose assets are distributed under a CC-BY
license (see `data/README.md`). Cosmos code and models are NVIDIA's and are downloaded under their own licenses.
