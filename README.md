# CARLA real-twin generation bundle (Cosmos-Transfer2.5)

This repository restyles simulator frames (CARLA) toward a photographic look with NVIDIA
[Cosmos-Transfer2.5-2B](https://github.com/nvidia-cosmos/cosmos-transfer2.5), after a few small comparison jobs.
Everything is scripted, resumable and logged; nothing needs editing.

**Needs:** one NVIDIA GPU with at least 48 GB of memory on an x86_64 Linux machine, about 100 GB of free disk,
internet for the setup step, a Hugging Face account, and `git`. No root access is needed.
The instructions below are for ASU's Sol cluster; any Slurm cluster or a single workstation works the same way.

## Steps on ASU Sol

**0. Hugging Face access (5 min, once).** Sign in at huggingface.co and open each page below. Accept the NVIDIA license if asked.

- https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B
- https://huggingface.co/nvidia/Cosmos-Guardrail1
- https://huggingface.co/nvidia/Cosmos-Reason1-7B
- https://huggingface.co/nvidia/Cosmos-Predict2.5-2B

Then create a read token (Settings → Access Tokens). Put it in your shell before `setup.sh`:

```bash
export HF_TOKEN=hf_your_read_token
```

**1. Get the bundle onto Sol** and keep the large working folder on scratch storage, not in your home folder:

```bash
git clone https://github.com/sangeetsu/carla-cosmos-realtwin.git realtwin && cd realtwin
```

```bash
export WORK=/scratch/$USER/realtwin_work
```

Use the same `WORK` value in every later step (add the export line to `~/.bashrc` to make that automatic).

**2. Setup (30–60 min, no GPU).** Sol's `lightwork` partition is meant for building environments:

```bash
srun -p lightwork -c 4 -t 4:00:00 --pty bash
```

Inside that session, from the `realtwin` folder, with `WORK` and `HF_TOKEN` exported:

```bash
bash setup.sh
```

If the downloads fail there for lack of internet, run `setup.sh` on the login node instead.

Sol's operating system has an older C library (glibc 2.28) than NVIDIA's Transformer Engine needs (2.29), which shows up as
`GLIBC_2.29 not found`. `setup.sh` therefore also pulls a small Ubuntu 22.04 Apptainer image into `WORK`, and the GPU scripts
re-run themselves inside it automatically with the GPU passed through. Apptainer exists on Sol's compute nodes, which is
another reason to run `setup.sh` on `lightwork`. If the pull fails, Sol's own image works too: list the images with `showsimg`
and `export SIF=/path/to/ubuntu-22.04.sif`. Do not `module load` CUDA or Python first; the environment brings its own.

Already hit `GLIBC_2.29 not found`? Update the bundle and rerun setup. It keeps what is already installed and adds the image:

```bash
git pull && bash setup.sh
```

Then submit the GPU job again.

**3. Submit the GPU jobs.** This queues four chained one-day jobs; each continues where the previous stopped:

```bash
bash slurm/submit_chain.sh 4
```

To use two GPUs for 16 hours in a single job instead, pass the options through:

```bash
bash slurm/submit_chain.sh 1 -G a100:2 --cpus-per-task=16 --mem=128G --time=16:00:00
```

Run `setup.sh` before the GPU job starts; it needs no GPU and the job stops at once without it.
The first job runs a 15-minute GPU check before anything else. Its log, `realtwin_<jobid>.out`, must show `CHECK PASSED`.
If it shows `CHECK FAILED`, the GPU was probably too small; see "Choosing a GPU" below.

**4. Watch progress.** `squeue -u $USER` shows the jobs. `tail -f results/logs/gen.log` shows each finished batch.
`results/summary.txt` is refreshed at the end of every job. If the four jobs end before the work does, run the submit command again.

**5. Send the results back.** When everything is done, the last job writes `realtwin_results_<host>_<date>.tar` in this folder
(a few GB). To send what exists earlier, run `bash pack_results.sh` at any time. Any file-sharing link is fine.

## Choosing a GPU on Sol

The heaviest set-up needs about 40 GB of GPU memory, measured on the reference machine, so `slurm/sol.sbatch` asks for an
80 GB A100 (`-G a100:1` with `-C a100_80`, on the `public` partition, one-day jobs).

| Sol GPU | Works? |
|---|---|
| A100 80 GB (`-G a100:1 -C a100_80`) | Yes: the default |
| H100 96 GB, on lab-owned nodes | Yes, if your lab owns some: `-G h100:1`, `--partition=general --qos=grp_<labname>`, no `-C` line |
| A100 40 GB (`a100_40`) | No: too little memory for the main set-up |
| A30 24 GB, A100 MIG slices | No: too little memory |
| GH200 (`arm` partition), MI200 (`fpga` partition) | No: ARM CPU or AMD GPU; the software needs x86_64 and NVIDIA CUDA |

Runs are one at a time per GPU. A job with several GPUs (`-G a100:2`) uses all of them in parallel, taking the GPU numbers
Slurm assigns; allow about 64 GB of `--mem` per GPU. On an 80 GB or larger card
you may also allow two runs per GPU with `MAX_SLOTS_PER_GPU=2`; `check.sh` still caps this by the measured memory.
Sol's `public` partition allows up to 7 days per job and `htc` up to 4 hours; shorter jobs usually start sooner.
`general` nodes used with the `private` QoS can be cancelled when their owners need them. Nothing is lost: resubmit and the run continues.

## How long it takes, and why nothing is lost

On the reference GPU (RTX A6000), one frame takes about 26 s with edge control and about 35 s with edge + depth + segmentation.
The whole default run is roughly 4 days of GPU time at that speed; an A100 should be faster. The first job's
`results/logs/timing.csv` shows the real speed.

Every finished frame is saved immediately. When a job hits its time limit or is cancelled, at most the frames in progress
are lost, and the next run picks up from there. Frames are never generated twice. Town03 and Town05 come first in the
main job, so they are usable long before the rest.

## What runs, in order

| Job | What | Samples | GPU time on the reference GPU |
|---|---|---|---|
| J1 | Reproducibility: 30 frames the reference GPU already made, from the identical input file and from a lossless PNG | 60 | 25 min |
| J2 | Four label-preserving set-ups on 262 frames: edge + depth + segmentation; NVIDIA's four-control car recipe; edge + light blur; boxes pasted back (background-only restyle) | 1,048 | about 9 h |
| J4 | Video mode on two 35-frame drives, standard model and the distilled 4-step model | 4 clips | under 1 h |
| J3 | The main job: every frame of eight towns with the bulk set-up, Town03 and Town05 first | 8,350 | about 80 h |
| J5 | Only on request (`JOBS=J1,J2,J4,J3,J5`): the last 250 frames of each town again with another seed | 1,950 | about 19 h |

All settings are fixed in `tools/gen.py`: 720p, 35 steps, guidance 3, fixed seeds, fixed prompt.
The bulk set-up is edge + depth + segmentation (`BULK_SETUP=M1`). `check.sh` tests it and falls back to edge-only (`D`) if it fails.
Video mode (J4) needs about 65 GB of GPU memory according to NVIDIA; on smaller cards it fails on its own and the rest continues.
Cosmos's video guardrail occasionally rejects an ordinary street frame. The run retries such frames with new seeds; a handful may stay missing.

## Good to know

- **Guardrails stay on.** Cosmos's safety guardrails run on every prompt and output. These scripts never disable them,
  and the NVIDIA Open Model License does not allow bypassing them.
- **One run per GPU.** That is the default. Several GPUs in one job are used in parallel automatically. `MAX_SLOTS_PER_GPU=2`
  allows two runs per large GPU, and `WORKERS=N` forces a total.
- **GPU nodes without internet.** Handled automatically: when a node cannot reach huggingface.co, the scripts switch to offline mode
  and use the models `setup.sh` downloaded into `WORK`.
- **Low-RAM machines.** On machines with less than 64 GB RAM, `setup.sh` applies a small patch, `patches/text_encoder_lowram.patch`,
  that streams the text-encoder weights to the GPU. It changes no weights.
- **Only some jobs.** For example, `JOBS=J3 bash run_overnight.sh`, or `BULK_TOWNS="Town03 Town05" ...` for some towns.
  Keep the same `BULK_SETUP` for every job of one run.
- **Without Slurm.** On a workstation, `bash check.sh`, then `nohup bash run_overnight.sh > run.out 2>&1 &`; rerun it to continue.
- **Dry run.** After `setup.sh`, `bash run_overnight.sh --mock` exercises all the plumbing in a few minutes with a stand-in generator,
  no GPU needed. It writes only to `results_mock/` and never affects the real run.

## Layout

```
setup.sh  check.sh  run_overnight.sh  pack_results.sh   setup, GPU check, the (resumable) run, packing
slurm/sol.sbatch  slurm/submit_chain.sh                 one Sol GPU job, and a chain of them
env.sh                                                  paths and versions (Cosmos commit 2ff49d07, uv 0.13.4, Python 3.10)
tools/gen.py                                            job definitions, scheduling, cropping, retries
tools/summarize.py                                      progress summary + contact sheet (results/summary.txt)
tools/predownload.py                                    Hugging Face access check and model downloads
tools/make_inputs_local.sh                              how data/ was made (provenance only)
data/inputs/                                            input frames (letterboxed one-frame H.264) and guided-generation masks
tools/make_masks_local.py                               how the masks were made (provenance only)
data/reference/                                         outputs of the reference GPU for the reproducibility job
```

## License

Scripts: MIT (see `LICENSE`). Input frames: rendered with the CARLA simulator, whose assets are distributed under a CC-BY
license (see `data/README.md`). Cosmos code and models are NVIDIA's and are downloaded under their own licenses.
