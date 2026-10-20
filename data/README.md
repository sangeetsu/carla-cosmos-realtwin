# Data

Frames were rendered with the [CARLA simulator](https://github.com/carla-simulator/carla) 0.9.16 (clear noon, 1242×375,
KITTI-like camera). CARLA-specific assets are distributed under a CC-BY license; CARLA code under MIT.

- `inputs/D/<Town>/frame_XXXXXX.mp4`: each frame centred on a black 1242×684 canvas (16:9, so Cosmos's 720p mode scales it
  uniformly by 1.03 instead of squeezing it) and stored as a one-frame H.264 file. Six towns; Town01 has 1,000 frames, the others 1,050.
- `inputs/M/<Town>/`: the extra frames for the multi-control comparison (records 800–1040, every 10th).
- `inputs/repro/Town03/`: 30 frames as the identical input files used on the reference GPU, plus the original JPEGs.
- `inputs/video/`: two 35-record drives (records are 1 s apart in simulation time), letterboxed, 16 fps.
- `reference/Town03_local_A6000/`: the reference GPU's outputs (RTX A6000, same commit and settings), cropped to 1242×375.

`../tools/make_inputs_local.sh` is the exact script that produced these files.
