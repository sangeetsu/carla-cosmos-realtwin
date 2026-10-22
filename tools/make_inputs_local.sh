#!/usr/bin/env bash
# Provenance: how data/ was built on the source workstation (2026-10-20). Not needed on the GPU machine.
# Letterbox = the 1242x375 CARLA frame centred on a black 1242x684 canvas (16:9 at Cosmos "720", uniform x1.03), one-frame H.264 mp4,
# exactly the command used for the local runs (ffmpeg 4.4.2), so inputs are bit-identical to the local pilot's.
set -u; SRC=/mnt/features/carla_doses; B="$(cd "$(dirname "$0")/.." && pwd)"; D=$B/data
enc1(){ ffmpeg -loglevel error -y -loop 1 -i "$1" -t 0.1 -r 10 -vf "pad=1242:684:0:154:black" -c:v libx264 -pix_fmt yuv420p "$2"; }
export -f enc1
# 1. real-twin inputs: all eight towns, every record (Town01 has 1000, the others 1050)
for T in Town03 Town05 Town10HD Town07 Town01 Town02 Town04 Town06; do mkdir -p $D/inputs/D/$T
  for k in $(seq 0 1049); do f=$(printf "frame_%06d" $k); [ -s $D/inputs/D/$T/$f.mp4 ] || echo "$SRC/${T}_d00_ref/$f.jpg $D/inputs/D/$T/$f.mp4"; done
done | xargs -P 4 -n 2 bash -c 'enc1 "$0" "$1"'
# 2. reproducibility: Town03 records 0..29 — the local pilot's own input mp4s, the original jpgs, and the local A6000 outputs (cropped to 1242x375)
mkdir -p $D/inputs/repro/Town03 $D/reference/Town03_local_A6000
for k in $(seq 0 29); do f=$(printf "frame_%06d" $k)
  cp /mnt/features/cosmos/pilot/gen_Town03/in/$f.mp4 $D/inputs/repro/Town03/$f.mp4; cp $SRC/Town03_d00_ref/$f.jpg $D/inputs/repro/Town03/$f.jpg
  cp /mnt/features/cosmos/pilot/Town03_d50_realtwin/$f.jpg $D/reference/Town03_local_A6000/$f.jpg; done
# 3. video clips: 35 consecutive records (1 s apart in simulation time), letterboxed, 16 fps
mkdir -p $D/inputs/video
clip(){ local T=$1 s=$2 n=$3; ffmpeg -loglevel error -y -framerate 16 -start_number $s -i $SRC/${T}_d00_ref/frame_%06d.jpg -frames:v 35 -vf "pad=1242:684:0:154:black" -c:v libx264 -pix_fmt yuv420p $D/inputs/video/$n.mp4; }
clip Town03 70 Town03_r070-104; clip Town10HD 140 Town10HD_r140-174
echo "inputs: $(find $D/inputs -name '*.mp4' | wc -l) mp4, $(du -sh $D | cut -f1)"
echo "inputs: $(find $D/inputs -name '*.mp4' | wc -l) mp4, $(du -sh $D | cut -f1)"
# 4. guided-generation masks: tools/make_masks_local.py
