#!/bin/bash
cd "$(dirname "$0")"
run() { python3 steps/04_quantize_unlearn.py --arch $1 --seed $2 --bits 8,6,4; }
( for s in 5 6; do run dlinear $s; done
  while [ ! -f checkpoints/mlp/seed5_retrained.pt ]; do sleep 20; done
  for s in 4 5; do run mlp $s; done ) > results/runA.log 2>&1 &
( while [ ! -f checkpoints/patchtst/seed2_retrained.pt ]; do sleep 20; done
  run patchtst 2 ) > results/runB.log 2>&1 &
