#!/usr/bin/env bash
# Reproduce every experiment of the paper. CPU jobs run as many 4-thread processes in parallel;
# E4/E7 use one GPU if available (set OT_DEVICE=cpu to force CPU).
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
J=${JOBS:-40}
export OT_THREADS=4
LOG=../logs; mkdir -p $LOG
cpu() { OT_DEVICE=cpu CUDA_VISIBLE_DEVICES= "$@"; }

# E1, E3, E2-curves: single processes
cpu $PY e1_correctness.py > $LOG/e1.log 2>&1 &
cpu $PY e3_stability.py   > $LOG/e3.log 2>&1 &
cpu $PY e2_convergence.py --part curves > $LOG/e2_curves.log 2>&1 &
# E2 statistics (30 instances) and E5 (all sci-Plex3 dose paths, 2 eps), sharded
{
  for s in $(seq 0 9); do echo "e2_convergence.py --part stats --shard $s --nshards 10"; done
  for eps in 0.05 0.01; do for L in CVCL_0031 CVCL_0023 CVCL_0004; do for s in $(seq 0 7); do
    echo "e5_warmstart.py --line $L --eps $eps --shard $s --nshards 8"; done; done; done
} | xargs -P $J -I{} bash -c "OT_DEVICE=cpu CUDA_VISIBLE_DEVICES= $PY {} > $LOG/\$(echo '{}' | tr ' ' '_' | tr -d '-').log 2>&1"
# E6 (UOT) on CPU, three parts in parallel
cpu $PY e6_unbalanced.py --part mass    > $LOG/e6_mass.log 2>&1 &
cpu $PY e6_unbalanced.py --part biology > $LOG/e6_bio.log 2>&1 &
cpu $PY e6_unbalanced.py --part sweep   > $LOG/e6_sweep.log 2>&1 &
wait
# E4 scaling and E7 atlas (GPU)
$PY e4_scaling.py > $LOG/e4.log 2>&1
$PY e7_atlas.py   > $LOG/e7.log 2>&1
