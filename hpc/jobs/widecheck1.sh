#!/bin/sh
# ============================================================================
# Un_Sh -- test of the wide mode (2026-10-08) on ONE node: seven tasks of E1/E3a with their original definitions
#   (same seeds, default run cap).  All 15 ranks work on one task at a time (BSTAR_WIDE=1).
#   Results must be identical to E1/E3a; the time per task, compared with the 8-thread times of E1/E3a, gives the
#   speed of one wide node.  (tools/prepare.py widecheck)
#
# Submit from the project root:   pjsub jobs/widecheck1.sh       (about 5 min)
# Output: data/widecheck1/rank_0000.txt, log logs/widecheck1.log
# ============================================================================

#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM --mpi proc=15
#PJM -L elapse=0:30:00
#PJM -j
#PJM -N UnSh_wide1
#PJM -o logs/widecheck1.log

set -eu

export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_PIN_DOMAIN=omp
export BSTAR_WIDE=1

echo "Un_Sh wide-mode check, 1 node -- job ${PJM_JOBID:-<local>} on $(hostname), $(date)"
module load intel/2024.1
module load impi
[ -x src/bstar_mpi ] || { echo "src/bstar_mpi not found: build it on the login node first (cd src && make)"; exit 1; }
mkdir -p data/widecheck1 logs
mpiexec -n "${PJM_MPI_PROC}" ./src/bstar_mpi tasks/widecheck.txt nets data/widecheck1
echo "wide-mode check (1 node) finished at $(date)"
