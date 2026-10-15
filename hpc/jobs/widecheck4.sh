#!/bin/sh
# ============================================================================
# Un_Sh -- test of the wide mode (2026-10-08) on FOUR nodes: the same seven tasks as jobs/widecheck1.sh, all 60
#   ranks on one task at a time.  Results must be identical to E1/E3a; the times give the scaling over nodes.
#
# Submit from the project root:   pjsub jobs/widecheck4.sh       (a few minutes)
# Output: data/widecheck4/rank_0000.txt, log logs/widecheck4.log
# ============================================================================

#PJM -L rscgrp=a-batch
#PJM -L node=4
#PJM --mpi proc=60
#PJM -L elapse=0:30:00
#PJM -j
#PJM -N UnSh_wide4
#PJM -o logs/widecheck4.log

set -eu

export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_PIN_DOMAIN=omp
export BSTAR_WIDE=1

echo "Un_Sh wide-mode check, 4 nodes -- job ${PJM_JOBID:-<local>} on $(hostname), $(date)"
module load intel/2024.1
module load impi
[ -x src/bstar_mpi ] || { echo "src/bstar_mpi not found: build it on the login node first (cd src && make)"; exit 1; }
mkdir -p data/widecheck4 logs
mpiexec -n "${PJM_MPI_PROC}" ./src/bstar_mpi tasks/widecheck.txt nets data/widecheck4
echo "wide-mode check (4 nodes) finished at $(date)"
