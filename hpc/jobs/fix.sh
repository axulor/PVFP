#!/bin/sh
# ============================================================================
# Un_Sh -- reruns with the revised estimator (fix, 2026-10-07): 54 tasks on which the old refine widened its window
#   (48 in E2 stage 1, 2 in E1) or which sat 3-4 % off their neighbours (4).  The revised refine never widens the
#   window and never accepts a root extrapolated from a window that lies entirely on one side of zero; tasks with an
#   ordinary first window give identical results under both versions.  Results override E1/E2 by (net, rule, eps1, eps2).
#
# Layout: ONE process with 120 OpenMP threads per node (measured: ~4x faster per task than 8 threads), so the long
#   RHG tasks stay well inside the 30 min limit.
# Submit (after uploading src/model.hpp and rebuilding: cd src && make clean && make):
#   cd /path/to/PVFP/hpc
#   pjsub --bulk --sparam 1-4 jobs/fix.sh
# Sub-job k reads tasks/fix_partKK.txt and writes data/fix/partKK/rank_0000.txt; log logs/fix_<k>.log.
# ============================================================================

#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM --mpi proc=1
#PJM -L elapse=0:30:00
#PJM -j
#PJM -N UnSh_fix
#PJM -o logs/fix_%b.log

set -eu

PART=$(printf "%02d" "${PJM_BULKNUM:?submit as a bulk job: pjsub --bulk --sparam 1-4 jobs/fix.sh}")
TASKS="tasks/fix_part${PART}.txt"
OUT="data/fix/part${PART}"

export OMP_NUM_THREADS=120
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_PIN_DOMAIN=omp

echo "======================================================"
echo "Un_Sh reruns with the revised estimator (1 x 120 threads) -- part ${PART}"
echo "  Job ID     : ${PJM_JOBID:-<local>}   bulk number : ${PJM_BULKNUM}"
echo "  Host       : $(hostname)"
echo "  Date       : $(date)"
echo "  Tasks      : ${TASKS}   ->  ${OUT}"
echo "======================================================"

module load intel/2024.1
module load impi

[ -x src/bstar_mpi ] || { echo "src/bstar_mpi not found: build it on the login node first (cd src && make)"; exit 1; }
[ -f "${TASKS}" ] || { echo "missing ${TASKS}: upload tasks/fix_part*.txt"; exit 1; }
mkdir -p "${OUT}" logs

mpiexec -n 1 ./src/bstar_mpi "${TASKS}" nets "${OUT}"

echo "======================================================"
echo "fix reruns, part ${PART} finished at $(date)"
echo "======================================================"
