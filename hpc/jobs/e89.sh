#!/bin/sh
# ============================================================================
# Un_Sh -- experiments E8 and E9 (figure 5), standard parts: selection strength theta and production delta on the
#   uniform diagonal eps1 = eps2 = eps (tools/prepare.py e89 <round>).  Tasks predicted to take more than 20 min on
#   8 threads are in the wide parts (jobs/e89w.sh), not here.
#
# Submit as a bulk job from the project root (K = number of parts printed by prepare.py):
#   cd /path/to/PVFP/hpc
#   pjsub --bulk --sparam 1-K jobs/e89.sh
# Sub-job k reads tasks/e89_partKK.txt and writes data/e89/partKK/rank_NNNN.txt; log logs/e89_<k>.log.
#
# Layout           : 1 node, 15 MPI ranks x 8 OpenMP threads; rank 0 hands out tasks, 14 ranks compute.
# Run cap          : BSTAR_RCAP = 40960000 runs per point (2026-10-08: precision 0.2 % also at weak selection).
# Expected runtime : about 15 min per part plus the slowest task; 30 min limit.
# If a part hits the limit: resubmit only that part (pjsub --bulk --sparam k-k jobs/e89.sh).  Finished tasks are
#                    skipped and an interrupted task continues from its checkpoint (data/e89/partKK/ckpt/).
# Build src/ on the login node first (cd src && make); this script does not call make.
# ============================================================================

#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM --mpi proc=15
#PJM -L elapse=0:30:00
#PJM -j
#PJM -N UnSh_E89
#PJM -o logs/e89_%b.log

set -eu

PART=$(printf "%02d" "${PJM_BULKNUM:?submit as a bulk job: pjsub --bulk --sparam 1-K jobs/e89.sh}")
TASKS="tasks/e89_part${PART}.txt"
OUT="data/e89/part${PART}"

export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_PIN_DOMAIN=omp
export BSTAR_RCAP=40960000

echo "======================================================"
echo "Un_Sh experiments E8/E9 -- part ${PART}"
echo "  Job ID     : ${PJM_JOBID:-<local>}   bulk number : ${PJM_BULKNUM}"
echo "  Host       : $(hostname)"
echo "  Date       : $(date)"
echo "  Tasks      : ${TASKS}   ->  ${OUT}"
echo "  MPI ranks  : ${PJM_MPI_PROC}   OpenMP threads per rank : ${OMP_NUM_THREADS}   run cap : ${BSTAR_RCAP}"
echo "======================================================"

module load intel/2024.1
module load impi

[ -x src/bstar_mpi ] || { echo "src/bstar_mpi not found: build it on the login node first (cd src && make)"; exit 1; }
[ -f "${TASKS}" ] || { echo "missing ${TASKS}: upload tasks/e89_part*.txt"; exit 1; }
mkdir -p "${OUT}" logs

mpiexec -n "${PJM_MPI_PROC}" ./src/bstar_mpi "${TASKS}" nets "${OUT}"

echo "======================================================"
echo "E8/E9, part ${PART} finished at $(date)"
echo "======================================================"
