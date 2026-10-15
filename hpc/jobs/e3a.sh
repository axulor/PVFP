#!/bin/sh
# ============================================================================
# Un_Sh -- experiment E3a (figure 7): the 35 main networks not in E2 x visibility settings
#   (eps1, eps2) in {(1,0), (0,1), (1,0.5), (0.5,0.5)} x 2 allocation rules = 280 critical-benefit estimates
#   (theta = delta = 1, precision 0.2 %).  The five E2 networks already have these points.
#   (tools/prepare.py e3a 1)
#
# Submit as a bulk job with one part, from the project root:
#   cd /path/to/PVFP/hpc
#   pjsub --bulk --sparam 1-1 jobs/e3a.sh
# Sub-job k reads tasks/e3a_partKK.txt and writes data/e3a/partKK/rank_NNNN.txt; log logs/e3a_<k>.log.
#
# Layout           : as E1 -- 1 node, 15 MPI ranks x 8 OpenMP threads; rank 0 hands out tasks, 14 ranks compute.
# Expected runtime : about 11 min plus the slowest task (estimated ~6 min) -> 15-20 min; 30 min limit.
# If it hits the limit: resubmit the same command; finished tasks are skipped, a task that was running starts again.
# The program is not rebuilt here: src/ is unchanged since the third E2 round (built on the login node then).
# ============================================================================

#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM --mpi proc=15
#PJM -L elapse=0:30:00
#PJM -j
#PJM -N UnSh_E3a
#PJM -o logs/e3a_%b.log

set -eu

PART=$(printf "%02d" "${PJM_BULKNUM:?submit as a bulk job: pjsub --bulk --sparam 1-1 jobs/e3a.sh}")
TASKS="tasks/e3a_part${PART}.txt"
OUT="data/e3a/part${PART}"

export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_PIN_DOMAIN=omp

echo "======================================================"
echo "Un_Sh experiment E3a -- part ${PART}"
echo "  Job ID     : ${PJM_JOBID:-<local>}   bulk number : ${PJM_BULKNUM}"
echo "  Host       : $(hostname)"
echo "  Date       : $(date)"
echo "  Tasks      : ${TASKS}   ->  ${OUT}"
echo "  MPI ranks  : ${PJM_MPI_PROC}   OpenMP threads per rank : ${OMP_NUM_THREADS}"
echo "======================================================"

module load intel/2024.1
module load impi

[ -x src/bstar_mpi ] || { echo "src/bstar_mpi not found: build it on the login node first (cd src && make)"; exit 1; }
[ -f "${TASKS}" ] || { echo "missing ${TASKS}: upload tasks/e3a_part*.txt"; exit 1; }
mkdir -p "${OUT}" logs

mpiexec -n "${PJM_MPI_PROC}" ./src/bstar_mpi "${TASKS}" nets "${OUT}"

echo "======================================================"
echo "E3a, part ${PART} finished at $(date)"
echo "======================================================"
