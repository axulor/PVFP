#!/bin/sh
# ============================================================================
# Un_Sh -- experiment E2, reruns (e2r): stage-1 tasks recomputed with the polished estimator (src/model.hpp,
#   2026-10-07).  60 tasks: 48 whose refine window had been widened (a few were up to 25 % off) and 12 left
#   unfinished at the 30 min limit (RHG_g3.0_T0.1, uniform rule, eps1 >= 1.1, eps2 >= 1.0; 17-40 min each with
#   8 threads).  Results replace the stage-1 values of the same (net, rule, eps1, eps2).
#
# Layout: ONE process with 120 OpenMP threads per node; the tasks of a part run one after another, each on the
#   whole node (about 10x faster per task than with 8 threads), so a long task does not hit the time limit.
#
# Submit as a bulk job of 6 single-node sub-jobs, from the project root (after rebuilding: cd src && make):
#   cd /path/to/PVFP/hpc
#   pjsub --bulk --sparam 1-6 jobs/e2r.sh
# Sub-job k reads tasks/e2r_partKK.txt and writes data/e2r/partKK/rank_0000.txt; log logs/e2r_<k>.log.
# A part that hits the limit is resubmitted alone (pjsub --bulk --sparam k-k jobs/e2r.sh); finished tasks are skipped.
# ============================================================================

#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM --mpi proc=1
#PJM -L elapse=0:30:00
#PJM -j
#PJM -N UnSh_E2r
#PJM -o logs/e2r_%b.log

set -eu

PART=$(printf "%02d" "${PJM_BULKNUM:?submit as a bulk job: pjsub --bulk --sparam 1-6 jobs/e2r.sh}")
TASKS="tasks/e2r_part${PART}.txt"
OUT="data/e2r/part${PART}"

export OMP_NUM_THREADS=120
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_PIN_DOMAIN=omp

echo "======================================================"
echo "Un_Sh experiment E2 reruns (1 x 120 threads) -- part ${PART}"
echo "  Job ID     : ${PJM_JOBID:-<local>}   bulk number : ${PJM_BULKNUM}"
echo "  Host       : $(hostname)"
echo "  Date       : $(date)"
echo "  Tasks      : ${TASKS}   ->  ${OUT}"
echo "======================================================"

module load intel/2024.1
module load impi

[ -x src/bstar_mpi ] || { echo "src/bstar_mpi not found: build it on the login node first (cd src && make)"; exit 1; }
[ -f "${TASKS}" ] || { echo "missing ${TASKS}: upload tasks/e2r_part*.txt"; exit 1; }
mkdir -p "${OUT}" logs

mpiexec -n 1 ./src/bstar_mpi "${TASKS}" nets "${OUT}"

echo "======================================================"
echo "E2 reruns, part ${PART} finished at $(date)"
echo "======================================================"
