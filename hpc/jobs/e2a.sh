#!/bin/sh
# ============================================================================
# Un_Sh -- experiment E2, stage 1 (figure 5): phase diagrams Lambda(eps1, eps2) on a 16 x 16 grid (step 0.1, 0..1.5)
#   for BA_m4, HK_m4_p0.9, RHG_g3.0_T0.1 and CPH, both allocation rules, precision 0.2 %.
#   2048 grid estimates minus the 40 diagonal points already computed in E1 = 2008 tasks, split into 10 parts of
#   similar estimated cost (tools/prepare.py e2a 10).
#
# Submit as a BULK job (10 independent single-node sub-jobs), from the project root:
#   cd /path/to/PVFP/hpc
#   pjsub --bulk --sparam 1-10 jobs/e2a.sh
# Sub-job k reads tasks/e2a_partKK.txt and writes data/e2a/partKK/rank_NNNN.txt; log logs/e2a_<k>.log.
#
# Layout           : as E1 -- 1 node, 15 MPI ranks x 8 OpenMP threads; rank 0 hands out tasks, 14 ranks compute.
# Expected runtime : about 12 min per part plus the slowest task (estimated from E1, up to ~6 min) -> 15-20 min;
#                    the 30 min limit keeps the sub-jobs short enough to start without queueing.
# If a part hits the limit: resubmit only that part, e.g. part 3:   pjsub --bulk --sparam 3-3 jobs/e2a.sh
#                    finished tasks are skipped; a task that was running when the job was killed starts again.
# The program must be built on the login node beforehand (cd src && make): ten sub-jobs compiling at the same
# time in the same folder could overwrite each other, so this script does not call make.
# ============================================================================

#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM --mpi proc=15
#PJM -L elapse=0:30:00
#PJM -j
#PJM -N UnSh_E2a
#PJM -o logs/e2a_%b.log

set -eu

PART=$(printf "%02d" "${PJM_BULKNUM:?submit as a bulk job: pjsub --bulk --sparam 1-10 jobs/e2a.sh}")
TASKS="tasks/e2a_part${PART}.txt"
OUT="data/e2a/part${PART}"

export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_PIN_DOMAIN=omp

echo "======================================================"
echo "Un_Sh experiment E2, stage 1 -- part ${PART}"
echo "  Job ID     : ${PJM_JOBID:-<local>}   bulk number : ${PJM_BULKNUM}"
echo "  Host       : $(hostname)"
echo "  Date       : $(date)"
echo "  Tasks      : ${TASKS}   ->  ${OUT}"
echo "  MPI ranks  : ${PJM_MPI_PROC}   OpenMP threads per rank : ${OMP_NUM_THREADS}"
echo "======================================================"

module load intel/2024.1
module load impi

[ -x src/bstar_mpi ] || { echo "src/bstar_mpi not found: build it on the login node first (cd src && make)"; exit 1; }
[ -f "${TASKS}" ] || { echo "missing ${TASKS}: upload tasks/e2a_part*.txt"; exit 1; }
mkdir -p "${OUT}" logs

mpiexec -n "${PJM_MPI_PROC}" ./src/bstar_mpi "${TASKS}" nets "${OUT}"

echo "======================================================"
echo "E2 stage 1, part ${PART} finished at $(date)"
echo "======================================================"
