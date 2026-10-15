#!/bin/sh
# ============================================================================
# Un_Sh -- experiments E8/E9 (figure 5): RESUME the tasks of a standard part (jobs/e89.sh) that were still running
#   when that part hit its 30-min limit.  Same task file and output folder as the part (tasks/e89_partKK.txt,
#   data/e89/partKK/), but in wide mode on one node: all 15 ranks work on one unfinished task at a time and continue
#   from its checkpoint (data/e89/partKK/ckpt/).  Finished tasks are skipped.  Results are bit-identical.
#
# Submit one bulk job per part to be resumed (K = part number), from the project root:
#   cd /path/to/PVFP/hpc
#   pjsub --bulk --sparam K-K jobs/e89r.sh
# Log logs/e89r_<K>.log; results in data/e89/partKK/rank_0000.txt (next to the rank files of the standard run).
#
# Layout  : 1 node, 15 MPI ranks x 8 OpenMP threads, all on one task (about 11x one 8-thread worker, measured).
# Run cap : BSTAR_RCAP = 40960000 (must equal the standard parts: the checkpoints are only valid for this cap).
# If it hits the limit again: resubmit the same command; the task continues from its newest checkpoint.
# ============================================================================

#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM --mpi proc=15
#PJM -L elapse=0:30:00
#PJM -j
#PJM -N UnSh_E89r
#PJM -o logs/e89r_%b.log

set -eu

PART=$(printf "%02d" "${PJM_BULKNUM:?submit as a bulk job: pjsub --bulk --sparam K-K jobs/e89r.sh}")
TASKS="tasks/e89_part${PART}.txt"
OUT="data/e89/part${PART}"

export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_PIN_DOMAIN=omp
export BSTAR_RCAP=40960000
export BSTAR_WIDE=1

echo "======================================================"
echo "Un_Sh experiments E8/E9 -- resume part ${PART} in wide mode"
echo "  Job ID     : ${PJM_JOBID:-<local>}   bulk number : ${PJM_BULKNUM}"
echo "  Host       : $(hostname)"
echo "  Date       : $(date)"
echo "  Tasks      : ${TASKS}   ->  ${OUT}"
echo "  MPI ranks  : ${PJM_MPI_PROC}   OpenMP threads per rank : ${OMP_NUM_THREADS}   run cap : ${BSTAR_RCAP}"
echo "  Checkpoints: $(ls ${OUT}/ckpt 2>/dev/null | tr '\n' ' ')"
echo "======================================================"

module load intel/2024.1
module load impi

[ -x src/bstar_mpi ] || { echo "src/bstar_mpi not found: build it on the login node first (cd src && make)"; exit 1; }
[ -d "${OUT}" ] || { echo "missing ${OUT}: this script resumes a part that has already run"; exit 1; }

mpiexec -n "${PJM_MPI_PROC}" ./src/bstar_mpi "${TASKS}" nets "${OUT}"

echo "======================================================"
echo "E8/E9 resume of part ${PART} finished at $(date)"
echo "======================================================"
