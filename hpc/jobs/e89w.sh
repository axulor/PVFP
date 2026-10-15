#!/bin/sh
# ============================================================================
# Un_Sh -- experiments E8 and E9 (figure 5), WIDE parts: the long tasks (predicted > 20 min on 8 threads, mostly
#   theta = 0.1).  Wide mode (BSTAR_WIDE=1): all 60 ranks of the 4 nodes work on ONE task at a time; every rank does
#   a share of the runs and the counts are summed (MPI_Allreduce), so the results are bit-identical to the standard
#   layout.  (tools/prepare.py e89 <round>)
#
# Submit as a bulk job from the project root (K = number of wide parts printed by prepare.py):
#   cd /path/to/PVFP/hpc
#   pjsub --bulk --sparam 1-K jobs/e89w.sh
# Sub-job k reads tasks/e89w_partKK.txt and writes data/e89w/partKK/rank_0000.txt; log logs/e89w_<k>.log.
#
# Layout           : 4 nodes, 15 MPI ranks x 8 OpenMP threads per node (60 ranks, 480 cores), all on one task.
# Run cap          : BSTAR_RCAP = 40960000 runs per point.
# Expected runtime : about 18 min per part (assumed speed: 12 x one 8-thread worker per node; measured in round 1
#                    by jobs/widecheck*.sh); 30 min limit.
# If a part hits the limit: resubmit only that part (pjsub --bulk --sparam k-k jobs/e89w.sh).  Finished tasks are
#                    skipped and the interrupted task continues from its checkpoint (data/e89w/partKK/ckpt/).
# Build src/ on the login node first (cd src && make); this script does not call make.
# ============================================================================

#PJM -L rscgrp=a-batch
#PJM -L node=4
#PJM --mpi proc=60
#PJM -L elapse=0:30:00
#PJM -j
#PJM -N UnSh_E89w
#PJM -o logs/e89w_%b.log

set -eu

PART=$(printf "%02d" "${PJM_BULKNUM:?submit as a bulk job: pjsub --bulk --sparam 1-K jobs/e89w.sh}")
TASKS="tasks/e89w_part${PART}.txt"
OUT="data/e89w/part${PART}"

export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_PIN_DOMAIN=omp
export BSTAR_RCAP=40960000
export BSTAR_WIDE=1

echo "======================================================"
echo "Un_Sh experiments E8/E9, wide -- part ${PART}"
echo "  Job ID     : ${PJM_JOBID:-<local>}   bulk number : ${PJM_BULKNUM}"
echo "  Host       : $(hostname)"
echo "  Date       : $(date)"
echo "  Tasks      : ${TASKS}   ->  ${OUT}"
echo "  MPI ranks  : ${PJM_MPI_PROC}   OpenMP threads per rank : ${OMP_NUM_THREADS}   run cap : ${BSTAR_RCAP}"
echo "======================================================"

module load intel/2024.1
module load impi

[ -x src/bstar_mpi ] || { echo "src/bstar_mpi not found: build it on the login node first (cd src && make)"; exit 1; }
[ -f "${TASKS}" ] || { echo "missing ${TASKS}: upload tasks/e89w_part*.txt"; exit 1; }
mkdir -p "${OUT}" logs

mpiexec -n "${PJM_MPI_PROC}" ./src/bstar_mpi "${TASKS}" nets "${OUT}"

echo "======================================================"
echo "E8/E9 wide, part ${PART} finished at $(date)"
echo "======================================================"
