#!/bin/sh
# ============================================================================
# Un_Sh -- experiment E1 (figures 3 and 4): uniform visibility eps1 = eps2 = eps in {0, 0.1, 0.2, 0.3, 1},
#   40 networks x 5 eps x 2 allocation rules = 400 critical-benefit estimates (precision 0.2 %).
#
# Submit from the project root (after the validation job has passed):
#   cd /path/to/PVFP/hpc
#   pjsub jobs/e1.sh
#
# Layout           : 1 node, 15 MPI ranks x 8 OpenMP threads = 120 cores.  Rank 0 hands out tasks; the other
#                    14 ranks each run one task at a time with 8 threads (the independent runs of a task are
#                    split over the threads; results do not depend on the number of threads).
# Expected runtime : about 20-30 core-hours in total -> roughly 15-30 min wall time; the 1 h limit leaves room for
#                    the slowest tasks (dense networks whose run count escalates to the cap).
#                    If the job hits the walltime, resubmit the same script: finished tasks are skipped
#                    (a task that was running when the job was killed starts again from the beginning).
# Output           : data/e1/rank_NNNN.txt   (download this folder)
# ============================================================================

#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM --mpi proc=15
#PJM -L elapse=1:00:00
#PJM -j
#PJM -N UnSh_E1
#PJM -o logs/e1.log

set -eu

export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_PIN_DOMAIN=omp

echo "======================================================"
echo "Un_Sh experiment E1"
echo "  Job ID     : ${PJM_JOBID:-<local>}"
echo "  Host       : $(hostname)"
echo "  Date       : $(date)"
echo "  Submit dir : $(pwd)"
echo "  MPI ranks  : ${PJM_MPI_PROC}   OpenMP threads per rank : ${OMP_NUM_THREADS}"
echo "======================================================"

module load intel/2024.1
module load impi
module list

mkdir -p data/e1 logs

echo "-- build (no-op if already built on the login node) --"
cd src
make
cd ..
echo "Build OK."

echo "-- run --"
mpiexec -n "${PJM_MPI_PROC}" ./src/bstar_mpi tasks/e1.txt nets data/e1

echo "======================================================"
echo "E1 job finished at $(date)"
echo "======================================================"
