#!/bin/sh
# ============================================================================
# Un_Sh -- validation job: 17 tasks whose results are known from the local Python code
#   (figure 2: karate club, 8 tasks; and 9 points of experiment E1).
#   Runs with the same layout as the production jobs (15 MPI ranks x 8 OpenMP threads), so it also checks that
#   the threaded build reproduces the Python results.
#
# Submit from the project root:
#   cd /path/to/PVFP/hpc
#   pjsub jobs/validate.sh
#
# Expected runtime : about 5 min (longest task: ER_k16 U, ~12 min on one core, ~2 min with 8 threads).
# Resource plan    : 1 node, 15 ranks x 8 threads, 30 min walltime.
# Output           : data/validate_v6/rank_NNNN.txt   (download this folder)
# ============================================================================

#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM --mpi proc=15
#PJM -L elapse=0:30:00
#PJM -j
#PJM -N UnSh_validate
#PJM -o logs/validate_v6.log

set -eu

export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_PIN_DOMAIN=omp

echo "======================================================"
echo "Un_Sh validation job"
echo "  Job ID     : ${PJM_JOBID:-<local>}"
echo "  Host       : $(hostname)"
echo "  Date       : $(date)"
echo "  Submit dir : $(pwd)"
echo "  MPI ranks  : ${PJM_MPI_PROC}   OpenMP threads per rank : ${OMP_NUM_THREADS}"
echo "======================================================"

module load intel/2024.1
module load impi
module list

mkdir -p data/validate_v6 logs

echo "-- build (no-op if already built on the login node) --"
cd src
make
cd ..
echo "Build OK."

echo "-- run --"
mpiexec -n "${PJM_MPI_PROC}" ./src/bstar_mpi tasks/validate.txt nets data/validate_v6

echo "======================================================"
echo "validation job finished at $(date)"
echo "======================================================"
