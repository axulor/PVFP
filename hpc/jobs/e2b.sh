#!/bin/sh
# ============================================================================
# Un_Sh -- experiment E2, stage 2 (figure 5): refinement of the critical line Lambda = 0.
#   On every row eps1 of every network, the grid cell in eps2 where Lambda changes sign (from stage 1) gets
#   4 extra points (spacing 0.02), both rules, precision 0.2 %.  Tasks are generated locally after stage 1:
#     python hpc/tools/collect.py e2a      (results/final/e2a_hpc.csv)
#     python hpc/tools/prepare.py e2b K    (tasks/e2b_part01.txt ... e2b_partKK.txt; prints K-part estimates)
#
# Submit as a BULK job of K single-node sub-jobs (K as printed by prepare.py), from the project root:
#   cd /path/to/PVFP/hpc
#   pjsub --bulk --sparam 1-K jobs/e2b.sh
# Sub-job k reads tasks/e2b_partKK.txt and writes data/e2b/partKK/; log logs/e2b_<k>.log.
# A part that hits the limit is resubmitted alone (pjsub --bulk --sparam k-k jobs/e2b.sh); finished tasks are
# skipped.  Build on the login node beforehand (cd src && make); this script does not call make.
# ============================================================================

#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM --mpi proc=15
#PJM -L elapse=0:30:00
#PJM -j
#PJM -N UnSh_E2b
#PJM -o logs/e2b_%b.log

set -eu

PART=$(printf "%02d" "${PJM_BULKNUM:?submit as a bulk job: pjsub --bulk --sparam 1-K jobs/e2b.sh}")
TASKS="tasks/e2b_part${PART}.txt"
OUT="data/e2b/part${PART}"

export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_PIN_DOMAIN=omp

echo "======================================================"
echo "Un_Sh experiment E2, stage 2 -- part ${PART}"
echo "  Job ID     : ${PJM_JOBID:-<local>}   bulk number : ${PJM_BULKNUM}"
echo "  Host       : $(hostname)"
echo "  Date       : $(date)"
echo "  Tasks      : ${TASKS}   ->  ${OUT}"
echo "  MPI ranks  : ${PJM_MPI_PROC}   OpenMP threads per rank : ${OMP_NUM_THREADS}"
echo "======================================================"

module load intel/2024.1
module load impi

[ -x src/bstar_mpi ] || { echo "src/bstar_mpi not found: build it on the login node first (cd src && make)"; exit 1; }
[ -f "${TASKS}" ] || { echo "missing ${TASKS}: generate it with tools/prepare.py e2b and upload it"; exit 1; }
mkdir -p "${OUT}" logs

mpiexec -n "${PJM_MPI_PROC}" ./src/bstar_mpi "${TASKS}" nets "${OUT}"

echo "======================================================"
echo "E2 stage 2, part ${PART} finished at $(date)"
echo "======================================================"
