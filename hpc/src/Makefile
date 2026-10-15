# Makefile for bstar_mpi (Un_Sh project, design doc v5, section 3.B.5)
# Target platform: Genkai (Xeon Platinum 8490H, Intel oneAPI 2024.1).  Load the modules first:
#   module load intel/2024.1
#   module load impi
# -qopenmp: threads inside a task (OMP_NUM_THREADS); results do not depend on the number of threads.
# -fp-model=precise -ffp-contract=off: no value-changing optimisations and no FMA contraction, so the results agree
# with the Python reference up to differences in the math library (exp, pow) in the last bit.

CXX       := mpiicpx
CXXFLAGS  := -O3 -xCORE-AVX512 -qopenmp -fp-model=precise -ffp-contract=off -std=c++17 -Wall -Wextra

# ---- fallback if mpiicpx is not available (GNU compiler with Intel MPI's wrapper) ----
# CXX      := mpicxx
# CXXFLAGS := -O3 -march=sapphirerapids -fopenmp -ffp-contract=off -std=c++17 -Wall -Wextra

TARGET    := bstar_mpi

.PHONY: all clean info

all: $(TARGET)

$(TARGET): bstar_mpi.cpp model.hpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) bstar_mpi.cpp

info:
	@echo "CXX      = $(CXX)"
	@echo "CXXFLAGS = $(CXXFLAGS)"

clean:
	rm -f $(TARGET)
