MPICXX = mpicxx
SPECDEP = makefile
FLAG = ~/include/eigen3/
STD = -std=c++11

general: src/dmd_mpi.cpp
	$(MPICXX) -I $(FLAG) $(STD) src/dmd_mpi.cpp -o bin/dmd_mpi.out
clean:
	rm bin/dmd_mpi.out
