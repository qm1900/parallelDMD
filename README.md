#### Parallel DMD  

This is a parallel implementation for the classic dynamic mode decomposition.

The DMD modes are computed via SVD method, and the Cholesky decomposition is used to compute the SVD matrices.

#### Dependency
* Eigen3
* MPI library with C++ bindings

#### Compiling

* `make`: compiling source code computing a general input matrix
* `make clean`: remove compiled binaries

#### Usage
* `mpirun -n <cores> dmd_mpi.out <inputFile> <xn> <tn>` 
    * _cores_ is the number of processors to be requested
    * _inputFile_ should be an _xn_ by _tn+1_ matrix
    * _xn_ is the number of rows
    * _tn_ is the number of columns



