#### Parallel DMD  

This is a parallel implementation for the classic SVD-based dynamic mode decomposition, where the Cholesky decomposition is used to compute the SVD matrices.

Output results are only with respect to the un-tweaked amplitudes, eigenvalues, and spatial modes, and no selection criterion is specified within the code since the selection criterion can be different and flexible.

#### Dependency
* Eigen3
* MPI library with C++ bindings

#### Compiling

* `make`: compiling source code to ./bin/dmd_mpi.out
* `make clean`: remove compiled binaries

#### Usage
* `mpirun -n <cores> dmd_mpi.out <inputFile> <xn> <tn>` 
    * _cores_ is the number of processors to be requested
    * _inputFile_ should be an _xn_ by _tn+1_ matrix
    * _xn_ is the number of rows
    * _tn_ is the number of columns



