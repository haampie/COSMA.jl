[![Build Status](https://travis-ci.org/haampie/COSMA.jl.svg?branch=master)](https://travis-ci.org/haampie/COSMA.jl)

# COSMA.jl communication optimal matrix-matrix multiplication for DistributedArrays.jl over MPI

COSMA.jl provides wrappers for [eth-cscs/COSMA](https://github.com/eth-cscs/COSMA) to do communication-optimal matrix-matrix multiplication for DArray's of element types `Float32`, `Float64`, `ComplexF32` and `ComplexF64`.

Install via the package manager

```julia
using Pkg
Pkg.add("COSMA")
```

A typical prerequisite is to use MPIClusterManager to setup some MPI ranks and to load the package everywhere:

```julia
using MPIClusterManager, DistributedArrays, Distributed

manager = MPIManager(np = 6)
addprocs(manager)

@everywhere using COSMA

# Just on the host we have to configure the mapping of Julia's pids to MPI ranks (hopefully this can be removed in a later release)
COSMA.use_manager(manager)
```

Next create some distributed matrices and multiply them:

```julia
using LinearAlgebra

# Float64 matrices, automatically distributed over the MPI ranks
A = drand(100, 100)
B = drand(100, 100)

# Use DistributedArrays to allocate the new matrix C and multiply using COSMA
C = A * B

# Or allocate your own distributed target matrix C:
A_complex = drand(ComplexF32, 100, 100)
B_complex = drand(ComplexF32, 100, 100)
C_complex = dzeros(ComplexF32, 100, 100)

mul!(C_complex, A_complex, B_complex)
```

## Using a custom MPI implementation

COSMA.jl depends on MPI.jl, which ships MPICH as a default MPI library. If you need a system-specific version, see the instructions from the [docs of MPI.jl](https://juliaparallel.github.io/MPI.jl/latest/configuration/).

## Notes about Julia's DArray type

COSMA supports Julia's DArray matrix distribution perfectly, and is in fact more powerful: Julia's DArray supports only a single local block per MPI rank, whereas COSMA supports an arbitrary number of them.
