[![Build Status](https://travis-ci.org/haampie/COSMA.jl.svg?branch=master)](https://travis-ci.org/haampie/COSMA.jl)

# COSMA.jl communication optimal matrix-matrix multiplication for DistributedArrays.jl over MPI

Usage example:

```julia
using MPIClusterManager, DistributedArrays, Distributed

manager = MPIManager(np = 6)
addprocs(manager)

@everywhere using COSMA

# Some config for the mapping of Julia's pids to MPI ranks on the host, maybe we can make this automatic later
COSMA.use_manager(manager)

C = drand(100, 100)
A = drand(100, 100)
B = drand(100, 100)

mul!(C, A, B)
```
