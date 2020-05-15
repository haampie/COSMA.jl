using MPI
using MPIClusterManagers
using Distributed
using DistributedArrays
using Test
using LinearAlgebra

let
    manager = MPIManager(np = 6)
    addprocs(manager)

    @everywhere using COSMA

    COSMA.use_manager(manager)

    @testset for f = (identity, transpose, adjoint), g = (identity, transpose, adjoint), T = (Float32, Float64, ComplexF32, ComplexF64)
        # Skip adjoints of real matrices
        if (f == adjoint || g == adjoint) && T <: Real
            continue
        end

        m, n, k = 200, 100, 300

        alpha = T(2.0)
        beta = T(3.0)

        # Distributed arrays
        C_d = drand(T, m, n)
        A_d = f(f != identity ? drand(T, k, m) : drand(T, m, k))
        B_d = g(g != identity ? drand(T, n, k) : drand(T, k, n))

        # Do the same computation locally
        C_h = collect(C_d)
        A_h = collect(A_d)
        B_h = collect(B_d)

        # Reference
        mul!(C_h, A_h, B_h, alpha, beta)
        mul!(C_d, A_d, B_d, alpha, beta)

        # Test by copying to the first node
        @test all(collect(C_d) .≈ C_h)
    end
end