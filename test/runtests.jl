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

    @testset "Incompatible sizes" begin
        @test_throws DimensionMismatch mul!(drand(6, 6), drand(3, 6), drand(5, 6))
        @test_throws DimensionMismatch mul!(drand(ComplexF64, 6, 6), drand(ComplexF64, 6, 3)', drand(ComplexF64, 5, 6))
        @test_throws DimensionMismatch mul!(drand(6, 6), transpose(drand(6, 3)), drand(5, 6))
        @test_throws DimensionMismatch mul!(drand(4, 3), transpose(drand(3, 3)), drand(3, 3))
    end

    @testset "C = alpha * $f(A) * $g(B) + beta * C ($T)" for f = (identity, transpose, adjoint), g = (identity, transpose, adjoint), T = (Float32, Float64, ComplexF32, ComplexF64)
        # Skip adjoints of real matrices
        if (f == adjoint || g == adjoint) && T <: Real
            continue
        end

        m, n, k = 20, 10, 30

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
