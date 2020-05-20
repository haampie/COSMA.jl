module COSMA

import LinearAlgebra: mul!
using LinearAlgebra: BlasFloat, BlasReal, BlasComplex, Adjoint, Transpose
using DistributedArrays
using Distributed
using MPI
using MPIClusterManagers: MPIManager
using COSMA_jll: cosma

const DMatrix{T} = DArray{T,2}

const mpimanager = Base.RefValue{Union{Nothing,MPIManager}}(nothing)

function use_manager(manager::MPIManager)
    mpimanager[] = manager
end

struct DArrayWithMPIRanks{T,TA,TB}
    array::TA
    ranks::TB
end

function DArrayWithMPIRanks(D::DArray{T}) where T
    ranks = map(pid -> mpimanager[].j2mpi[pid], D.pids)

    return DArrayWithMPIRanks{T,typeof(D),typeof(ranks)}(D, ranks)
end

# Here we could add an underscore for fortran bindings if necessary
macro cosmafunc(x)
    return Expr(:quote, x)
end

struct Block
    data::Ptr{Nothing}
    ld::Cint
    row::Cint
    col::Cint
end

struct JuliaLayout
    rowblocks::Cint
    colblocks::Cint
    rowsplit::Vector{Cint}
    colsplit::Vector{Cint}
    owners::Vector{Cint}
    nlocalblocks::Cint
    localblocks::Vector{Block}
end

function JuliaLayout(A::DArrayWithMPIRanks{T}, rank) where T
    rowblocks = Cint(size(A.ranks, 1))
    colblocks = Cint(size(A.ranks, 2))
    rowsplit = map(i -> Cint(i - 1), A.array.cuts[1])
    colsplit = map(i -> Cint(i - 1), A.array.cuts[2])
    owners = map(Cint, A.ranks)

    # Get the blocks we own
    indices = findall(i -> i == rank, owners)
    @assert length(indices) == 1 # should be enforced by DArray
    block_row, block_col = Cint.(first(indices).I .- 1)
    
    localblocks = Block(
        Base.unsafe_convert(Ptr{Cvoid}, pointer(localpart(A.array))),
        stride(localpart(A.array), 2),
        block_row,
        block_col
    )

    return JuliaLayout(rowblocks, colblocks, rowsplit, colsplit, vec(owners), 1, [localblocks])
end

struct Layout
    rowblocks::Cint
    colblocks::Cint
    rowsplit::Ptr{Cint}
    colsplit::Ptr{Cint}
    owners::Ptr{Cint}
    nlocalblocks::Cint
    blocks::Ptr{Block}
end

Layout(A::JuliaLayout) = Layout(
    A.rowblocks,
    A.colblocks,
    pointer(A.rowsplit),
    pointer(A.colsplit),
    pointer(A.owners),
    A.nlocalblocks,
    pointer(A.localblocks)
)

for (fname, elty) in ((:dmultiply_using_layout,:Float64),
                      (:smultiply_using_layout,:Float32),
                      (:zmultiply_using_layout,:ComplexF64),
                      (:cmultiply_using_layout,:ComplexF32))
    @eval begin
        function gemm!(C::DArrayWithMPIRanks{$elty}, transa::AbstractChar, transb::AbstractChar, A::DArrayWithMPIRanks{$elty}, B::DArrayWithMPIRanks{$elty}, alpha::$elty, beta::$elty)
            # Get our rank
            comm = MPI.COMM_WORLD
            rank = MPI.Comm_rank(comm)

            A_julia = JuliaLayout(A, rank)
            B_julia = JuliaLayout(B, rank)
            C_julia = JuliaLayout(C, rank)

            GC.@preserve A_julia B_julia C_julia A B C begin
                A_cosma = Layout(A_julia)
                B_cosma = Layout(B_julia)
                C_cosma = Layout(C_julia)

                ccall((@cosmafunc($fname), cosma), Cvoid, (
                    MPI.MPI_Comm #= comm =#,
                    Ref{Cchar} #= transa =#,
                    Ref{Cchar} #= transb =#,
                    Ref{$elty} #= alpha =#,
                    Ref{Layout} #= layout_a =#,
                    Ref{Layout} #= layout_b =#,
                    Ref{$elty} #= beta =#,
                    Ref{Layout} #= layout_c =#,
                ),
                    comm, transa, transb, alpha, A_cosma, B_cosma, beta, C_cosma
                )
            end
        end
    end
end

function gemm_wrapper!(C::DMatrix{T}, transa::AbstractChar, transb::AbstractChar, A::DMatrix{T}, B::DMatrix{T}, alpha::T, beta::T) where {T<:BlasFloat}
    Ad1, Ad2 = (transa == 'N') ? (1,2) : (2,1)
    Bd1, Bd2 = (transb == 'N') ? (1,2) : (2,1)
    mA, nA = (size(A, Ad1), size(A, Ad2))
    mB, nB = (size(B, Bd1), size(B, Bd2))
    if mB != nA
        throw(DimensionMismatch("matrix A has dimensions ($mA, $nA), matrix B has dimensions ($mB, $nB)"))
    end
    if size(C,1) != mA || size(C,2) != nB
        throw(DimensionMismatch("result C has dimensions $(size(C)), needs ($mA, $nB)"))
    end

    # Get unique workers
    pids = unique(vcat(vec(C.pids), vec(A.pids), vec(B.pids)))

    # Julia pids are not mpi ranks :(
    A_mpi = DArrayWithMPIRanks(A)
    B_mpi = DArrayWithMPIRanks(B)
    C_mpi = DArrayWithMPIRanks(C)

    @sync for p in pids
        @spawnat p gemm!(C_mpi, transa, transb, A_mpi, B_mpi, alpha, beta)
    end

    return C
end

function promote_alpha_beta(a, b, ::Type{T}) where {T}
    a_prom, b_prom = promote(a, b, zero(T))
    a_prom, b_prom
end

# Pirate the mul! implementation
mul!(C::DMatrix{T}, A::DMatrix{T}, B::DMatrix{T}, a::Number, b::Number) where T<:BlasFloat = 
    gemm_wrapper!(C, 'N', 'N', A, B, promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, trA::Transpose{T, <:DMatrix{T}}, B::DMatrix{T}, a::Number, b::Number) where T<:BlasFloat =
    gemm_wrapper!(C, 'T', 'N', parent(trA), B, promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, A::DMatrix{T}, trB::Transpose{T, <:DMatrix{T}}, a::Number, b::Number) where T<:BlasFloat =
    gemm_wrapper!(C, 'N', 'T', A, parent(trB), promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, trA::Transpose{T, <:DMatrix{T}}, trB::Transpose{T, <:DMatrix{T}}, a::Number, b::Number) where T<:BlasFloat =
    gemm_wrapper!(C, 'T', 'T', parent(trA), parent(trB), promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, adjA::Adjoint{T, <:DMatrix{T}}, B::DMatrix{T}, a::Real, b::Real) where T<:BlasReal =
    gemm_wrapper!(C, 'T', 'N', parent(adjA), B, promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, adjA::Adjoint{T, <:DMatrix{T}}, B::DMatrix{T}, a::Number, b::Number) where T<:BlasComplex =
    gemm_wrapper!(C, 'C', 'N', parent(adjA), B, promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, A::DMatrix{T}, adjB::Adjoint{T, <:DMatrix{T}}, a::Real, b::Real) where T<:BlasReal =
    gemm_wrapper!(C, 'N', 'T', A, parent(adjB), promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, A::DMatrix{T}, adjB::Adjoint{T, <:DMatrix{T}}, a::Number, b::Number) where T<:BlasComplex =
    gemm_wrapper!(C, 'N', 'C', A, parent(adjB), promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, adjA::Adjoint{T, <:DMatrix{T}}, adjB::Adjoint{T, <:DMatrix{T}}, a::Real, b::Real) where T<:BlasReal =
    gemm_wrapper!(C, 'T', 'T', parent(adjA), parent(adjB), promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, adjA::Adjoint{T, <:DMatrix{T}}, adjB::Adjoint{T, <:DMatrix{T}}, a::Number, b::Number) where T<:BlasComplex =
    gemm_wrapper!(C, 'C', 'C', parent(adjA), parent(adjB), promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, trA::Transpose{T, <:DMatrix{T}}, adjB::Adjoint{T, <:DMatrix{T}}, a::Real, b::Real) where T<:BlasReal =
    gemm_wrapper!(C, 'T', 'T', parent(trA), parent(adjB), promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, trA::Transpose{T, <:DMatrix{T}}, adjB::Adjoint{T, <:DMatrix{T}}, a::Number, b::Number) where T<:BlasComplex =
    gemm_wrapper!(C, 'T', 'C', parent(trA), parent(adjB), promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, adjA::Adjoint{T, <:DMatrix{T}}, trB::Transpose{T, <:DMatrix{T}}, a::Real, b::Real) where T<:BlasReal =
    gemm_wrapper!(C, 'T', 'T', parent(adjA), parent(trB), promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, adjA::Adjoint{T, <:DMatrix{T}}, trB::Transpose{T, <:DMatrix{T}}, a::Number, b::Number) where T <: BlasComplex =
    gemm_wrapper!(C, 'C', 'T', parent(adjA), parent(trB), promote_alpha_beta(a, b, T)...)
end
