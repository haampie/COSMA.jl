module COSMA

import LinearAlgebra: mul!
using LinearAlgebra: BlasFloat, BlasReal, BlasComplex, Adjoint, Transpose
using DistributedArrays
using Distributed
using MPI
using MPIClusterManagers: MPIManager

const DMatrix{T} = DArray{T,2}

const mpimanager = Base.RefValue{Union{Nothing,MPIManager}}(nothing)

# Currently a hard-coded path to the shared lib
const lib = "/home/harmen/Documents/cscs/COSMA/build/here/usr/local/lib/libcosma.so"

function use_manager(manager::MPIManager)
    mpimanager[] = manager
end

struct Layout{T}
    rowblocks::Cint
    colblocks::Cint
    rowsplit::Vector{Cint}
    colsplit::Vector{Cint}
    owners::Vector{Cint}
    nlocalblock::Cint
    localblock_data::Vector{Ptr{T}}
    localblock_ld::Vector{Cint}
    localblock_row::Vector{Cint}
    localblock_col::Vector{Cint}
end

"""
Convert a DArray data type to something COSMA accepts
"""
function Layout(A::DMatrix{T}, owners_, rank) where {T<:BlasFloat}
    rowblocks = Cint(size(A.pids, 1))
    colblocks = Cint(size(A.pids, 2))
    rowsplit = map(i -> Cint(i - 1), A.cuts[1])
    colsplit = map(i -> Cint(i - 1), A.cuts[2])
    owners = map(Cint, owners_)
    
    nlocalblock = Cint(1)
    localblock_data = [pointer(localpart(A))]
    localblock_ld = [stride(localpart(A), 2)]

    # Get the blocks we own
    indices = findall(i -> i == rank, owners)
    localblock_row = map(index -> Cint(index.I[1] - 1), indices)
    localblock_col = map(index -> Cint(index.I[2] - 1), indices)

    return Layout{T}(
        rowblocks,
        colblocks,
        rowsplit,
        colsplit,
        vec(owners),
        nlocalblock,
        localblock_data,
        localblock_ld,
        localblock_row,
        localblock_col
    )
end

distribution_ranks(D::DArray) = map(pid -> mpimanager[].j2mpi[pid], D.pids)

# Here we could add an underscore for fortran bindings if necessary
macro cosmafunc(x)
    return Expr(:quote, x)
end

for (fname, elty) in ((:dmultiply,:Float64),
                      (:smultiply,:Float32),
                      (:zmultiply,:ComplexF64),
                      (:cmultiply,:ComplexF32))
    @eval begin
        function gemm!(C::Layout, transa::AbstractChar, transb::AbstractChar, A::Layout, B::Layout, alpha::$elty, beta::$elty, comm)
            @time ccall((@cosmafunc($fname), lib), Cvoid, (
                MPI.MPI_Comm #= comm =#,
                Ref{Cchar} #= transa =#,
                Ref{Cchar} #= transb =#,
                Ref{$elty} #= alpha =#,
                Ref{Cint} #= rowblocks_a =#, Ref{Cint} #= colblocks_a =#, Ptr{Cint} #= rowsplit_a =#, Ptr{Cint} #= colsplit_a =#, Ptr{Cint} #= owners_a =#,
                Ref{Cint} #= nlocalblock_a =#, Ptr{Ptr{$elty}} #= localblock_data_a =#, Ptr{Cint} #= localblock_ld_a =#, Ptr{Cint} #= localblock_row_a =#, Ptr{Cint} #= localblock_col_a =#,
                Ref{Cint} #= rowblocks_b =#, Ref{Cint} #= colblocks_b =#, Ptr{Cint} #= rowsplit_b =#, Ptr{Cint} #= colsplit_b =#, Ptr{Cint} #= owners_b =#,
                Ref{Cint} #= nlocalblock_b =#, Ptr{Ptr{$elty}} #= localblock_data_b =#, Ptr{Cint} #= localblock_ld_b =#, Ptr{Cint} #= localblock_row_b =#, Ptr{Cint} #= localblock_col_b =#,
                Ref{$elty} #= beta =#,
                Ref{Cint} #= rowblocks_c =#, Ref{Cint} #= colblocks_c =#, Ptr{Cint} #= rowsplit_c =#, Ptr{Cint} #= colsplit_c =#, Ptr{Cint} #= owners_c =#,
                Ref{Cint} #= nlocalblock_c =#, Ptr{Ptr{$elty}} #= localblock_data_c =#, Ptr{Cint} #= localblock_ld_c =#, Ptr{Cint} #= localblock_row_c =#, Ptr{Cint} #= localblock_col_ =#
            ),
                comm,
                transa,
                transb,
                alpha,
                A.rowblocks, A.colblocks, A.rowsplit, A.colsplit, A.owners,
                A.nlocalblock, A.localblock_data, A.localblock_ld, A.localblock_row, A.localblock_col,
                B.rowblocks, B.colblocks, B.rowsplit, B.colsplit, B.owners,
                B.nlocalblock, B.localblock_data, B.localblock_ld, B.localblock_row, B.localblock_col,
                beta,
                C.rowblocks, C.colblocks, C.rowsplit, C.colsplit, C.owners,
                C.nlocalblock, C.localblock_data, C.localblock_ld, C.localblock_row, C.localblock_col
            )
        end
    end
end

function gemm_wrapper!(C::DMatrix{T}, transa::AbstractChar, transb::AbstractChar, A::DMatrix{T}, B::DMatrix{T}, alpha::T, beta::T) where {T<:BlasFloat}
    # Get unique workers
    pids = unique(vcat(vec(C.pids), vec(A.pids), vec(B.pids)))
        
    # Julia pids are not mpi ranks :(
    owners_A = distribution_ranks(A)
    owners_B = distribution_ranks(B)
    owners_C = distribution_ranks(C)

    @sync for p in pids
        @spawnat p begin
            # Get our rank
            comm = MPI.COMM_WORLD
            rank = MPI.Comm_rank(comm)

            # Transform DArray to COSMA-like input
            A_layout = Layout(A, owners_A, rank)
            B_layout = Layout(B, owners_B, rank)
            C_layout = Layout(C, owners_C, rank)

            gemm!(C_layout, transa, transb, A_layout, B_layout, alpha, beta, comm)
        end
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
mul!(C::DMatrix{T}, trA::Transpose{<:Any, <:DMatrix{T}}, B::DMatrix{T}, a::Number, b::Number) where T<:BlasFloat =
    gemm_wrapper!(C, 'T', 'N', parent(trA), B, promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, A::DMatrix{T}, trB::Transpose{<:Any, <:DMatrix{T}}, a::Number, b::Number) where T<:BlasFloat =
    gemm_wrapper!(C, 'N', 'T', A, parent(trB), promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, trA::Transpose{<:Any, <:DMatrix{T}}, trB::Transpose{<:Any, <:DMatrix{T}}, a::Number, b::Number) where T<:BlasFloat =
    gemm_wrapper!(C, 'T', 'T', parent(trA), parent(trB), promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, adjA::Adjoint{<:Any, <:DMatrix{T}}, B::DMatrix{T}, a::Real, b::Real) where T<:BlasReal =
    gemm_wrapper!(C, 'T', 'N', parent(adjA), B, promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, adjA::Adjoint{<:Any, <:DMatrix{T}}, B::DMatrix{T}, a::Number, b::Number) where T<:BlasComplex =
    gemm_wrapper!(C, 'C', 'N', parent(adjA), B, promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, A::DMatrix{T}, adjB::Adjoint{<:Any, <:DMatrix{T}}, a::Real, b::Real) where T<:BlasReal =
    gemm_wrapper!(C, 'N', 'T', A, parent(adjB), promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, A::DMatrix{T}, adjB::Adjoint{<:Any, <:DMatrix{T}}, a::Number, b::Number) where T<:BlasComplex =
    gemm_wrapper!(C, 'N', 'C', A, parent(adjB), promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, adjA::Adjoint{<:Any, <:DMatrix{T}}, adjB::Adjoint{<:Any, <:DMatrix{T}}, a::Real, b::Real) where T<:BlasReal =
    gemm_wrapper!(C, 'T', 'T', parent(adjA), parent(adjB), promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, adjA::Adjoint{<:Any, <:DMatrix{T}}, adjB::Adjoint{<:Any, <:DMatrix{T}}, a::Number, b::Number) where T<:BlasComplex =
    gemm_wrapper!(C, 'C', 'C', parent(adjA), parent(adjB), promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, trA::Transpose{<:Any, <:DMatrix{T}}, adjB::Adjoint{T, <:DMatrix{T}}, a::Real, b::Real) where T<:BlasReal =
    gemm_wrapper!(C, 'T', 'T', parent(trA), parent(adjB), promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, trA::Transpose{<:Any, <:DMatrix{T}}, adjB::Adjoint{<:Any, <:DMatrix{T}}, a::Number, b::Number) where T<:BlasComplex =
    gemm_wrapper!(C, 'T', 'C', parent(trA), parent(adjB), promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, adjA::Adjoint{T, <:DMatrix{T}}, trB::Transpose{<:Any, <:DMatrix{T}}, a::Real, b::Real) where T<:BlasReal =
    gemm_wrapper!(C, 'T', 'T', parent(adjA), parent(trB), promote_alpha_beta(a, b, T)...)
mul!(C::DMatrix{T}, adjA::Adjoint{<:Any, <:DMatrix{T}}, trB::Transpose{<:Any, <:DMatrix{T}}, a::Number, b::Number) where T <: BlasComplex =
    gemm_wrapper!(C, 'C', 'T', parent(adjA), parent(trB), promote_alpha_beta(a, b, T)...)
end
