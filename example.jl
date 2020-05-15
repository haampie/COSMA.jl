using MPI
using MPIClusterManagers
using Distributed
using DistributedArrays

# Currently a hard-coded path to the shared lib
const lib = "/home/harmen/Documents/cscs/COSMA/build/here/usr/local/lib/libcosma.so"

# And a hard-coded number of workers...
const manager = MPIManager(np = 4)
addprocs(manager)

# Make code available on all workers
@everywhere begin
    using DistributedArrays
    using MPI

    struct COSMA{T}
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

    function COSMA(A::DArray{T,2}, julia_to_mpi, rank) where T
        rowblocks = Cint(size(A.pids, 1))
        colblocks = Cint(size(A.pids, 2))
        rowsplit = map(i -> Cint(i - 1), A.cuts[1])
        colsplit = map(i -> Cint(i - 1), A.cuts[2])
        owners = map(i -> Cint(julia_to_mpi[i]), A.pids)
        
        nlocalblock = Cint(1)
        localblock_data = [pointer(localpart(A))]
        localblock_ld = [stride(localblock_data, 2)]

        # Get the blocks we own
        indices = findall(i -> i == rank, owners)
        localblock_row = map(index -> Cint(index.I[1] - 1), indices)
        localblock_col = map(index -> Cint(index.I[2] - 1), indices)

        return COSMA{T}(
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
end

function example()
    C = dzeros(100, 100)
    A = drand(100, 100)
    B = drand(100, 100)

    # get unique workers
    pids = unique(vcat(vec(C.pids), vec(A.pids), vec(B.pids)))

    # julia pids are not mpi ranks :(
    mapping = manager.j2mpi

    @sync for p in pids
        @spawnat p begin
            transa = 'N'
            transb = 'N'
            alpha = 1.0
            beta = 0.0

            # get our rank
            comm = MPI.COMM_WORLD
            rank = MPI.Comm_rank(comm)

            # transform darray cosma-like input
            A_cosma = COSMA(A, mapping, rank)
            B_cosma = COSMA(B, mapping, rank)
            C_cosma = COSMA(C, mapping, rank)

            ccall((:multiply, lib), Cvoid, (
                MPI.MPI_Comm, #comm,
                Ref{Cchar}, #transa,
                Ref{Cchar}, #transb,
                Ref{Cdouble}, #alpha,
                Ref{Cint}, #rowblocks_a,
                Ref{Cint}, #colblocks_a,
                Ptr{Cint}, #rowsplit_a,
                Ptr{Cint}, #colsplit_a,
                Ptr{Cint}, #owners_a,
                Ref{Cint}, #nlocalblock_a,
                Ptr{Ptr{Cdouble}}, #localblock_data_a,
                Ptr{Cint}, #localblock_ld_a,
                Ptr{Cint}, #localblock_row_a,
                Ptr{Cint}, #localblock_col_a,
                Ref{Cint}, #rowblocks_b,
                Ref{Cint}, #colblocks_b,
                Ptr{Cint}, #rowsplit_b,
                Ptr{Cint}, #colsplit_b,
                Ptr{Cint}, #owners_b,
                Ref{Cint}, #nlocalblock_b,
                Ptr{Ptr{Cdouble}}, #localblock_data_b,
                Ptr{Cint}, #localblock_ld_b,
                Ptr{Cint}, #localblock_row_b,
                Ptr{Cint}, #localblock_col_b,
                Ref{Cdouble}, #beta,
                Ref{Cint}, #rowblocks_c,
                Ref{Cint}, #colblocks_c,
                Ptr{Cint}, #rowsplit_c,
                Ptr{Cint}, #colsplit_c,
                Ptr{Cint}, #owners_c,
                Ref{Cint}, #nlocalblock_c,
                Ptr{Ptr{Cdouble}}, #localblock_data_c,
                Ptr{Cint}, #localblock_ld_c,
                Ptr{Cint}, #localblock_row_c,
                Ptr{Cint} #localblock_col_c
            ),
                comm, #comm,
                transa, #transa,
                transb, #transb,
                alpha, #alpha,
                A_cosma.rowblocks, #rowblocks_a,
                A_cosma.colblocks, #colblocks_a,
                A_cosma.rowsplit, #rowsplit_a,
                A_cosma.colsplit, #colsplit_a,
                A_cosma.owners, #owners_a,
                A_cosma.nlocalblock, #nlocalblock_a,
                A_cosma.localblock_data, #localblock_data_a,
                A_cosma.localblock_ld, #localblock_ld_a,
                A_cosma.localblock_row, #localblock_row_a,
                A_cosma.localblock_col, #localblock_col_a,
                B_cosma.rowblocks, #rowblocks_b,
                B_cosma.colblocks, #colblocks_b,
                B_cosma.rowsplit, #rowsplit_b,
                B_cosma.colsplit, #colsplit_b,
                B_cosma.owners, #owners_b,
                B_cosma.nlocalblock, #nlocalblock_b,
                B_cosma.localblock_data, #localblock_data_b,
                B_cosma.localblock_ld, #localblock_ld_b,
                B_cosma.localblock_row, #localblock_row_b,
                B_cosma.localblock_col, #localblock_col_b,
                beta, #beta,
                C_cosma.rowblocks, #rowblocks_c,
                C_cosma.colblocks, #colblocks_c,
                C_cosma.rowsplit, #rowsplit_c,
                C_cosma.colsplit, #colsplit_c,
                C_cosma.owners, #owners_c,
                C_cosma.nlocalblock, #nlocalblock_c,
                C_cosma.localblock_data, #localblock_data_c,
                C_cosma.localblock_ld, #localblock_ld_c,
                C_cosma.localblock_row, #localblock_row_c,
                C_cosma.localblock_col #localblock_col_c
            )
        end
    end

    A, B, C
end