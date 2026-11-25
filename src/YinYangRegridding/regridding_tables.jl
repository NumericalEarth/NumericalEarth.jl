#####
##### Sparse regridding tables in compressed-row form: destination cell `n = i + (j - 1) Nx` of a region
##### is Σₚ weight[p] * source[source_i[p], source_j[p], 1] over p = first_entry[n] : first_entry[n+1] - 1.
##### Tables are built once on the CPU and applied by a kernel, so they need no sparse-matrix package
##### and run on any architecture.
#####

struct RegriddingTable{P, I, W}
    first_entry :: P # first entry of each destination cell, of length (cells + 1)
    source_i :: I    # x-index of each entry's source cell
    source_j :: I    # y-index of each entry's source cell
    weight :: W      # weight of each entry
end

Adapt.adapt_structure(to, table::RegriddingTable) =
    RegriddingTable(adapt(to, table.first_entry), adapt(to, table.source_i),
                    adapt(to, table.source_j), adapt(to, table.weight))

move_to_architecture(arch, table::RegriddingTable) =
    RegriddingTable(on_architecture(arch, table.first_entry), on_architecture(arch, table.source_i),
                    on_architecture(arch, table.source_j), on_architecture(arch, table.weight))

Base.summary(table::RegriddingTable) =
    string("RegriddingTable with ", length(table.weight), " entries for ", length(table.first_entry) - 1, " cells")

Base.show(io::IO, table::RegriddingTable) = print(io, summary(table))

"""
$(SIGNATURES)

Return the `RegriddingTable` with entries `(destinations[p], sources[p], weights[p])`, where destinations
and sources are linear indices into regions with `destination_cells` cells and `source_Nx` cells in x.
Entries with zero weight are dropped.
"""
function RegriddingTable(FT, destination_cells, source_Nx, destinations, sources, weights)
    kept = findall(!iszero, weights)
    order = kept[sortperm(view(destinations, kept))]

    first_entry = ones(Int, destination_cells + 1)
    for p in order
        first_entry[destinations[p] + 1] += 1
    end
    for n in 1:destination_cells
        first_entry[n + 1] += first_entry[n] - 1
    end

    source_i = Int32[mod1(sources[p], source_Nx) for p in order]
    source_j = Int32[(sources[p] - 1) ÷ source_Nx + 1 for p in order]

    return RegriddingTable(first_entry, source_i, source_j, FT[weights[p] for p in order])
end

@kernel function _apply_regridding_table!(destination, source, table, Nx, accumulate)
    i, j = @index(Global, NTuple)
    n = i + (j - 1) * Nx

    @inbounds begin
        value = zero(eltype(destination))
        for p in table.first_entry[n]:table.first_entry[n + 1] - 1
            value += table.weight[p] * source[table.source_i[p], table.source_j[p], 1]
        end
        destination[i, j, 1] = ifelse(accumulate, destination[i, j, 1] + value, value)
    end
end

# Apply `table` to the lowest level of `source`, a region's data, writing or accumulating into the lowest
# level of `destination` on `destination_grid`.
function apply_regridding_table!(destination, destination_grid, source, table; accumulate=false)
    arch = architecture(destination_grid)
    Nx = size(destination_grid, 1)
    launch!(arch, destination_grid, :xy, _apply_regridding_table!, destination, source, table, Nx, accumulate)
    return nothing
end
