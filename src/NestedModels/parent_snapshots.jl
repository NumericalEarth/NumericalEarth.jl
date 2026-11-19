#####
##### ParentSnapshots: a two-level window of a live parent's state, for a child to interpolate in time
#####

using Oceananigans: instantiated_location
using Oceananigans.OutputReaders: FieldTimeSeries, Clamp

struct ParentSnapshots{S, W}
    sources :: S   # the live parent fields that are read
    windows :: W   # one two-level `FieldTimeSeries` per source
end

"""
$(TYPEDSIGNATURES)

Hold the parent fields `sources` at the previous and current exchange times, so a child between the two
interpolates the parent in time. Both levels start as the state at `time`. Pass `snapshots.windows` to
[`parent_boundary_conditions`](@ref) or [`parent_forcings`](@ref), and `snapshots` to `NestedModel` as its
exchanger; the fields read need not belong to the `Simulation` that is stepped (e.g. the atmosphere of an
`EarthSystemModel`).
"""
function ParentSnapshots(sources::NamedTuple, time)
    windows = map(sources) do source
        window = FieldTimeSeries(instantiated_location(source), source.grid, [time - 1, time]; time_indexing = Clamp())
        parent(window[1]) .= parent(source)
        parent(window[2]) .= parent(source)
        return window
    end
    return ParentSnapshots{typeof(sources), typeof(windows)}(sources, windows)
end

# A new exchange time moves the current level to the previous one; the same time only refreshes it
# (`update_state!` and checkpoint restores exchange at a time already held).
function exchange_state!(snapshots::ParentSnapshots, time)
    for (source, window) in zip(snapshots.sources, snapshots.windows)
        current_time = last(window.times)
        if time != current_time
            parent(window[1]) .= parent(window[2])
            window.times = range(current_time, time; length = 2)
        end
        parent(window[2]) .= parent(source)
    end
    return nothing
end
