module NumericalEarthConservativeRegriddingExt

using ConservativeRegridding: ConservativeRegridding, Regridder, regrid!
using Oceananigans: instantiated_location
using Oceananigans.Architectures: CPU, architecture, on_architecture
using Oceananigans.Fields: Field, interior
using Oceananigans.Grids: AbstractGrid
using Oceananigans.ImmersedBoundaries: underlying_grid
using NumericalEarth.EarthSystemModels.InterfaceComputations: InterfaceComputations, ConservativeExchange

function InterfaceComputations.conservative_exchange(component_grid, exchange_grid::AbstractGrid)
    arch = architecture(exchange_grid)
    cpu_exchange_grid = underlying_grid(on_architecture(CPU(), exchange_grid))

    # The exchange grid's manifold for both grids avoids a radius mismatch between the components
    manifold = ConservativeRegridding.GOCore.best_manifold(cpu_exchange_grid)
    from_component = Regridder(manifold, cpu_exchange_grid, component_grid)
    to_component = Regridder(copy(transpose(from_component.intersections)),
                             from_component.src_areas, from_component.dst_areas,
                             from_component.src_temp, from_component.dst_temp)

    return ConservativeExchange(on_architecture(arch, from_component), on_architecture(arch, to_component))
end

cell_values(field::Field) = vec(interior(field))
cell_values(array) = vec(array)

# Both copies of a tripolar fold row are filled only for a `Field` on the underlying grid
regrid_target(field::Field) = Field(instantiated_location(field), underlying_grid(field.grid);
                                    data = field.data, indices = field.indices)
regrid_target(array) = vec(array)

function InterfaceComputations.regrid_surface_field!(destination, regridder::Regridder, source)
    # Stage `source` in the regridder's dense buffer so `mul!` dispatches to the GPU sparse kernel
    regridder.src_temp .= cell_values(source)
    regrid!(regrid_target(destination), regridder, regridder.src_temp)
    return destination
end

end # module NumericalEarthConservativeRegriddingExt
