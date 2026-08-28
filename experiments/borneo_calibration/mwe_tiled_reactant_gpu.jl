# Minimal reproduction: a TiledLandInterface whose `fraction` is a Field fails to build on a
# ReactantState grid with the GPU backend (CUDA codegen of `_blend_tiled_land_fluxes!` in
# `reconcile_state!`), while a Number fraction builds.

using NumericalEarth
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant, CUDA

Reactant.set_default_backend("gpu")

grid = LatitudeLongitudeGrid(ReactantState(); size = (4, 4), latitude = (0, 1), longitude = (0, 1),
                             topology = (Bounded, Bounded, Flat))
atmosphere = PrescribedAtmosphere(grid, 0:3600:7200)
land = SlabLand(grid)

soil = DryLayerHumidity(Float64; porosity = 0.4,
                        dry_layer_depth = StorageBasedDryLayerDepth(Float64; maximum_dry_layer_depth = 0.05, dry_layer_onset_saturation = 1.0),
                        vapor_exchange = DryLayerVaporPistonVelocity(Float64; minimum_dry_layer_depth = 1e-3, molecular_diffusivity = 2.4e-5),
                        thermal_exchange_depth = 0.05)
vegetated = CanopyAirSpace(Float64; soil, inner_iterations = 4)

for (label, fraction) in (("Number fraction", 0.7),
                          ("Field fraction", (f = Field{Center, Center, Nothing}(grid); parent(f) .= 0.7; f)))
    try
        interface = TiledLandInterface(grid, atmosphere, land; vegetated, fraction)
        AtmosphereLandModel(atmosphere, land; atmosphere_land_interface = interface)
        @info "$label: model built"
    catch err
        @error "$label: failed" exception = (err, catch_backtrace())
    end
end
