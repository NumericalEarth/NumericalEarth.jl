using NumericalEarth.DataWrangling: MetadataSet
using NumericalEarth.SoilGrids: SoilGrids2

# SoilGrids2 input source
function Terrarium.InputSources(
        dataset::SoilGrids2,
        grid::Terrarium.ColumnRingGrid,
        horizons = (Symbol(:horizon, i) for i in 1:6);
        name = nameof(typeof(dataset)),
        verbose = true
    )
    soilgrids_vars = (:sand_fraction, :silt_fraction, :clay_fraction, :bulk_density)
    metadataset = MetadataSet(soilgrids_vars...; dataset)
    arch = RingGrids.architecture(grid.rings)
    soilgrids_inputs = []
    for (idx, horizon) in enumerate(horizons)
        layer_inputs = Dict()
        for var in soilgrids_vars
            verbose && @info "Loading input data for $var on $horizon"
            var_field = Field(getproperty(metadataset, var))
            ring_field = RingGrids.on_architecture(arch, RingGrids.FullClenshawField(interior(var_field)[:, (end - 1):-1:2, end - idx + 1], input_as = Matrix))
            target_field = RingGrids.Field(grid.rings)
            RingGrids.interpolate!(target_field, ring_field)
            layer_inputs[var] = Terrarium.InputSource(grid, Field(target_field, grid); name = horizon => var)
        end
        # Ensure that mineral texture components with each horizon sum to unity
        # TODO: This should be fixed in the input data
        Terrarium.normalize_texture!(layer_inputs[:sand_fraction].field, layer_inputs[:silt_fraction].field, layer_inputs[:clay_fraction].field)
        append!(soilgrids_inputs, values(layer_inputs))
    end
    return Terrarium.InputSources(name, soilgrids_inputs...)
end
