using NumericalEarth.DataWrangling: MetadataSet
using NumericalEarth.SoilGrids: SoilGrids2

# The SoilGrids2 data file has no data in its first longitude column (lon = 0°), where every ring of the
# octahedral model grid starts; fill it from its two neighbors (lon = 0.09° and lon = 359.91°) so that these
# points do not get NaN texture, which `normalize_texture!` would otherwise turn into pure sand.
# TODO: This should be fixed in the input data
function fill_missing_first_longitude!(data::AbstractArray)
    west = data[end:end, :, :]
    east = data[2:2, :, :]
    data[1:1, :, :] .= ifelse.(isnan.(west), east, ifelse.(isnan.(east), west, (west .+ east) ./ 2))
    return data
end

# SoilGrids2 input source
function Terrarium.InputSources(
        dataset::SoilGrids2,
        grid::Terrarium.ColumnRingGrid,
        horizons = Tuple(Symbol(:horizon, i) for i in 1:6);
        name = nameof(typeof(dataset)),
        verbose = true
    )
    soilgrids_vars = (:sand_fraction, :silt_fraction, :clay_fraction, :bulk_density)
    metadataset = MetadataSet(soilgrids_vars...; dataset)
    arch = RingGrids.architecture(grid.rings)
    nhorizons = length(horizons)
    layer_inputs = [Dict() for _ in 1:nhorizons]
    for var in soilgrids_vars
        verbose && @info "Loading input data for $var"
        # load each variable once with all of its horizons; every load reads the whole dataset variable
        var_field = Field(getproperty(metadataset, var))
        # (lon, lat, horizon) with latitudes north to south without the poles and horizons from the top down
        data = Array(interior(var_field)[:, (end - 1):-1:2, end:-1:(end - nhorizons + 1)])
        fill_missing_first_longitude!(data)
        ring_field = RingGrids.on_architecture(arch, RingGrids.FullClenshawField(data, input_as = Matrix))
        # interpolate all horizons onto the model grid in a single batched call
        target_field = RingGrids.interpolate(grid.rings, ring_field)
        for (k, horizon) in enumerate(horizons)
            horizon_field = RingGrids.Field(target_field.data[:, k], grid.rings)
            layer_inputs[k][var] = Terrarium.InputSource(grid, Field(horizon_field, grid); name = horizon => var)
        end
    end
    soilgrids_inputs = []
    for inputs in layer_inputs
        # Ensure that mineral texture components with each horizon sum to unity
        # TODO: This should be fixed in the input data
        Terrarium.normalize_texture!(inputs[:sand_fraction].field, inputs[:silt_fraction].field, inputs[:clay_fraction].field)
        append!(soilgrids_inputs, values(inputs))
    end
    return Terrarium.InputSources(name, soilgrids_inputs...)
end
