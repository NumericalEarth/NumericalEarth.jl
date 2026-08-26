using NumericalEarth: CopernicusAlbedo
using NumericalEarth.DataWrangling: DataWrangling, Metadatum
using Oceananigans.Fields: interior

# `surface_albedo = CopernicusAlbedo()` in a RadiativeTransferModel: CGLS blue-sky albedo
# on the model grid at the dekad nearest the solar epoch; water/missing pixels (NaN in the
# land product) become open-water albedo.

albedo_epoch(solar_position::Breeze.ApparentSolarPosition) = solar_position.epoch
albedo_epoch(solar_position) = nothing

# The hook lands with NumericalEarth/Breeze.jl#857. TODO: unconditionalize after release.
if isdefined(Breeze.AtmosphereModels, :materialize_surface_property)
    @eval function Breeze.AtmosphereModels.materialize_surface_property(dataset::CopernicusAlbedo, grid, solar_position)
        epoch = albedo_epoch(solar_position)
        isnothing(epoch) && throw(ArgumentError(
            "Dating `surface_albedo = CopernicusAlbedo()` requires `ApparentSolarPosition(epoch = ...)`; " *
            "otherwise pass an albedo `Field` built from a `Metadatum` with an explicit date."))

        dates = DataWrangling.all_dates(dataset, :albedo)
        date = dates[max(1, searchsortedlast(dates, epoch))]

        α = Oceananigans.Field{Center, Center, Nothing}(grid)
        Oceananigans.set!(α, Metadatum(:albedo; dataset, date, region = BoundingBox(grid)))
        interior(α) .= ifelse.(isnan.(interior(α)), 0.06, interior(α))
        return α
    end
end
