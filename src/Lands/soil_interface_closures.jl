soil_hydrology(hydrology) = hydrology
soil_hydrology(hydrology::InterceptingHydrology) = hydrology.soil

# The plant-stress endpoints are fixed by the soil's retention curve, so they are evaluated once.
(s::InterfaceComputations.SaturationAtSuction)(land::SlabLand) =
    saturation_at_suction(soil_hydrology(land.hydrology), s.suction_head, land.grid)

function saturation_at_suction(hydrology::VariablySaturatedHydrology, suction_head, grid)
    saturation = KernelFunctionOperation{Center, Center, Nothing}(effective_saturation, grid, hydrology.retention_curve,
                                                                  convert(eltype(grid), suction_head))
    return compute!(Field(saturation))
end

saturation_at_suction(hydrology, suction_head, grid) =
    throw(ArgumentError("PlantAvailableWaterStress requires soil water retention physics."))

function (d::InterfaceComputations.SoilVaporDiffusivity)(land::SlabLand)
    vapor = d.vapor_exchange
    return soil_vapor_diffusivity(soil_hydrology(land.hydrology), vapor.tortuosity, vapor, land.saturation, land.grid)
end

soil_vapor_diffusivity(hydrology, ::ConstantTortuosity, vapor, saturation, grid) = vapor.molecular_diffusivity

function soil_vapor_diffusivity(hydrology::VariablySaturatedHydrology, ::PowerLawTortuosity, vapor, saturation, grid)
    return KernelFunctionOperation{Center, Center, Nothing}(surface_vapor_diffusivity, grid,
        saturation, hydrology.porosity, hydrology.residual_liquid_fraction, vapor)
end

@inline function surface_vapor_diffusivity(i, j, k, grid, saturation, porosity, residual, vapor)
    FT = eltype(grid)
    ν = convert(FT, stateindex(porosity, i, j, 1))
    θʳ = convert(FT, stateindex(residual, i, j, 1))
    θˡ = θʳ + saturation[i, j, 1] * (ν - θʳ)
    return effective_vapor_diffusivity(vapor, ν, θˡ)
end
