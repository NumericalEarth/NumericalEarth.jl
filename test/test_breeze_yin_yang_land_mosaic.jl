include("runtests_setup.jl")

using Breeze
using Oceananigans.Fields: OneField
using Oceananigans.Utils: launch!
using NumericalEarth.EarthSystemModels: DegreesKelvin

# The Yin-Yang adapter is evaluated in the extension by the drivers that use it
const NumericalEarthBreezeExt = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)
Base.include(NumericalEarthBreezeExt, joinpath(pkgdir(NumericalEarth), "ext", "NumericalEarthBreezeExt",
                                               "breeze_yin_yang_moist_hydrostatic_atmosphere.jl"))

const optics = (ocean_albedo = 0.05, ocean_emissivity = 0.97, sea_ice_albedo = 0.7, sea_ice_emissivity = 1.0)

function radiative_mosaic(arch, FT; ocean_fraction, sea_ice_concentration, ocean_temperature, sea_ice_temperature, land)
    grid = LatitudeLongitudeGrid(arch, FT; size=(4, 2), longitude=(0, 360), latitude=(-60, 60),
                                 topology=(Periodic, Bounded, Flat))
    surface_field(value) = (field = Field{Center, Center, Nothing}(grid); set!(field, value); field)
    surface = (surface_temperature = surface_field(0), surface_emissivity = surface_field(0), surface_albedo = surface_field(0))
    sea = (ocean_fraction = ocean_fraction isa Number ? surface_field(ocean_fraction) : ocean_fraction,
           sea_ice_concentration = surface_field(sea_ice_concentration),
           ocean_temperature = surface_field(ocean_temperature),
           sea_ice_temperature = surface_field(sea_ice_temperature),
           ocean_units = DegreesKelvin(),
           sea_ice_units = DegreesKelvin())
    land = map(value -> value isa Function ? value(surface_field) : value, land)

    launch!(arch, grid, :xy, NumericalEarthBreezeExt._mosaic_radiative_surface!,
            surface, sea, land, map(FT, optics), surface_field(-300), grid, Clock(time = zero(FT)))

    return map(field -> Array(interior(field))[1, 1, 1], surface)
end

@testset "Radiative mosaic of a Yin-Yang Breeze atmosphere on $(typeof(arch)) with $FT" for arch in test_architectures, FT in (Float64, Float32)
    ℵˢⁱ, Tᵒᶜ, Tˢⁱ = 0.25, 290, 260
    ocean, sea_ice = 1 - ℵˢⁱ, ℵˢⁱ

    @testset "Without land the mosaic is that of the ocean and the sea ice" begin
        no_land = NumericalEarthBreezeExt.mosaic_land_surface(nothing, nothing)
        mosaic = radiative_mosaic(arch, FT; ocean_fraction=OneField(), sea_ice_concentration=ℵˢⁱ,
                                  ocean_temperature=Tᵒᶜ, sea_ice_temperature=Tˢⁱ, land=no_land)

        ϵ = ocean * optics.ocean_emissivity + sea_ice * optics.sea_ice_emissivity
        ϵT⁴ = ocean * optics.ocean_emissivity * Tᵒᶜ^4 + sea_ice * optics.sea_ice_emissivity * Tˢⁱ^4
        @test mosaic.surface_emissivity ≈ FT(ϵ)
        @test mosaic.surface_albedo ≈ FT(ocean * optics.ocean_albedo + sea_ice * optics.sea_ice_albedo)
        @test mosaic.surface_temperature ≈ FT((ϵT⁴ / ϵ)^(1/4))
    end

    @testset "Land enters with the weight 1 - ocean_fraction, as in the net fluxes" begin
        ℵᵒᶜ, Tˡᵃ, αˡᵃ, ϵˡᵃ = 0.4, 305, 0.2, 0.95

        # The land's albedo as a field and its emissivity as a number, as a `SurfaceRadiationProperties` holds them
        land = (temperature = surface_field -> surface_field(Tˡᵃ), albedo = surface_field -> surface_field(αˡᵃ), emissivity = FT(ϵˡᵃ))
        mosaic = radiative_mosaic(arch, FT; ocean_fraction=ℵᵒᶜ, sea_ice_concentration=ℵˢⁱ,
                                  ocean_temperature=Tᵒᶜ, sea_ice_temperature=Tˢⁱ, land)

        fractions = (1 - ℵᵒᶜ, ℵᵒᶜ * ocean, ℵᵒᶜ * sea_ice)
        emissivities = (ϵˡᵃ, optics.ocean_emissivity, optics.sea_ice_emissivity)
        albedos = (αˡᵃ, optics.ocean_albedo, optics.sea_ice_albedo)
        temperatures = (Tˡᵃ, Tᵒᶜ, Tˢⁱ)

        ϵ = sum(fractions .* emissivities)
        @test sum(fractions) ≈ 1
        @test mosaic.surface_emissivity ≈ FT(ϵ)
        @test mosaic.surface_albedo ≈ FT(sum(fractions .* albedos))
        @test mosaic.surface_temperature ≈ FT((sum(fractions .* emissivities .* temperatures .^ 4) / ϵ)^(1/4))

        # A cell that is all land sees the land alone
        all_land = radiative_mosaic(arch, FT; ocean_fraction=0, sea_ice_concentration=ℵˢⁱ,
                                    ocean_temperature=Tᵒᶜ, sea_ice_temperature=Tˢⁱ, land)
        @test all_land.surface_emissivity ≈ FT(ϵˡᵃ)
        @test all_land.surface_albedo ≈ FT(αˡᵃ)
        @test all_land.surface_temperature ≈ FT(Tˡᵃ)
    end

    @testset "The land albedo that stands in for snow" begin
        grid = LatitudeLongitudeGrid(arch, FT; size=(4, 1), longitude=(0, 360), latitude=(-60, 60),
                                     topology=(Periodic, Bounded, Flat))
        temperature = Field{Center, Center, Nothing}(grid)
        albedo = Field{Center, Center, Nothing}(grid)
        interior(temperature) .= on_architecture(arch, reshape(FT[300, 273.15, 268.15, 240], 4, 1, 1))

        NumericalEarthBreezeExt.snow_proxy_land_albedo!(albedo, temperature)
        @test Array(interior(albedo))[:, 1, 1] ≈ FT[0.2, 0.2, 0.425, 0.65]
    end
end
