using NumericalEarth
using Oceananigans
using CUDA
using Test

using Downloads: download
using NumericalEarth.DataWrangling
using NumericalEarth.DataWrangling: metadata_path
using NumericalEarth.EN4
using NumericalEarth.ECCO
using NumericalEarth.ETOPO
using NumericalEarth.JRA55
using NumericalEarth.WOA

using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.Fields: interpolate!, interior
using Oceananigans: set!
using Oceananigans.TimeSteppers: update_state!

using CFTime
using Dates

using CUDA: @allowscalar

gpu_test = parse(Bool, get(ENV, "GPU_TEST", "false"))
test_architectures = gpu_test ? [GPU()] : [CPU()]

start_date = DateTimeProlepticGregorian(1993, 1, 1)

test_datasets = (ECCO2Monthly(),
                 ECCO2Daily(),
                 ECCO4Monthly(),
                 ECCO2DarwinMonthly(),
                 ECCO4DarwinMonthly(),
                 EN4Monthly(),
                )

test_names = Dict(
    ECCO2Monthly() => (:temperature, :salinity),
    ECCO2Daily() => (:temperature, :salinity),
    ECCO4Monthly() => (:temperature, :salinity),
    ECCO4DarwinMonthly() => (:temperature, :salinity, :phosphate),
    ECCO2DarwinMonthly() => (:temperature, :salinity, :phosphate),
    EN4Monthly() => (:temperature, :salinity),
)

test_fields = Dict(
    ECCO2Monthly() => (:T, :S),
    ECCO2Daily() => (:T, :S),
    ECCO4Monthly() => (:T, :S),
    ECCO4DarwinMonthly() => (:T, :S, :PO₄),
    ECCO2DarwinMonthly() => (:T, :S, :PO₄),
    EN4Monthly() => (:T, :S),
)


#####
##### Test utilities
#####

scalar(field) = Array(interior(field))[1, 1, 1]

bare_soil_humidity(FT) = DryLayerHumidity(FT;
    dry_layer_depth = StorageBasedDryLayerDepth(FT; maximum_dry_layer_depth = 0.015,
                                                dry_layer_onset_saturation = 0.5, dry_layer_exponent = 2),
    vapor_exchange  = DryLayerVaporPistonVelocity(FT; minimum_dry_layer_depth = 1e-3,
                                                  molecular_diffusivity = 2.4e-5, tortuosity = ConstantTortuosity()),
    thermal_exchange_depth = 0.05, porosity = 0.4)

# Components of a coupled single-column land model: a 1-cell grid, a filled `PrescribedAtmosphere`,
# a `SlabLand` (`BucketHydrology` + `SlabEnergy` unless `hydrology` is given), and (unless
# `radiation = nothing`) a filled `PrescribedRadiation`.
function coupled_land_components(arch, FT = Float64;
                                 grid = LatitudeLongitudeGrid(arch, FT; size = 1, latitude = 10, longitude = 10,
                                                              z = (-1, 0), topology = (Flat, Flat, Bounded)),
                                 Tair = 300.0, qair = 0.008, wind = 3.0, pressure = 101325.0, rain = nothing,
                                 hydrology = BucketHydrology(FT; maximum_water_storage = 150.0),
                                 Tland = 298.0, water = 45.0,
                                 shortwave = 600.0, longwave = 350.0, α = 0.2, ϵ = 0.95,
                                 radiation = PrescribedRadiation(grid; ocean_surface = nothing, sea_ice_surface = nothing,
                                                                 land_surface = SurfaceRadiationProperties(α, ϵ)))
    atmosphere = PrescribedAtmosphere(grid; surface_layer_height = 10, boundary_layer_height = 512)
    fill!(parent(atmosphere.temperature), Tair)
    fill!(parent(atmosphere.specific_humidity), qair)
    fill!(parent(atmosphere.velocities.u), wind)
    fill!(parent(atmosphere.pressure), pressure)
    if !isnothing(rain)
        fill!(parent(atmosphere.precipitation_flux.rain), rain)
        update_state!(atmosphere)
    end

    land = SlabLand(grid; hydrology, energy = SlabEnergy(FT))
    set!(land; T = Tland)
    isnothing(water) || fill!(parent(land.water_storage), water)

    if !isnothing(radiation)
        fill!(parent(radiation.downwelling_shortwave), shortwave)
        fill!(parent(radiation.downwelling_longwave), longwave)
        update_state!(radiation)
    end

    return grid, atmosphere, land, radiation
end

const coupled_land_component_keys = (:grid, :Tair, :qair, :wind, :pressure, :rain, :hydrology,
                                     :Tland, :water, :shortwave, :longwave, :α, :ϵ, :radiation)

# The coupled model on those components; keywords not listed above (e.g.
# `atmosphere_land_interface_temperature`) forward to `AtmosphereLandModel`. `time` sets the
# clock before the initializing `update_state!`.
function coupled_land_model(arch, FT = Float64; time = nothing, kw...)
    nt = (; kw...)
    component_keys = filter(k -> k in coupled_land_component_keys, keys(nt))
    model_keys     = filter(k -> !(k in coupled_land_component_keys), keys(nt))
    grid, atmosphere, land, radiation = coupled_land_components(arch, FT; nt[component_keys]...)
    model = AtmosphereLandModel(atmosphere, land; radiation, nt[model_keys]...)
    isnothing(time) || (model.clock.time = time)
    update_state!(model.land)
    update_state!(model)
    return model
end

function test_setting_from_metadata(arch, dataset, start_date, inpainting;
                                    loc = (Center, Center, Center),
                                    varnames = (:temperature, :salinity),
                                   )
    grid = LatitudeLongitudeGrid(arch;
                                 size = (10, 10, 10),
                                 latitude = (-60, -40),
                                 longitude = (10, 15),
                                 z = (-200, 0))

    field = Field{loc...}(grid)

    @test begin
        for name in varnames
            set!(field, Metadatum(name; dataset, date=start_date); inpainting)
        end
        true
    end

    return nothing
end

function test_timestepping_with_dataset(arch, dataset, start_date, inpainting;
                                        varnames  = (:temperature, :salinity),
                                        fldnames  = (:T, :S),
                                       )
    grid  = LatitudeLongitudeGrid(arch;
                                  size = (10, 10, 10),
                                  latitude = (-60, -40),
                                  longitude = (10, 15),
                                  z = (-200, 0),
                                  halo = (6, 6, 6))

    field = CenterField(grid)

    @test begin
        for name in varnames
            set!(field, Metadatum(name; dataset, date=start_date); inpainting)
        end
        true
    end

    ocean = ocean_simulation(grid; tracers=fldnames, verbose=false)
    set!(ocean.model, T=20, S=35)

    @test begin
        time_step!(ocean)
        time_step!(ocean)
        true
    end

    return nothing
end

function test_ocean_metadata_utilities(arch, dataset, dates, inpainting;
                                       varnames = (:temperature, :salinity),
                                      )
    for name in varnames
        metadata = Metadata(name; dates, dataset)
        filepaths = [metadata_path(datum) for datum in metadata]
        download_dataset_with_fallback(filepaths; dataset_name="$(typeof(dataset)) $name") do
            download(metadata)
        end
        restoring = DatasetRestoring(metadata, arch; rate=1/1000, inpainting)

        for datum in metadata
            @test isfile(metadata_path(datum))
        end

        fts = restoring.field_time_series
        @test fts isa FieldTimeSeries
        @test fts.grid isa LatitudeLongitudeGrid
        @test topology(fts.grid) == (Periodic, Bounded, Bounded)

        Nx, Ny, Nz = size(interior(fts))
        Nt = length(fts.times)

        @test Nx == size(metadata)[1]
        @test Ny == size(metadata)[2]
        @test Nz == size(metadata)[3]
        @test Nt == size(metadata)[4]

        @test @allowscalar fts.times[1] == native_times(metadata)[1]
        @test @allowscalar fts.times[end] == native_times(metadata)[end]

        datum = first(metadata)
        ψ = Field(datum, arch, inpainting=NearestNeighborInpainting(2))
        datapath = NumericalEarth.DataWrangling.inpainted_metadata_path(datum)
        @test isfile(datapath)
    end

    return nothing
end

function test_dataset_restoring(arch, dataset, dates, inpainting;
                                varnames = (:temperature, :salinity),
                                fldnames = (:T, :S),
                               )
    grid = LatitudeLongitudeGrid(arch;
                                 size = (100, 100, 10),
                                 latitude = (-75, 75),
                                 longitude = (0, 360),
                                 z = (-200, 0),
                                 halo = (6, 6, 6))

    φ₁ = @allowscalar grid.φᵃᶜᵃ[1]
    φ₂ = @allowscalar grid.φᵃᶜᵃ[21]
    φ₃ = @allowscalar grid.φᵃᶜᵃ[80]
    φ₄ = @allowscalar grid.φᵃᶜᵃ[100]
    z₁ = @allowscalar grid.z.cᵃᵃᶜ[6]

    mask = LinearlyTaperedPolarMask(northern = (φ₃, φ₄),
                                    southern = (φ₁, φ₂),
                                    z = (z₁, 0))

    for name in varnames
        metadata = Metadata(name; dates, dataset)
        filepaths = [metadata_path(datum) for datum in metadata]
        download_dataset_with_fallback(filepaths; dataset_name="$(typeof(dataset)) $name") do
            download(metadata)
        end
        var_restoring = DatasetRestoring(metadata, arch; mask, inpainting, rate=1/1000)

        fill!(var_restoring.field_time_series[1], 1.0)
        fill!(var_restoring.field_time_series[2], 1.0)

        field = NamedTuple{fldnames}(ntuple(i->CenterField(grid), length(fldnames)))

        # A window-averaged product has no node at its first date: the first sits half a window later.
        clock = Clock(; time = @allowscalar first(var_restoring.field_time_series.times))

        @allowscalar begin
            @test var_restoring(1, 1,   10, grid, clock, field) ≈ var_restoring.rate
            @test var_restoring(1, 11,  10, grid, clock, field) ≈ var_restoring.rate / 2
            @test var_restoring(1, 21,  10, grid, clock, field) == 0
            @test var_restoring(1, 80,  10, grid, clock, field) == 0
            @test var_restoring(1, 90,  10, grid, clock, field) ≈ var_restoring.rate / 2
            @test var_restoring(1, 100, 10, grid, clock, field) ≈ var_restoring.rate
            @test var_restoring(1, 1,   5,  grid, clock, field) == 0
            @test var_restoring(1, 10,  5,  grid, clock, field) == 0
        end
    end

    return nothing
end

function test_timestepping_with_dataset_restoring(arch, dataset, dates, inpainting;
                                                  varnames = (:temperature, :salinity),
                                                  fldnames = (:T, :S),
                                                 )
    grid = LatitudeLongitudeGrid(arch;
                                 size = (10, 10, 10),
                                 latitude = (-60, -40),
                                 longitude = (10, 15),
                                 z = (-200, 0),
                                 halo = (6, 6, 6))

    # Force only the last tracer.
    # Forcing more than one variable leads to parameter space errors
    metadata = Metadata(varnames[end]; dates, dataset)
    filepaths = [metadata_path(datum) for datum in metadata]
    download_dataset_with_fallback(filepaths; dataset_name="$(typeof(dataset)) $(varnames[end])") do
        download(metadata)
    end
    restoring = DatasetRestoring(metadata, arch; inpainting, rate=1/1000)
    forcing = NamedTuple{tuple(fldnames[end])}(tuple(restoring))
    ocean = ocean_simulation(grid; tracers=fldnames, forcing, verbose=false)
    set!(ocean.model, T=20, S=35)

    @test begin
        time_step!(ocean)
        time_step!(ocean)
        true
    end

    return nothing
end

function test_cycling_dataset_restoring(arch, dataset, dates, inpainting;
                                        varnames = (:temperature, :salinity),
                                        fldnames = (:T, :S),
                                       )
    grid = LatitudeLongitudeGrid(arch;
                                 size = (10, 10, 10),
                                 latitude = (-60, -40),
                                 longitude = (10, 15),
                                 z = (-200, 0),
                                 halo = (7, 7, 7))

    time_indices_in_memory = 2
    start_date = dates[1]
    end_date = dates[end]

    metadata = Metadata(varnames[end]; dates, dataset)

    # Dynamically create name of forcing based on dataset field name
    # Dynamically create name of forcing based on dataset field name
    forcing = NamedTuple{
            (fldnames[end],)
        }(
            (DatasetRestoring(metadata, arch;  time_indices_in_memory, inpainting, rate=1/1000),)
            )

    times = native_times(forcing[1].field_time_series.backend.metadata)
    ocean = ocean_simulation(grid, tracers=fldnames, forcing=forcing)
    set!(ocean.model, T=20, S=35)

    # start a bit after time_index
    time_index = 3
    time_interval = dataset isa ECCO2Daily ? Units.hour : Units.day
    ocean.model.clock.time = times[time_index] + 2 * time_interval
    update_state!(ocean.model)

    @test time_indices(forcing[1].field_time_series) ==
        Tuple(range(time_index, length=time_indices_in_memory))

    @test forcing[1].field_time_series.backend.start == time_index

    # Compile
    time_step!(ocean)

    # Try stepping out of the dataset bounds
    # start a bit after last time_index
    ocean.model.clock.time = last(times) + 2 * time_interval

    update_state!(ocean.model)

    @test begin
        time_step!(ocean)
        true
    end

    # The backend has cycled to the end
    @test time_indices(forcing[1].field_time_series) ==
        mod1.(Tuple(range(length(times), length=time_indices_in_memory)), length(times))
end

function test_inpainting_algorithm(arch, dataset, start_date, inpainting;
                                   varnames = (:temperature, :salinity),
                                  )
    for name in varnames
        var_metadatum = Metadatum(name; dataset, date=start_date)

        grid = LatitudeLongitudeGrid(arch,
                                 size = (20, 20, 10),
                                 latitude = (-75, 75),
                                 longitude = (0, 360),
                                 z = (-4000, 0),
                                 halo = (6, 6, 6))

        fully_inpainted_field = CenterField(grid)
        partially_inpainted_field = CenterField(grid)

        set!(fully_inpainted_field, var_metadatum; inpainting = NearestNeighborInpainting(Inf))
        set!(partially_inpainted_field, var_metadatum; inpainting = NearestNeighborInpainting(1))

        fully_inpainted_interior = on_architecture(CPU(), interior(fully_inpainted_field))
        partially_inpainted_interior = on_architecture(CPU(), interior(partially_inpainted_field))

        @test all(fully_inpainted_interior .!= 0)
        @test any(partially_inpainted_interior .== 0)
    end
    return nothing
end

include("synthetic_datasets.jl")
