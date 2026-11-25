include("runtests_setup.jl")

using NCDatasets: NCDataset, defVar
using NumericalEarth.DataWrangling: dataset_variable_name, metadata_url, native_times
using NumericalEarth.DataWrangling.PCMDIAMIP: PCMDIAMIPRelease
using NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger
using Oceananigans.Grids: φnodes
using Oceananigans.OutputReaders: Clamp
using Oceananigans.Units: Time, days

# The first months of a PCMDI-AMIP file, in the units and layout of the real one
synthetic_file_value(name, φ, month) = name == :sea_surface_temperature ? 10 + φ / 10 + month : 100 * month * sind(φ)

function write_synthetic_pcmdi_amip_file(metadata)
    longitude = 0.5:359.5
    latitude = -89.5:89.5
    data = Float32[synthetic_file_value(metadata.name, φ, month) for λ in longitude, φ in latitude, month in 1:length(metadata)]

    NCDataset(metadata_path(metadata), "c") do ds
        defVar(ds, "lon", collect(longitude), ("lon",))
        defVar(ds, "lat", collect(latitude), ("lat",))
        defVar(ds, dataset_variable_name(metadata), data, ("lon", "lat", "time"))
    end
end

january_to_march = (DateTime(1870, 1, 1), DateTime(1870, 3, 1))
dir = mktempdir()

@testset "PCMDI-AMIP metadata" begin
    metadata = Metadata(:sea_surface_temperature; dataset=PCMDIAMIPBoundaryConditions(), dates=january_to_march, dir)

    @test size(metadata) == (360, 180, 1, 3)
    @test native_times(metadata) == [15.5, 45, 74.5] .* days
    @test metadata_url(metadata) == "https://esgf-node.ornl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP7/CMIP/PCMDI/" *
                                    "PCMDI-AMIP-1-1-10/ocean/mon/tosbcs/gn/v20250807/" *
                                    "tosbcs_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-10_gn_187001-202212.nc"

    release = PCMDIAMIPRelease("1-1-6", "CMIP6", "v20191121", DateTime(2018, 12, 1))
    observed_sea_ice = Metadata(:sea_ice_concentration; dataset=PCMDIAMIPObservations(; release), dir)
    @test last(observed_sea_ice.dates) == DateTime(2018, 12, 1)
    @test metadata_url(observed_sea_ice) == "https://esgf-node.ornl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6/CMIP/PCMDI/" *
                                            "PCMDI-AMIP-1-1-6/seaIce/mon/siconc/gn/v20191121/" *
                                            "siconc_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-6_gn_187001-201812.nc"
end

for arch in test_architectures, dataset in (PCMDIAMIPBoundaryConditions(), PCMDIAMIPObservations())
    @testset "$(typeof(dataset)) on $(typeof(arch))" begin
        sea_surface_temperature = Metadata(:sea_surface_temperature; dataset, dates=january_to_march, dir)
        sea_ice_concentration = Metadata(:sea_ice_concentration; dataset, dates=january_to_march, dir)
        write_synthetic_pcmdi_amip_file(sea_surface_temperature)
        write_synthetic_pcmdi_amip_file(sea_ice_concentration)

        sst = FieldTimeSeries(sea_surface_temperature, arch; time_indexing=Clamp())
        ice = FieldTimeSeries(sea_ice_concentration, arch; time_indexing=Clamp())
        φ = reshape(Array(φnodes(sst.grid, Center())), 1, :, 1)

        # Kelvin and area fraction, each month read from its own record, overshoots included
        for month in 1:3
            @test all(Array(interior(sst[month])) .≈ Float32.(283.15 .+ φ ./ 10 .+ month))
            @test all(isapprox.(Array(interior(ice[month])), month .* sind.(φ), atol=1e-6))
        end

        # Halfway between mid-January and mid-February
        halfway = Time((sst.times[1] + sst.times[2]) / 2)
        @test all(Array(interior(sst[halfway])) .≈ Float32.(283.15 .+ φ ./ 10 .+ 1.5))

        grid = LatitudeLongitudeGrid(arch; size=(36, 18), longitude=(0, 360), latitude=(-90, 90),
                                     topology=(Periodic, Bounded, Flat))
        sst = FieldTimeSeries(sea_surface_temperature, grid; time_indexing=Clamp())
        ocean = PrescribedOcean(grid, sst.times; sea_surface_temperature=sst)
        time_step!(ocean, halfway.time)
        T = ComponentExchanger(ocean, grid).state.T
        φ = reshape(Array(φnodes(grid, Center())), 1, :, 1)
        @test all(isapprox.(Array(interior(T)), 283.15 .+ φ ./ 10 .+ 1.5, rtol=1e-6))
    end
end
