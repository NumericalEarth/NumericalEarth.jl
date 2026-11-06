include("runtests_setup.jl")

using Terrarium
using RingGrids: OctahedralGaussianGrid
using NCDatasets: NCDataset, defDim, defVar
using Oceananigans.Fields: interior

import NumericalEarth.DataWrangling.SoilGrids as SoilGridsModule

const NumericalEarthTerrariumExt = Base.get_extension(NumericalEarth, :NumericalEarthTerrariumExt)

# Raw SoilGrids2 values (g/kg for texture, cg/cm³ for bulk density) in file depth `d` (top down).
# Texture sums to less than 1000 g/kg so that `normalize_texture!` has something to do.
synthetic_soilgrids_value(name, d) = name == "sand" ? 500 :
                                     name == "silt" ? 300 :
                                     name == "clay" ? 50 + 10d :
                                                      130 # bdod

# A stand-in for the pre-baked Clenshaw file with only the `Mean` statistic and a missing first
# longitude column, as in the real data
function write_synthetic_soilgrids(path)
    Nλ, Nφ, Nz = size(SoilGrids2(), :sand_fraction)
    NCDataset(path, "c") do ds
        defDim(ds, "lon", Nλ)
        defDim(ds, "lat", Nφ)
        defDim(ds, "depth", Nz)
        defDim(ds, "statistic", 1)
        for name in ("sand", "silt", "clay", "bdod")
            data = Array{Int16}(undef, Nλ, Nφ, Nz, 1)
            for d in 1:Nz
                data[:, :, d, 1] .= synthetic_soilgrids_value(name, d)
            end
            data[1, :, :, 1] .= -32768
            variable = defVar(ds, name, Int16, ("lon", "lat", "depth", "statistic");
                              deflatelevel = 1, chunksizes = [Nλ, 64, 1, 1])
            variable[:, :, :, :] = data
        end
    end
    return path
end

source_values(sources, path) =
    Array(interior(only(s for s in sources.sources if Terrarium.varpath(s) == path).field))

@testset "SoilGrids2 fill_missing_first_longitude!" begin
    # (lon, lat, horizon): the first longitude becomes the mean of its two neighbors
    data = reshape(Float32[NaN NaN; 2 3; 4 5], 3, 2, 1)
    NumericalEarthTerrariumExt.fill_missing_first_longitude!(data)
    @test data[1, :, 1] == Float32[3, 4]

    # or the one neighbor that has data
    data = reshape(Float32[NaN NaN; 2 NaN; NaN 7], 3, 2, 1)
    NumericalEarthTerrariumExt.fill_missing_first_longitude!(data)
    @test data[1, :, 1] == Float32[2, 7]
end

@testset "Terrarium InputSources from SoilGrids2" begin
    # CPU only: the Terrarium column grid is not yet GPU-backed.
    gpu_test && return

    default_cache = SoilGridsModule.download_SoilGrids2_cache
    mktempdir() do cache
        setglobal!(SoilGridsModule, :download_SoilGrids2_cache, cache)
        try
            dataset = SoilGrids2()
            write_synthetic_soilgrids(metadata_path(Metadatum(:sand_fraction; dataset)))

            # Octahedral rings all start at longitude 0, the missing column of the data
            rings = OctahedralGaussianGrid(8)
            grid = ColumnRingGrid(CPU(), Float32, ExponentialSpacing(N = 6), rings)
            sources = Terrarium.InputSources(dataset, grid; verbose = false)

            @test length(sources.sources) == 6 * 4

            for d in 1:6
                horizon = Symbol(:horizon, d)
                sand = source_values(sources, (horizon, :sand_fraction))
                silt = source_values(sources, (horizon, :silt_fraction))
                clay = source_values(sources, (horizon, :clay_fraction))
                bulk_density = source_values(sources, (horizon, :bulk_density))

                # Normalized texture with the horizons in top-down order
                total = 850 + 10d
                @test all(isfinite, clay)
                @test all(sand .≈ 500 / total)
                @test all(silt .≈ 300 / total)
                @test all(clay .≈ (50 + 10d) / total)
                @test all(sand .+ silt .+ clay .≈ 1)

                # cg/cm³ → kg/dm³
                @test all(bulk_density .≈ 1.3)
            end
        finally
            setglobal!(SoilGridsModule, :download_SoilGrids2_cache, default_cache)
        end
    end
end
