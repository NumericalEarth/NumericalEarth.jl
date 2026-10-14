include("runtests_setup.jl")

using CDSAPI
using NCDatasets: NCDataset, defVar
using NumericalEarth.DataWrangling: BoundingBox, dataset_variable_name
using NumericalEarth.DataWrangling.ERA5: ERA5HourlySingleLevel, ERA5MonthlySingleLevel,
                                         ERA5YearlySingleLevel, ERA5HourlyLand, ERA5MonthlyLand

@testset "ERA5 bulk time-series loading" begin
    for arch in test_architectures,
        dataset in (ERA5HourlySingleLevel(), ERA5MonthlySingleLevel(), ERA5YearlySingleLevel(),
                    ERA5HourlyLand(), ERA5MonthlyLand())

        @testset "$(typeof(dataset)) on $arch" begin
            mktempdir() do dir
                monthly = dataset isa Union{ERA5MonthlySingleLevel, ERA5MonthlyLand}
                cadence = monthly ? Month(1) : Hour(1)
                first_date = monthly ? DateTime(2004, 11, 1) : DateTime(2004, 12, 31, 22)
                file_dates = collect(first_date:cadence:first_date + 6cadence)
                dates = file_dates[[1, 2, 3, 5, 6, 7]]
                region = BoundingBox(longitude=(10, 11), latitude=(40, 41))
                metadata = Metadata(:temperature; dataset, dates, region, dir)
                paths = metadata_path(metadata)
                spacing = dataset isa Union{ERA5HourlyLand, ERA5MonthlyLand} ? 0.1 : 0.25
                longitude = collect(9.5:spacing:11.5)
                latitude = reverse(collect(39.5:spacing:41.5))
                time_name = monthly ? "time" : "valid_time"

                for path in unique(paths)
                    requested_dates = dates[findall(==(path), paths)]
                    stored_dates = collect(first(requested_dates):cadence:last(requested_dates))
                    NCDataset(path, "c") do ds
                        defVar(ds, "longitude", longitude, ("longitude",))
                        defVar(ds, "latitude", latitude, ("latitude",))
                        defVar(ds, time_name, stored_dates, (time_name,))
                        values = [Float32(250 + φ + λ + findfirst(==(date), file_dates))
                                  for λ in longitude, φ in latitude, date in stored_dates]
                        defVar(ds, dataset_variable_name(metadata), values, ("longitude", "latitude", time_name))
                    end
                end

                for window in (length(dates), 3)
                    series = FieldTimeSeries(metadata, arch; time_indices_in_memory=window)
                    for n in (1, 6, 2, 4)
                        expected = Field(metadata[n], arch)
                        @test Array(interior(series[n])) == Array(interior(expected))
                    end
                end
            end
        end
    end
end
