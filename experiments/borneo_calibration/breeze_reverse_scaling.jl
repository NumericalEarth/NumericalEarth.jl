# Where the Breeze reverse-pass compile explodes: for a doubly periodic x–z Breeze model,
# separate the Reactant pipeline (`@code_hlo`) from XLA's backend compile (`@compile`),
# across grid size, float type and microphysics.
#
#   ARCH=cpu julia --project=docs breeze_reverse_scaling.jl

include(joinpath(@__DIR__, "breeze_reverse_setup.jl"))

measure("6×8 Float64";                     Nx = 6, Nz = 8)
measure("6×8 Float64, no microphysics";    Nx = 6, Nz = 8, microphysics = nothing)
measure("6×8 Float32";                     Nx = 6, Nz = 8, FT = Float32)
measure("8×8 Float64 (the 81 GB case)";    Nx = 8, Nz = 8)
measure("16×8 Float64";                    Nx = 16, Nz = 8)
measure("6×8 Float32 compile";             Nx = 6, Nz = 8, FT = Float32, compile = true)
measure("6×8 Float64 compile";             Nx = 6, Nz = 8, compile = true)
