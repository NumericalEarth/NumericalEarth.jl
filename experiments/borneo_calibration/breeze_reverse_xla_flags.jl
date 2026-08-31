# The Breeze reverse-pass compiles with `XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion`
# (Breeze CI's own setting, suggested on Breeze#947): the same cases as the baseline sweep, for
# comparison against 851 s / 67.6 GB (6×8 Float32), 904 s / 73.5 GB / 263 s-per-step (6×8 Float64)
# and the unfinished 3 h 20 min 8×8 Float64. The flag must be in the environment before Julia starts.
#
#   XLA_FLAGS='--xla_disable_hlo_passes=multi_output_fusion' ARCH=cpu julia --project=docs breeze_reverse_xla_flags.jl

include(joinpath(@__DIR__, "breeze_reverse_setup.jl"))

@info "XLA_FLAGS = $(get(ENV, "XLA_FLAGS", "(unset)"))"

measure("6×8 Float32 compile, no MOF";  Nx = 6, Nz = 8, FT = Float32, compile = true)
measure("6×8 Float64 compile, no MOF";  Nx = 6, Nz = 8, compile = true)
measure("8×8 Float32 compile, no MOF";  Nx = 8, Nz = 8, FT = Float32, compile = true)
measure("8×8 Float64 compile, no MOF";  Nx = 8, Nz = 8, compile = true)
