include("runtests_setup.jl")

using NumericalEarth.Grids: yin_yang_bilinear_weights, yin_yang_area_average_weights, panel_centers
using Oceananigans.MultiRegion: YinYangGrid
using Test

# A smooth field on the sphere: a polynomial in the Cartesian coordinates
smooth(λ, φ) = (x = cosd(φ) * cosd(λ); y = cosd(φ) * sind(λ); z = sind(φ); 1 + x + 2y + 3z + x * y - z^2)

yin_yang(Nλ) = YinYangGrid(CPU(); size = (Nλ, Nλ ÷ 3, 1), z = (0, 1), halo = (3, 3, 3))
panel_values(grid) = vcat((vec(smooth.(panel_centers(grid, r)...)) for r in 1:2)...)

# 1° latitude-longitude targets, poles included
λ = vec([λ for λ in -179.5:179.5, φ in -89.5:89.5])
φ = vec([φ for λ in -179.5:179.5, φ in -89.5:89.5])

@testset "Yin-Yang bilinear interpolation" begin
    errors = map((48, 96)) do Nλ
        grid = yin_yang(Nλ)
        indices, weights = yin_yang_bilinear_weights(grid, λ, φ)
        @test all(w -> w ≥ -1e-9, weights)
        @test all(s -> s ≈ 1, sum(weights; dims = 1))
        maximum(abs, vec(sum(weights .* panel_values(grid)[indices]; dims = 1)) .- smooth.(λ, φ))
    end
    @test errors[1] / errors[2] > 3   # second order
end

@testset "Yin-Yang area-weighted averaging" begin
    grid = yin_yang(48)
    area = cosd.(φ)
    averages(wet) = yin_yang_area_average_weights(grid, λ, φ, area, wet)

    for (r, average) in enumerate(averages(trues(length(λ))))
        A = zeros(length(panel_centers(grid, r)[1]))
        sums = zeros(length(A))
        for (n, m, w) in zip(average.rows, average.columns, average.weights)
            A[n] += w * smooth(λ[m], φ[m])
            sums[n] += w
        end
        covered = unique(average.rows)
        @test all(s -> s ≈ 1, sums[covered])
        @test maximum(abs, A[covered] .- vec(smooth.(panel_centers(grid, r)...))[covered]) < 0.05
        @test all(≈(1), average.wet_fraction[covered])
    end

    northern = φ .> 0
    for (r, average) in enumerate(averages(northern))
        φp = panel_centers(grid, r)[2]
        @test all(average.wet_fraction[φp .> 5] .≈ 1)
        @test all(average.wet_fraction[φp .< -5] .== 0)
    end
end
