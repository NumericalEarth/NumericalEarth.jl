# The slab's water budget over a 20-day run, accumulated every step from the model's own
# fluxes and runoff diagnostics (domain-mean mm per period), for the calibration CALIBRATION.
#
#   REFINEMENT=1 ARCH=cpu NSTEPS=2880 END_DATE=2020-04-22 DEEP_FLUX=darcy \
#   CALIBRATION=map_logK_r1_gpu_darcy_12d CALIBRATION_STEPS=1764 julia --project=docs water_budget.jl

include(joinpath(@__DIR__, "map_setup.jl"))

calibration = jldopen(f -> Dict(k => f[k] for k in keys(f)), get(ENV, "CALIBRATION", "map_logK_r1_gpu_darcy_12d") * ".jld2")
fields = map_fields(cpu_grid, fill(h₀, Nx, Ny))
s = surface_parameters(static, cpu_grid, FT)
model = borneo_coupled_model(cpu_grid, FT, forcing, s; slab_depth = surface_field(cpu_grid), exchanger_correction = correction,
                             surface_layer_height, boundary_layer_height, inner_iterations, similarity_iterations, hydrology_options(cpu_grid)...)
with_calibration(calibration)(model)
initialize_map!(model, fields.h, fields.θ₀, fields.T₀, fields.q₀, fields.θᵈ₀)
land_model = model.land

names = (:rain, :throughfall, :canopy_evaporation, :infiltration, :vapor_flux, :surface_runoff, :subsurface_runoff, :pond_runoff,
         :deep_liquid_flux, :saturation_excess_pond, :canopy_store, :pond_store, :slab_store)
dm(f) = mean(interior(f, :, :, 1)[land])
periods = ("days 0–6.25" => 1:900, "days 6.25–9.5" => 901:1368, "days 9.5–12.25" => 1369:1764, "days 12.25–16.5" => 1765:2376, "days 16.5–20" => 2377:2880)
totals = Dict(p => Dict(n => 0.0 for n in names) for (p, _) in periods)
stores = Dict(p => zeros(3) for (p, _) in periods)
for (p, r) in periods
    stores[p][1] = dm(land_model.prognostic.canopy_water_storage); stores[p][2] = dm(land_model.prognostic.surface_water_storage); stores[p][3] = dm(land_model.water_storage)
    for n in r
        time_step!(model, Δt)
        t = totals[p]
        t[:rain]               += Δt * dm(model.interfaces.exchanger.atmosphere.state.Jʳⁿ)
        t[:throughfall]        += Δt * dm(land_model.diagnostics.throughfall)
        t[:canopy_evaporation] += Δt * dm(land_model.diagnostics.wet_canopy_evaporation)
        t[:infiltration]       += -Δt * dm(land_model.diagnostics.surface_liquid_flux)
        t[:vapor_flux]         += Δt * dm(land_model.fluxes.vapor_flux)
        t[:surface_runoff]     += Δt * dm(land_model.diagnostics.surface_runoff)
        t[:subsurface_runoff]  += Δt * dm(land_model.diagnostics.subsurface_runoff)
        t[:pond_runoff]        += Δt * dm(land_model.diagnostics.surface_water_runoff)
        t[:deep_liquid_flux]   += Δt * dm(land_model.diagnostics.deep_liquid_flux)
    end
    t = totals[p]
    t[:canopy_store] = dm(land_model.prognostic.canopy_water_storage) - stores[p][1]
    t[:pond_store]   = dm(land_model.prognostic.surface_water_storage) - stores[p][2]
    t[:slab_store]   = dm(land_model.water_storage) - stores[p][3]
end
@printf("%-16s %6s %8s %8s %8s %7s %7s %7s %7s %8s | %8s %8s %8s\n", "period", "rain", "thrufall", "canopyE", "infiltr", "vapor", "R sfc", "R sub", "R pond", "deep J", "ΔWᶜ", "ΔS pond", "ΔM slab")
for (p, _) in periods
    t = totals[p]
    @printf("%-16s %6.1f %8.1f %8.1f %8.1f %7.1f %7.1f %7.1f %7.1f %8.1f | %8.2f %8.2f %8.1f\n", p, t[:rain], t[:throughfall], t[:canopy_evaporation],
            t[:infiltration], t[:vapor_flux], t[:surface_runoff], t[:subsurface_runoff], t[:pond_runoff], t[:deep_liquid_flux], t[:canopy_store], t[:pond_store], t[:slab_store])
end
println("\nslab closure check per period: ΔM − (infiltration − vapor + deep J − R sub) should be ≈ 0")
for (p, _) in periods
    t = totals[p]
    @printf("%-16s %+8.2f mm\n", p, t[:slab_store] - (t[:infiltration] - t[:vapor_flux] + t[:deep_liquid_flux] - t[:subsurface_runoff]))
end
