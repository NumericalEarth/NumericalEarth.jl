using NumericalEarth
using Statistics
using ClimaSeaIce
using Oceananigans
using Oceananigans.Grids
using Oceananigans.Units
using NumericalEarth.Oceans
using NumericalEarth.ECCO
using NumericalEarth.JRA55
using NumericalEarth.WOA
using Printf
using Dates
using CUDA
using JLD2
using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity
using Oceananigans.Coriolis: EENConserving
using Oceananigans.Models.VarianceDissipationComputations

function omip_simulation(grid; forcing_dir, restoring_dir, filename)

    tracer_advection   = WENO(order=7; minimum_buffer_upwind_order=3)
    momentum_advection = WENOVectorInvariant(order=5)
    free_surface       = SplitExplicitFreeSurface(grid; substeps=150) 

    @inline νhb(i, j, k, grid, ℓx, ℓy, ℓz, clock, fields, λ) = Oceananigans.Operators.Az(i, j, k, grid, ℓx, ℓy, ℓz)^2 / λ

    horizontal_viscosity = HorizontalScalarBiharmonicDiffusivity(ν=νhb, discrete_form=true, parameters=40days) 
    catke_closure = NumericalEarth.Oceans.default_ocean_closure() 
    eddy_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=250) 
    closure = (catke_closure, eddy_closure, horizontal_viscosity)
    coriolis = HydrostaticSphericalCoriolis(scheme = EnstrophyConserving())

    # WOA monthly salinity restoring with piston velocity 1/6 m/day
    # following the OMIP protocol (Griffies et al., 2009; Danabasoglu et al., 2014)
    woa_dataset = WOAMonthly()
    Smetadata = Metadata(:salinity; dir=restoring_dir, dataset=woa_dataset)

    piston_velocity = 1/6 # m/day
    restoring_rate = piston_velocity / (Δzˢ * days)
    @inline surface_mask(x, y, z, t) = z ≥ zˢ
    FS = DatasetRestoring(Smetadata, grid; rate=restoring_rate, mask=surface_mask, time_indices_in_memory=12)

    ocean = ocean_simulation(grid; Δt=1minutes,
                            momentum_advection,
                            tracer_advection,
                            coriolis,
                            timestepper = :SplitRungeKutta3,
                            free_surface,
                            radiative_forcing = nothing,
                            forcing = (; S = FS),
                            closure)

    set!(ocean.model, T=Metadatum(:temperature; dir=restoring_dir, dataset=WOAAnnual()),
                      S=Metadatum(:salinity;    dir=restoring_dir, dataset=WOAAnnual()))

    #####
    ##### A Prognostic Sea-ice model
    #####

    # Default sea-ice dynamics and salinity coupling are included in the defaults
    sea_ice = sea_ice_simulation(grid, ocean; advection=WENO(order=7, minimum_buffer_upwind_order=1)) 

    set!(sea_ice.model, h=Metadatum(:sea_ice_thickness;     dir=restoring_dir, dataset=ECCO4Monthly()),
                        ℵ=Metadatum(:sea_ice_concentration; dir=restoring_dir, dataset=ECCO4Monthly()))

    #####
    ##### A Prescribed Atmosphere model
    #####

    dir = forcing_dir
    date = DateTime(1958, 1, 1)
    dataset = MultiYearJRA55()
    backend = JRA55NetCDFBackend(30)

    atmosphere = JRA55PrescribedAtmosphere(arch; dir, dataset, backend, include_rivers_and_icebergs=true, start_date=date)
    radiation  = Radiation()

    #####
    ##### An ocean-sea ice coupled model
    #####

    omip = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
    omip = Simulation(omip, Δt=20minutes, stop_time=100days) 

    omip.output_writers[:checkpointer] = Checkpointer(omip.model;
                                         schedule = TimeInterval(90.75days),
                                         prefix = filename * "_checkpoint",
                                         cleanup = false,
                                         verbose = true)

    wall_time = Ref(time_ns())

    uo, vo, wo = ocean.model.velocities
    To, So, eo = ocean.model.tracers
    ηo = ocean.model.free_surface.displacement
    bo = Oceananigans.Models.buoyancy_operation(ocean.model)

    ηo² = ηo^2
    uo² = uo^2
    vo² = vo^2
    To² = To^2
    So² = So^2
    bo² = bo^2

    uT = uo*To
    vT = vo*To
    uS = uo*So
    vS = vo*So
    wT = wo*To
    wS = wo*So

    mld = NumericalEarth.Diagnostics.MixedLayerDepthField(ocean.model.buoyancy, grid, ocean.model.tracers)

    κu = ocean.model.closure_fields[1].κu
    κc = ocean.model.closure_fields[1].κc

    Uⁿ⁻¹ = Oceananigans.Fields.VelocityFields(grid)
    Uⁿ   = Oceananigans.Fields.VelocityFields(grid)
    ϵT = VarianceDissipation(:T, grid; Uⁿ⁻¹, Uⁿ)
    ϵS = VarianceDissipation(:S, grid; Uⁿ⁻¹, Uⁿ)

    add_callback!(ocean, ϵT, IterationInterval(1))
    add_callback!(ocean, ϵS, IterationInterval(1))

    fT = VarianceDissipationComputations.flatten_dissipation_fields(ϵT)
    fS = VarianceDissipationComputations.flatten_dissipation_fields(ϵS)

    GTx = ∂x(To)^2 
    GTy = ∂y(To)^2 
    GTz = ∂z(To)^2 

    GSx = ∂x(So)^2 
    GSy = ∂y(So)^2 
    GSz = ∂z(So)^2 

    ocean_outputs = merge((; GTx, GTy, GTz, GSx, GSy, GSz, uo, vo, wo, To, So, ηo, bo, ηo², uo², vo², To², So², bo², uT, vT, uS, vS, wT, wS, mld, κu, κc), fT, fS)

    omip.output_writers[:ocean_averages] = JLD2Writer(ocean.model, ocean_outputs;
                                                    schedule = AveragedTimeInterval(30.25days),
                                                    filename = filename * "_ocean_averages",
                                                    including = [:grid])
                        
    ui, vi = sea_ice.model.velocities
    hi = sea_ice.model.ice_thickness
    ℵi = sea_ice.model.ice_concentration
    Ti = sea_ice.model.ice_thermodynamics.top_surface_temperature

    ice_outputs = (; ui, vi, ui² = ui^2, vi² = vi^2, hi, ℵi, Ti)

    omip.output_writers[:averages] = JLD2Writer(sea_ice.model, ice_outputs;
                                                schedule = AveragedTimeInterval(30.25days),
                                                filename = filename * "_sea_ice_averages",
                                                including = [:grid])

    τx = omip.model.interfaces.net_fluxes.ocean.u
    τy = omip.model.interfaces.net_fluxes.ocean.v
    JT = omip.model.interfaces.net_fluxes.ocean.T
    Js = omip.model.interfaces.net_fluxes.ocean.S
    Qc = omip.model.interfaces.atmosphere_ocean_interface.fluxes.sensible_heat
    Qv = omip.model.interfaces.atmosphere_ocean_interface.fluxes.latent_heat
    Qi = omip.model.interfaces.sea_ice_ocean_interface.fluxes.interface_heat
    Ji = omip.model.interfaces.sea_ice_ocean_interface.fluxes.salt

    fluxes = (; τx, τy, JT, Js, Qv, Qc, Qi, Ji)

    omip.output_writers[:fluxes] = JLD2Writer(ocean.model, fluxes; 
                                            filename = filename * "_fluxes_averages",
                                            including = [:grid],
                                            schedule = AveragedTimeInterval(30.25days))

    function progress(sim)
        sea_ice = sim.model.sea_ice
        ocean   = sim.model.ocean
        hmax = maximum(sea_ice.model.ice_thickness)
        ℵmax = maximum(sea_ice.model.ice_concentration)
        Tmax = maximum(sim.model.interfaces.atmosphere_sea_ice_interface.temperature)
        Tmin = minimum(sim.model.interfaces.atmosphere_sea_ice_interface.temperature)
        umax = maximum(ocean.model.velocities.u)
        vmax = maximum(ocean.model.velocities.v)
        wmax = maximum(ocean.model.velocities.w)

        step_time = 1e-9 * (time_ns() - wall_time[])

        msg1 = @sprintf("time: %s, iteration: %d, Δt: %s, ", prettytime(sim), iteration(sim), prettytime(sim.Δt))
        msg2 = @sprintf("max(h): %.2e m, max(ℵ): %.2e ", hmax, ℵmax)
        msg4 = @sprintf("extrema(T): (%.2f, %.2f) ᵒC, ", Tmax, Tmin)
        msg5 = @sprintf("maximum(u): (%.2f, %.2f, %.2f) m/s, ", umax, vmax, wmax)
        msg6 = @sprintf("wall time: %s \n", prettytime(step_time))

        @info msg1 * msg2 * msg4 * msg5 * msg6

        wall_time[] = time_ns()

        return nothing
    end

    # And add it as a callback to the simulation.
    add_callback!(omip, progress, IterationInterval(10))

    return omip
end
