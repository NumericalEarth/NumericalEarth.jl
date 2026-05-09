#####
##### `RucEnergy` — RUC-style two-temperature flux-driven slab energy
##### balance.
#####
##### Two prognostic variables `state.T` (ground skin temperature) and
##### `state.Tc` (two-source canopy temperature, Deardorff 1978). Two
##### flux accumulators: `fluxes.temperature_flux` (kinematic
##### `Jᵀ_g` [K m s⁻¹]) and `fluxes.canopy_temperature_flux`
##### (W m⁻²). Different sign / dimension conventions because the canopy
##### is parameterised by an areal heat capacity `(ρ c H)_c` directly,
##### while the ground slab is `H_g · ρ_g · c_g`.
#####
##### `RucEnergy` does not include force-restore deep-soil memory or
##### radiation balance internal to the slab — the entire net heat flux
##### is pre-assembled by the coupler upstream and flows in as
##### `temperature_flux`. Latent-heat back-coupling from snow melt and
##### soil freeze/thaw is performed by `RucHydrology` (which writes into
##### `state.T` for those phase-change exchanges).
#####
##### References:
#####   Deardorff, J. W., 1978: Efficient prediction of ground surface
#####     temperature and moisture, with inclusion of a layer of
#####     vegetation. J. Geophys. Res., 83, 1889–1903.
#####   Smirnova, T. G., J. M. Brown, and S. G. Benjamin, 1997:
#####     Performance of different soil model configurations in
#####     simulating ground surface temperature and surface fluxes.
#####     Mon. Wea. Rev., 125, 1870–1884.
#####   Smirnova, T. G., J. M. Brown, S. G. Benjamin, and J. S. Kenyon,
#####     2016: Modifications to the Rapid Update Cycle Land Surface
#####     Model (RUC LSM) available in the WRF model. Mon. Wea. Rev.,
#####     144, 1851–1865, doi:10.1175/MWR-D-15-0198.1.
#####

"""
    RucEnergy(parameters::RucSlabLandParameters)
    RucEnergy(FT = Float64; depth, density, heat_capacity, canopy_heat_capacity)

The RUC slab energy balance. `(depth, density, heat_capacity)` define
the ground areal heat capacity `(ρ c H)_g = depth · density ·
heat_capacity` in `J m⁻² K⁻¹`. `canopy_heat_capacity` is the canopy
areal `(ρ c H)_c` directly.
"""
struct RucEnergy{FT} <: AbstractEnergyBalance
    depth                :: FT   # H_g [m]
    density              :: FT   # ρ_g [kg m⁻³]
    heat_capacity        :: FT   # c_g [J kg⁻¹ K⁻¹]
    canopy_heat_capacity :: FT   # (ρ c H)_c [J m⁻² K⁻¹]
end

function RucEnergy(FT::Type = Float64;
                   depth = 0.10,
                   density = 1500,
                   heat_capacity = 1480,
                   canopy_heat_capacity = 1.0e4)
    return RucEnergy{FT}(convert(FT, depth),
                          convert(FT, density),
                          convert(FT, heat_capacity),
                          convert(FT, canopy_heat_capacity))
end

RucEnergy(p::RucSlabLandParameters{FT}) where FT =
    RucEnergy(FT;
              depth = p.depth,
              density = p.density,
              heat_capacity = p.heat_capacity,
              canopy_heat_capacity = p.canopy_heat_capacity)

prognostic_variables(::RucEnergy) = (:T, :Tc)
flux_variables(::RucEnergy)       = (:temperature_flux, :canopy_temperature_flux)

# Areal heat capacity helpers for closures that need to convert latent
# heat into a slab T increment.
@inline ground_heat_capacity(e::RucEnergy)  = e.density * e.heat_capacity * e.depth
@inline canopy_heat_capacity(e::RucEnergy)  = e.canopy_heat_capacity

@kernel function _ruc_step_temperature!(T, Jᵀ, Δt, H)
    i, j = @index(Global, NTuple)
    @inbounds T[i, j, 1] -= Jᵀ[i, j, 1] * Δt / H
end

@kernel function _ruc_step_canopy_temperature!(Tc, Jᵀ_c, Δt, H_canopy_eff)
    i, j = @index(Global, NTuple)
    @inbounds Tc[i, j, 1] -= Jᵀ_c[i, j, 1] * Δt / H_canopy_eff
end

function step!(energy::RucEnergy, state, fluxes, surface, parameters, grid, Δt)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _ruc_step_temperature!,
            state.T, fluxes.temperature_flux, Δt, energy.depth)
    launch!(arch, grid, :xy, _ruc_step_canopy_temperature!,
            state.Tc, fluxes.canopy_temperature_flux, Δt, energy.canopy_heat_capacity)
    return nothing
end

surface_temperature(::RucEnergy, state) = state.T

Base.summary(e::RucEnergy{FT}) where FT =
    string("RucEnergy{$FT}((ρcH)_g=", ground_heat_capacity(e),
           " J m⁻² K⁻¹, (ρcH)_c=", e.canopy_heat_capacity, " J m⁻² K⁻¹)")
