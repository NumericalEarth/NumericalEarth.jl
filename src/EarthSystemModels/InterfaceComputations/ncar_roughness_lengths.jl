"""
    NCARMomentumRoughnessLength{FT}

Momentum roughness length derived from the NCAR neutral 10-m drag coefficient
parameterization of Large & Yeager (2004).

The neutral drag coefficient ``C_{d,N10}`` is computed as a polynomial function
of wind speed ``U`` (floored at 0.5 m/s):

```math
1000 \\, C_{d,N10} = \\frac{2.7}{U} + 0.142 + \\frac{U}{13.09} - 3.14807 \\times 10^{-10} \\, U^6
```

for ``U < 33`` m/s, and ``C_{d,N10} = 2.34 \\times 10^{-3}`` otherwise.

The roughness length is then back-computed as

```math
z_0 = h \\exp\\left( - \\frac{\\kappa}{\\sqrt{C_{d,N10}}} \\right)
```

where ``h`` is the reference height (10 m) and ``\\kappa`` is the von Kármán constant.
"""
struct NCARMomentumRoughnessLength{FT}
    von_karman_constant :: FT
    reference_height :: FT
end

Base.summary(::NCARMomentumRoughnessLength{FT}) where FT = "NCARMomentumRoughnessLength{$FT}"
Base.show(io::IO, ::NCARMomentumRoughnessLength{FT}) where FT = print(io, "NCARMomentumRoughnessLength{$FT}")

"""
    NCARMomentumRoughnessLength(FT = Oceananigans.defaults.FloatType;
                                 von_karman_constant = 0.4,
                                 reference_height = 10)

Construct an `NCARMomentumRoughnessLength` using the Large & Yeager (2004) drag
coefficient parameterization.

Keyword Arguments
=================

- `von_karman_constant`: The von Kármán constant. Default: 0.4.
- `reference_height`: The reference height in meters. Default: 10.
"""
function NCARMomentumRoughnessLength(FT = Oceananigans.defaults.FloatType;
                                      von_karman_constant = 0.4,
                                      reference_height = 10)

    return NCARMomentumRoughnessLength(convert(FT, von_karman_constant),
                                        convert(FT, reference_height))
end

@inline function roughness_length(ℓ::NCARMomentumRoughnessLength{FT}, u★, U, args...) where FT
    κ = ℓ.von_karman_constant
    h = ℓ.reference_height

    # Floor wind speed at 0.5 m/s
    U = max(U, convert(FT, 0.5))

    # NCAR neutral 10-m drag coefficient (Large & Yeager 2004)
    Cd_N10 = ifelse(U < convert(FT, 33),
                     (convert(FT, 2.7) / U + convert(FT, 0.142) + U / convert(FT, 13.09) - convert(FT, 3.14807e-10) * U^6) / convert(FT, 1000),
                     convert(FT, 2.34e-3))

    # Back-compute roughness length from drag coefficient
    z₀ = h * exp(-κ / sqrt(Cd_N10))

    return z₀
end

"""
    NCARScalarRoughnessLength{FT}

Scalar roughness length derived from the NCAR neutral scalar transfer coefficient
parameterization of Large & Yeager (2004).

The neutral scalar transfer coefficient is related to the drag coefficient as

```math
1000 \\, C_{x,N10} = c \\, \\sqrt{C_{d,N10}}
```

where ``c`` is an empirical coefficient (32.7 for heat, 34.6 for moisture).

The scalar roughness length is then back-computed as

```math
z_{0s} = h \\exp\\left( - \\frac{\\kappa \\, \\sqrt{C_{d,N10}}}{C_{x,N10}} \\right)
```

where ``h`` is the reference height and ``\\kappa`` is the von Kármán constant.
"""
struct NCARScalarRoughnessLength{FT}
    von_karman_constant :: FT
    reference_height :: FT
    coefficient :: FT
end

Base.summary(::NCARScalarRoughnessLength{FT}) where FT = "NCARScalarRoughnessLength{$FT}"
Base.show(io::IO, ::NCARScalarRoughnessLength{FT}) where FT = print(io, "NCARScalarRoughnessLength{$FT}")

"""
    NCARScalarRoughnessLength(FT = Oceananigans.defaults.FloatType;
                               von_karman_constant = 0.4,
                               reference_height = 10,
                               coefficient = 32.7)

Construct an `NCARScalarRoughnessLength` using the Large & Yeager (2004) scalar
transfer coefficient parameterization.

Keyword Arguments
=================

- `von_karman_constant`: The von Kármán constant. Default: 0.4.
- `reference_height`: The reference height in meters. Default: 10.
- `coefficient`: The empirical coefficient relating scalar and drag coefficients.
  Use 32.7 for heat (unstable) and 34.6 for moisture. Default: 32.7.
"""
function NCARScalarRoughnessLength(FT = Oceananigans.defaults.FloatType;
                                    von_karman_constant = 0.4,
                                    reference_height = 10,
                                    coefficient = 32.7)

    return NCARScalarRoughnessLength(convert(FT, von_karman_constant),
                                      convert(FT, reference_height),
                                      convert(FT, coefficient))
end

@inline function roughness_length(ℓ::NCARScalarRoughnessLength{FT}, ℓu, u★, U, args...) where FT
    κ = ℓ.von_karman_constant
    h = ℓ.reference_height
    c = ℓ.coefficient

    # Floor wind speed at 0.5 m/s
    U = max(U, convert(FT, 0.5))

    # NCAR neutral 10-m drag coefficient (Large & Yeager 2004)
    Cd_N10 = ifelse(U < convert(FT, 33),
                     (convert(FT, 2.7) / U + convert(FT, 0.142) + U / convert(FT, 13.09) - convert(FT, 3.14807e-10) * U^6) / convert(FT, 1000),
                     convert(FT, 2.34e-3))

    # Scalar transfer coefficient: 1000 * Cx_N10 = c * sqrt(Cd_N10)
    Cx_N10 = c * sqrt(Cd_N10) / convert(FT, 1000)

    # Back-compute scalar roughness length
    z₀s = h * exp(-κ * sqrt(Cd_N10) / Cx_N10)

    return z₀s
end
