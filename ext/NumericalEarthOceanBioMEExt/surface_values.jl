import OceanBioME.Models.GasExchangeModel: surface_value

struct OnlineWindVelocity{U, V, G}
       u :: U
       v :: V
    grid :: G
end

@inline function surface_value(f::NamedTuple{(:u, :v), }, i, j, grid, clock, args...)
    t = clock.time

    x = xnode(i, j, grid.Nz, grid, Center(), Center(), Center())
    y = ynode(i, j, grid.Nz, grid, Center(), Center(), Center())

    return interpolate((x, y, 0), Time(clock.time), f.fts, f.location, f.grid)
end
