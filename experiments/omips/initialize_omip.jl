function initialize_omip!(model, checkpoint)
    ocean  = model.ocean.model
    seaice = model.sea_ice.model

    file = JLD2.jldopen(checkpoint)

    uo = file["uo"]
    vo = file["vo"]
    wo = file["wo"]
    To = file["To"]
    So = file["So"]
    eo = try
        file["eo"]
    catch
            0
    end
    ηo = file["ηo"]
    ui = file["ui"]
    vi = file["vi"]
    hi = file["hi"]
    ℵi = file["ℵi"]

    clock = file["clock"]

    set!(ocean, u=uo, v=vo, w=wo, T=To, S=So, η=ηo)

    try 
        set!(ocean, e=eo)
    catch
        @warn "It was not possible to restart e"
    end
    set!(seaice, h=hi, ℵ=ℵi)
    set!(seaice.velocities.u, ui)
    set!(seaice.velocities.v, vi)

    synch!(ocean.clock,  clock)
    synch!(seaice.clock, clock)

    Oceananigans.initialize!(ocean)
end

function synch!(clock1::Clock, clock2)
    # Synchronize the clocks
    clock1.time = clock2.time
    clock1.iteration = clock2.iteration
    clock1.last_Δt = clock2.last_Δt
end

synch!(model1, model2) = synch!(model1.clock, model2.clock)
