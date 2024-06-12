function sim!(P::Vector, C::Vector, dt::Float32)
    # Threads.@threads 
    for p in P
        integrate!(p, getfield(p,:param), dt)
        record!(p)
    end
    # Threads.@threads 
    for c in C
        forward!(c, getfield(c,:param))
        record!(c)
    end
end

function sim!(P, C; dt = 0.1f0, duration = 10.f0)
    dt = Float32(dt)
    duration= Float32(duration)
    pbar = ProgressBar(0.f0:dt:(duration - dt))
    for t = pbar
        sim!(P, C, dt)
    end
end

function train!(P, C, dt, t = 0)
    for p in P
        integrate!(p, p.param, Float32(dt))
        record!(p)
    end
    for c in C
        forward!(c, c.param)
        plasticity!(c, c.param, Float32(dt), Float32(t))
        record!(c)
    end
end

function train!(P, C; dt = 0.1ms, duration = 10ms)
    for t = 0ms:dt:(duration - dt)
        train!(P, C, Float32(dt), Float32(t))
    end
end

export sim!, train!