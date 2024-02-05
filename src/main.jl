function sim!(P, C, dt)
    for p in P # for all the neurons in the population
        integrate!(p, p.param, Float32(dt)) 
        record!(p)
    end
    for c in C
        forward!(c, c.param)
        record!(c)
    end
end

function sim!(P, C; dt = 0.1ms, duration = 10ms)
    for t = 0ms:dt:(duration-dt)
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
    for t = 0ms:dt:(duration-dt)
        train!(P, C, Float32(dt), Float32(t))
    end
end
