## Set physiology
SNN.@load_units
struct Physiology
    Ri::Float32 ## in Ω*cm
    Rd::Float32 ## in Ω*cm^2
    Cd::Float32 ## in pF/cm^2
end

HUMAN = Physiology(200 * Ω * cm, 38907 * Ω * cm^2, 0.5μF / cm^2)
MOUSE = Physiology(200 * Ω * cm, 1700Ω * cm^2, 1μF / cm^2)

"""
    G_axial(;Ri=Ri,d=d,l=l)
    Axial conductance of a cylinder of length l and diameter d
    return Conductance in nS
"""
function G_axial(; Ri = Ri, d = d, l = l)
    ((π * d * d) / (Ri * l * 4))
end

"""
    G_mem(;Rd=Rd,d=d,l=l)
    Membrane conductance of a cylinder of length l and diameter d
    return Conductance in nS
"""
function G_mem(; Rd = Rd, d = d, l = l)
    ((l * d * π) / Rd)
end

"""
    C_mem(;Cd=Cd,d=d,l=l)
    Capacitance of a cylinder of length l and diameter d
    return Capacitance in pF
"""
function C_mem(; Cd = Cd, d = d, l = l)
    (Cd * π * d * l)
end

function create_dendrite(; d::Real = 4um, l::Real, s = "H")
    @unpack Ri, Rd, Cd = s == "M" ? MOUSE : HUMAN
    if l <= 0
        return (gm = 1.0f0, gax = 0.0f0, C = 1.0f0, l = -1, d = d)
    else
        return (
            gm = G_mem(Rd = Rd, d = d, l = l),
            gax = G_axial(Ri = Ri, d = d, l = l),
            C = C_mem(Cd = Cd, d = d, l = l),
            l = l,
            d = d,
        )
    end
end

export create_dendrite