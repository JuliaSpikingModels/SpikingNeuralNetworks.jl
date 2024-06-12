@snn_kw struct IFParameter{FT = Float32} <: AbstractIFParameter
    τm::FT = 20ms
    Vt::FT = -50mV
    Vr::FT = -60mV
    El::FT = Vr
    R::FT = nS / gL # Resistance
    ΔT::FT = 2mV # Slope factor
    τre::FT = 1ms # Rise time for excitatory synapses
    τde::FT = 6ms # Decay time for excitatory synapses
    τri::FT = 0.5ms # Rise time for inhibitory synapses
    τdi::FT = 2ms # Decay time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential
    E_e::FT = 0mV # Reversal potential
    τabs::FT = 1ms # Absolute refractory period
end

abstract type AbstractIF end

@snn_kw mutable struct IF{VFT = Vector{Float32},VBT = Vector{Bool}} <: AbstractIF
    param::IFParameter = IFParameter()
    N::Int32 = 100
    v::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    ge::VFT = zeros(N)
    gi::VFT = zeros(N)
    fire::VBT = zeros(Bool, N)
    I::VFT = zeros(N)
    records::Dict = Dict()
    he::Vector{Float64} = zeros(N)
    hi::Vector{Float64} = zeros(N)
    timespikes::Vector{Float64} = zeros(N)
end

"""
    [Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
"""
IF

function integrate!(p::IF, param::IFParameter, dt::Float32)
    @unpack N, v, ge, gi, fire, I, records, he, hi, timespikes = p
    @unpack τm, Vt, Vr, El, R, ΔT, τre, τde, τri, τdi, E_i, E_e, τabs = param
    @inbounds for i = 1:N
        # # Refractory period
        # if (t - timespikes[i]) < τabs
        #     v[i] = v[i]
        #     continue
        # end

        v[i] += dt * (
            - (v[i] - El)  # leakage
            + R * ge[i] * (E_e - v[i]) + R * gi[i] * (E_i - v[i]) #synaptic term
        ) / τm

        ge[i] += dt * (- ge[i] / τde + he[i]) 
        he[i] -= dt * he[i] / τre

        gi[i] += dt * (- gi[i] / τdi + hi[i])
        hi[i] -= dt * hi[i] / τri
    end

    @inbounds for i = 1:N
        # # Refractory period
        # if (t - timespikes[i]) < τabs
        #     v[i] = Vr
        #     continue
        # end

        fire[i] = v[i] > Vt
        v[i] = ifelse(fire[i], Vr, v[i])


    end
end
