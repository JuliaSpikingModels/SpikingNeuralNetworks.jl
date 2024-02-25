C = 281pF        #(pF)
gL = 40nS         #(nS) leak conductance #BretteGerstner2005 says 30 nS

@snn_kw struct IFConstParameter{FT = Float32} <: AbstractIFParameter
    τm::FT = C / gL # Membrane time constant
    Vt::FT = -50mV # Membrane potential threshold
    Vr::FT = -70.6mV # Reset potential
    El::FT = Vr # Resting membrane potential 
    R::FT = nS / gL # Resistance
    C::FT = 300pF
    ΔT::FT = 2mV # Slope factor
    τabs::FT = 1ms # Absolute refractory period
end

@snn_kw mutable struct IFConst{VFT = Vector{Float32},VBT = Vector{Bool}} <: AbstractIF
    param::IFConstParameter = IFConstParameter()
    N::Int32 = 100 # Number of neurons
    v::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    fire::VBT = zeros(Bool, N) # Store spikes
    θ::VFT = ones(N) * param.Vt # Array with membrane potential thresholds
    I::VFT = zeros(N) # Current
    records::Dict = Dict()
    timespikes::Vector{Float64} = zeros(N)
end

"""
	[Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
"""

function integrate!(p::IFConst, param::IFConstParameter, dt::Float32, t::Float64)
    @unpack N, v, fire, θ, I, records, timespikes = p
    @unpack τm, Vt, Vr, El, R, C, ΔT, τabs = param
    @inbounds for i ∈ 1:N

        # Refractory period
        if (t - timespikes[i]) < τabs
            v[i] = v[i]
            continue
        end

        # Membrane potential
        v[i] +=
            dt * (
                - (v[i] - El)  # leakage
                + ΔT * exp((v[i] - θ[i]) / ΔT) # exponential term
                + R * 500pA  # constant input current
            ) / τm
        
    end

    @inbounds for i ∈ 1:N # iterates over all neurons at a specific time step
        # Refractory period
        if (t - timespikes[i]) < τabs
            v[i] = Vr
            continue
        end

        fire[i] = v[i] > 0 # It's not Vt anymore because of the exponential term 
        v[i] = ifelse(fire[i], Vr, v[i]) # if there is a spike, set membrane potential to reset potential
        if fire[i]
            timespikes[i] = t
        end 
    end
end

# function spike_count(p::AdEx)
