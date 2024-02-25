C = 281pF        #(pF)
gL = 40nS         #(nS) leak conductance #BretteGerstner2005 says 30 nS

@snn_kw struct AdExConstParameter{FT = Float32} <: AbstractIFParameter
    τm::FT = C / gL # Membrane time constant
    Vt::FT = -50mV # Membrane potential threshold
    Vr::FT = -70.6mV # Reset potential
    El::FT = Vr # Resting membrane potential 
    R::FT = nS / gL # Resistance
    C::FT = 300pF
    ΔT::FT = 2mV # Slope factor
    a::FT = 4nS # Subthreshold adaptation parameter
    b::FT = 80.5nA # Spike-triggered adaptation parameter (amount by which the voltage is increased at each threshold crossing)
    τw::FT = 144ms # Adaptation time constant (Spike-triggered adaptation time scale)
    τabs::FT = 1ms # Absolute refractory period

    ## Dynamic spike threshold
    At::FT = 10mV # Post spike threshold increase
    τT::FT = 30ms # Adaptive threshold time scale
end

@snn_kw mutable struct AdExConst{VFT = Vector{Float32},VBT = Vector{Bool}} <: AbstractIF
    param::AdExConstParameter = AdExConstParameter()
    N::Int32 = 100 # Number of neurons
    v::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    w::VFT = zeros(N) # Adaptation current
    fire::VBT = zeros(Bool, N) # Store spikes
    θ::VFT = ones(N) * param.Vt # Array with membrane potential thresholds
    I::VFT = zeros(N) # Current
    records::Dict = Dict()
    timespikes::Vector{Float64} = zeros(N)
end

"""
	[Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
"""

function integrate!(p::AdExConst, param::AdExConstParameter, dt::Float32, t::Float64)
    @unpack N, v, w, fire, θ, I, records, timespikes = p
    @unpack τm, Vt, Vr, El, R, C, ΔT, a, b, τw, τabs, At, τT = param
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
                - R * w[i] # adaptation
            ) / τm

        w[i] += dt * (a * (v[i] - El) - w[i]) / τw        
    end

    @inbounds for i ∈ 1:N # iterates over all neurons at a specific time step
        # Refractory period
        if (t - timespikes[i]) < τabs
            v[i] = Vr
            w[i] = w[i]
            continue
        end

        fire[i] = v[i] > 0 # It's not Vt anymore because of the exponential term 
        v[i] = ifelse(fire[i], Vr, v[i]) # if there is a spike, set membrane potential to reset potential
        w[i] = ifelse(fire[i], w[i] + b, w[i]) # if there is a spike, increase adaptation current by an amount of b 
        if fire[i]
            timespikes[i] = t
        end 
    end
end

# function spike_count(p::AdEx)
