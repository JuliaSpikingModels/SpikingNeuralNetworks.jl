C = 281pF        #(pF)
gL = 40nS         #(nS) leak conductance #BretteGerstner2005 says 30 nS

@snn_kw struct AdExParameter{FT = Float32} <: AbstractIFParameter
    τm::FT = C / gL # Membrane time constant
    # τe::FT = 5ms # Time constant excitatory synapse
    # τi::FT = 10ms # Time constant inhibitory synapse
    Vt::FT = -50mV # Membrane potential threshold
    Vr::FT = -70.6mV # Reset potential
    El::FT = Vr # Resting membrane potential 
    R::FT = nS / gL # Resistance
    ΔT::FT = 2mV # Slope factor
    τw::FT = 144ms # Adaptation time constant (Spike-triggered adaptation time scale)
    a::FT = 4nS # Subthreshold adaptation parameter
    b::FT = 80.5nA # Spike-triggered adaptation parameter (amount by which the voltage is increased at each threshold crossing)

    ## Synapses
    τre::FT = 1ms # Rise time for excitatory synapses
    τde::FT = 6ms # Decay time for excitatory synapses
    τri::FT = 0.5ms # Rise time for inhibitory synapses
    τdi::FT = 2ms # Decay time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential excitatory synapses 
    E_e::FT = 0mV #Reversal potential excitatory synapses

    ## Dynamic spike threshold
    At::FT = 10mV # Post spike threshold increase
    τT::FT = 30ms # Adaptive threshold time scale
end

@snn_kw mutable struct AdEx{VFT = Vector{Float32},VBT = Vector{Bool}} <: AbstractIF
    param::AdExParameter = AdExParameter()
    N::Int32 = 100 # Number of neurons
    v::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    w::VFT = zeros(N) # Adaptation current
    fire::VBT = zeros(Bool, N) # Store spikes
    θ::VFT = ones(N) * param.Vt # Array with membrane potential thresholds
    I::VFT = zeros(N) # Current
    # synaptic conductance
    ge::VFT = zeros(N) # Time-dependent conductivity that opens whenever a presynaptic excitatory spike arrives
    gi::VFT = zeros(N) # Time-dependent conductivity that opens whenever a presynaptic inhibitory spike arrives
    he::Vector{Float64} = zeros(N)
    hi::Vector{Float64} = zeros(N)
    records::Dict = Dict()
end

"""
	[Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
"""

function integrate!(p::AdEx, param::AdExParameter, dt::Float32)
    @unpack N, v, w, ge, gi, fire, I, θ, records, he, hi = p
    @unpack τm, Vt, Vr, El, R, ΔT, τw, a, b, τre, τde, τri, τdi, At, τT, E_i, E_e = param
    @inbounds for i ∈ 1:N

        # Membrane potential
        v[i] +=
            dt * 1 / τm * (
                - (v[i] - El)  # leakage
                + ΔT * exp((v[i] - θ[i]) / ΔT) # exponential term
                - R * (ge[i]*(v[i]-E_e ) + gi[i]*(v[i]-E_i)) #synaptic term
                - R * w[i] + I[i] # adaptation
            ) 


        # mV = mV + ms * (nS + nS - (mV - mV) + mV * exp((mV - mV)/mV) - 1/nS * pA + pA) / ms
        # v: mV
        # dt: ms
        # ge: nS (doesn't match ?)
        # gi: nS (doesn't match ?)
        # El: mV
        # ΔT: mV
        # θ: mV
        # R = 1/gL: 1/nS --> ohm's law: I=V×C=V/R --> V=IxR (mV = pA x 1/nS, 10^-3 = 10^-12/10^-9)
        # I: pA (doesn't match ?)
        # w: pA

        # Adaptation current
        w[i] += dt * (a * (v[i] - El) - w[i]) / τw
        # pA = pA + ms * (nS * (mV - mV) - pA) / ms  --> ohm's law: I = V x 1/R (pA = mV x nS)
        # w: pA
        # a: nS

        # Conductance-based synapse model: uses the dynamics of the ion channels on the membrane of the post-synaptic neuron to describe the synapse

        # Single exponential
        # ge[i] += dt * -ge[i] / τe # Excitatory synapse
        # gi[i] += dt * -gi[i] / τi # Inhibitory synapse

        # Double exponential version 1
        ge[i] += - dt * ge[i] / τde + he[i]
        he[i] += - dt * he[i] / τre

        gi[i] += - dt * gi[i] / τdi + hi[i]
        hi[i] += - dt * hi[i] / τri

        θ[i] += dt * (Vt - θ[i]) / τT
        # Double exponential version 2
        # ge[i] = exp32(-dt*τd⁻)*(ge[i] + dt*he[i])
        # he[i] = exp32(-dt*τr⁻)*(he[i])

        # gi[i] = exp32(-dt*τd⁻)*(gi[i] + dt*hi[i])
        # hi[i] = exp32(-dt*τr⁻)*(hi[i])

    end
    @inbounds for i ∈ 1:N # TODO: why separate? don't we need to check if there is a spike in the previous loop
        # to update correctly v and vt?

        # if i ==333 
        #     @debug v[i], θ[i]
        # end

        fire[i] = v[i] > 0.0
    
        θ[i] = ifelse(fire[i], θ[i] + At, θ[i]) 
        v[i] = ifelse(fire[i], Vr, v[i]) # if there is a spike, set membrane potential to reset potential
        w[i] = ifelse(fire[i], w[i] + b * τw, w[i]) # if there is a spike, increase adaptation current by an amount of b
        # b: pA
    end
end

# function spike_count(p::AdEx)
