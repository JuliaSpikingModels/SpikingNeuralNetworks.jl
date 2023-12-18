C = 281pF        #(pF)
gL = 40nS         #(nS) leak conductance #BretteGerstner2005 says 30 nS

@snn_kw struct AdExParameter{FT=Float32} <: AbstractIFParameter
	τm::FT = C/gL # Membrane time constant
    τe::FT = 5ms # Time constant excitatory synapse
    τi::FT = 10ms # Time constant inhibitory synapse
    Vt::FT = -50mV # Membrane potential threshold
    Vr::FT = -70.6mV # Reset potential
    El::FT = Vr # Resting membrane potential 
	R::FT  = nS/gL # Resistance
	ΔT::FT = 2mV # Slope factor
	τw::FT = 144ms # Adaptation time constant (Spike-triggered adaptation time scale)
	a::FT = 4nS # Subthreshold adaptation parameter
	b::FT = 80.5nA # Spike-triggered adaptation parameter (amount by which the voltage is increased at each threshold crossing)
end

@snn_kw mutable struct AdEx{VFT = Vector{Float32},VBT = Vector{Bool}} <: AbstractIF
    param::AdExParameter = AdExParameter()
    N::Int32 = 100 # Number of time steps / time window
    v::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    w::VFT = zeros(N) # Adaptation current
    ge::VFT = zeros(N) # Time-dependent conductivity that opens whenever a presynaptic excitatory spike arrives
    gi::VFT = zeros(N) # Time-dependent conductivity that opens whenever a presynaptic inhibitory spike arrives
    fire::VBT = zeros(Bool, N) # Store spikes
    θ::VFT = ones(N)*param.Vt # Array with membrane potential thresholds
    I::VFT = zeros(N) # Current
    records::Dict = Dict()
end

"""
    [Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
"""

function integrate!(p::AdEx, param::AdExParameter, dt::Float32)
    @unpack N, v, w, ge, gi, fire, I, θ = p
    @unpack τm, τe, τi, Vt, Vr, El, R, ΔT, τw, a, b = param
    @inbounds for i = 1:N

        # Membrane potential
        v[i] += dt * 1/τm * (R*(ge[i] + gi[i]) - (v[i] - El) + ΔT * exp((v[i] - θ[i]) / ΔT) - R * w[i] + I[i])
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
		w[i] += dt * (a*(v[i]-El) -w[i] )/τw 
        # pA = pA + ms * (nS * (mV - mV) - pA) / ms  --> ohm's law: I = V x 1/R (pA = mV x nS)
        # w: pA
        # a: nS

        # Conductance-based synapse model: uses the dynamics of the ion channels on the membrane of the post-synaptic neuron to describe the synapse
        ge[i] += dt * -ge[i] / τe # Excitatory synapse
        gi[i] += dt * -gi[i] / τi # Inhibitory synapse
    end
    @inbounds for i = 1:N
        fire[i] = v[i] > 0.
        v[i] = ifelse(fire[i], Vr, v[i]) # if there is a spike, set membrane potential to reset potential
		w[i] = ifelse(fire[i], w[i]+b*τw, w[i]) # if there is a spike, increase adaptation current by an amount of b
        # b: pA
    end
end

# function spike_count(p::AdEx)
