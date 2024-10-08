abstract type  AbstractAdExParameter <: AbstractGeneralizedIFParameter end

C = 281pF        #(pF)
gL = 40nS         #(nS) leak conductance #BretteGerstner2005 says 30 nS

@snn_kw struct AdExParameter{FT = Float32} <: AbstractAdExParameter
	τm::FT = C / gL # Membrane time constant
	Vt::FT = -50mV # Membrane potential threshold
	Vr::FT = -70.6mV # Reset potential
	El::FT = -70.6mV # Resting membrane potential 
	R::FT = nS / gL # Resistance
	ΔT::FT = 2mV # Slope factor
	τw::FT = 144ms # Adaptation time constant (Spike-triggered adaptation time scale)
	a::FT = 4nS # Subthreshold adaptation parameter
	b::FT = 80.5pA # Spike-triggered adaptation parameter (amount by which the voltage is increased at each threshold crossing)
	τabs::FT = 1ms # Absolute refractory period

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

@snn_kw struct AdExParameterSingleExponential{FT = Float32} <: AbstractAdExParameter
	τm::FT = C / gL # Membrane time constant
	Vt::FT = -50mV # Membrane potential threshold
	Vr::FT = -70.6mV # Reset potential
	El::FT = -70.6mV # Resting membrane potential 
	R::FT = nS / gL # Resistance
	ΔT::FT = 2mV # Slope factor
	τw::FT = 144ms # Adaptation time constant (Spike-triggered adaptation time scale)
	a::FT = 4nS # Subthreshold adaptation parameter
	b::FT = 80.5pA # Spike-triggered adaptation parameter (amount by which the voltage is increased at each threshold crossing)
	τabs::FT = 1ms # Absolute refractory period

	## Synapses
	τe::FT = 6ms # Decay time for excitatory synapses
	τi::FT = 0.5ms # Rise time for inhibitory synapses
	E_i::FT = -75mV # Reversal potential excitatory synapses 
	E_e::FT = 0mV #Reversal potential excitatory synapses

	## Dynamic spike threshold
	At::FT = 10mV # Post spike threshold increase
	τT::FT = 30ms # Adaptive threshold time scale
end

@snn_kw  struct AdEx{VFT = Vector{Float32}, VBT = Vector{Bool}, AdExT <: AbstractAdExParameter} <: AbstractGeneralizedIF
	param::AdExT = AdExParameter()
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

function integrate!(p::AdEx, param::T, dt::Float32) where T<:AbstractAdExParameter
	update_synapses!(p, param, dt)
	update_soma!(p, param, dt)

end

function update_synapses!(p::AdEx, param::AdExParameter, dt::Float32)
	@unpack N, ge, gi, he, hi = p
	@unpack τde, τre, τdi, τri = param
	@inbounds for i ∈ 1:N
		ge[i] += dt * (-ge[i] / τde + he[i])
		he[i] -= dt * he[i] / τre
		gi[i] += dt * (-gi[i] / τdi + hi[i])
		hi[i] -= dt * hi[i] / τri
	end
end

function update_synapses!(p::AdEx, param::AdExParameterSingleExponential, dt::Float32)
	@unpack N, ge, gi, he, hi = p
	@unpack τe, τi = param
	@fastmath @inbounds for i ∈ 1:N
		ge[i] += dt * (-ge[i] / τe)
		gi[i] += dt * (-gi[i] / τi)
	end
end

function update_soma!(p::AdEx, param::T, dt::Float32) where T<:AbstractAdExParameter
	@unpack N, v, w, fire, θ, I, ge, gi, = p
	@unpack τm, Vt, Vr, El, R, ΔT, τw, a, b, At, τT,E_e, E_i  = param
	@inbounds for i ∈ 1:N
		# Adaptation current 
		w[i] += dt * (a * (v[i] - El) - w[i]) / τw
		# Membrane potential
		v[i] +=
			dt * (
				-(v[i] - El)  # leakage
				+
				ΔT * exp((v[i] - θ[i]) / ΔT) # exponential term
				+
				R * ge[i] * (E_e - v[i]) +
				R * gi[i] * (E_i - v[i]) #synaptic term: conductance times membrane potential difference gives synaptic current
				-
				R * w[i] # adaptation
			) / τm
		# Double exponential
		θ[i] += dt * (Vt - θ[i]) / τT

		# Refractory period
		v[i] = ifelse(fire[i], Vr, v[i])
		fire[i] = v[i] > θ[i] + 5.0f0
		v[i] = ifelse(fire[i], 10.0f0, v[i]) # if there is a spike, set membrane potential to reset potential
		θ[i] = ifelse(fire[i], θ[i] + At, θ[i])
		w[i] = ifelse(fire[i], w[i] + b, w[i]) # if there is a spike, increase adaptation current by an amount of b 
	end
end

export AdEx, AdExParameter, AdExParameterSingleExponential