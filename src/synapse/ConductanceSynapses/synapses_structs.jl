abstract type AbstractReceptor end
abstract type AbstractSynapse end

@with_kw struct Receptor <:AbstractReceptor
	E_rev::Float32
	τr::Float32
	τd::Float32
	gsyn::Float32
	α::Float32
	function Receptor(E_rev, τr, τd, gsyn)
		new(E_rev, τr, τd, gsyn*norm_synapse(τr,τd), α_synapse(τr, τd))
	end
	function Receptor(E_rev, τd, gsyn)
		new(E_rev, NaN, τd, gsyn,1)
	end
end

@with_kw struct ReceptorVoltage <: AbstractReceptor
	E_rev::Float32
	τr::Float32
	τd::Float32
	gsyn::Float32
	α::Float32
	b::Float32
	k::Float32
	v::Float32
	function ReceptorVoltage(E_rev, τr, τd, gsyn, b, k, v)
		new(E_rev, τr, τd, gsyn*norm_synapse(τr,τd), α_synapse(τr, τd), b, k, v)
	end
	function ReceptorVoltage(E_rev, τd, gsyn, b, k, v)
		new(E_rev, -1, τd, gsyn, 1, b, k, v)
	end
end

@with_kw struct Synapse <: AbstractSynapse
	AMPA::Receptor
	NMDA::ReceptorVoltage
	GABAa::Receptor
	GABAb::Receptor
end

@with_kw struct SynapseInh <: AbstractSynapse
	GABAa::Receptor
	GABAb::Receptor
end

@with_kw struct SynapseExc <: AbstractSynapse
	AMPA::Receptor
	NMDA::ReceptorVoltage
end

#=========================================
			Synaptic fit
=========================================#

function norm_synapse(synapse::Union{Receptor, ReceptorVoltage})
	norm_synapse(synapse.τr, synapse.τd)
end

function norm_synapse(τr,τd)
	p = [1, τr, τd]
    t_p  = p[2]*p[3]/(p[3] -p[2]) * log(p[3] / p[2])
	return 1/(-exp(-t_p/p[2]) + exp(-t_p/p[3]))
end

# α is the factor that has to be placed in-front of the differential equation as such the analytical integration corresponds to the double exponential function. Further details are discussed in the Julia notebook about synapses
function α_synapse(τr, τd)
	return (τd-τr)/(τd*τr)
end

function get_gsyn(synapse::Union{Receptor, ReceptorVoltage})
	synapse.gsyn/norm_synapse(synapse)
end
function set_gsyn(synapse::Union{Receptor, ReceptorVoltage}, value)
	synapse.gsyn = value * norm_synapse(synapse)
end

#==========================================
			Synaptic Parameters
==========================================#

function exc_inh_synapses(exc::Function, inh::Function, compartment::String)
	AMPA, NMDA = exc(compartment)
	GABAa, GABAb = inh(compartment)
    return Synapse(AMPA, NMDA, GABAa, GABAb)
end

@inline function NMDA_nonlinear(NMDA::Receptor, v::Float32)::Float32
		 return 1 ##NMDA
 end

const Mg_mM     = 1f0
@inline function NMDA_nonlinear(NMDA::ReceptorVoltage, v::Float32)::Float32
		 return (1+(Mg_mM/NMDA.b)*exp32(NMDA.k*(v)))^-1 ##NMDA
 end
