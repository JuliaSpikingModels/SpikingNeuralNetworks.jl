abstract type SpikingSynapse <: AbstractSparseSynapse end
abstract type SynapseTripod <: AbstractSparseSynapse end

include("spiking_synapses/spiking_synapse_IF.jl")
include("spiking_synapses/spiking_synapse_Tripod.jl")
# include("plasticity_parameters.jl")

# export SpikingSynapse
