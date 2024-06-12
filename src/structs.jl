abstract type AbstractSynapseParameter end
abstract type AbstractNeuronParameter end

abstract type AbstractIFParameter <: AbstractNeuronParameter end
abstract type SpikingSynapseParameter<:AbstractSynapseParameter end