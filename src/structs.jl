abstract type AbstractSynapseParameter end
abstract type AbstractNeuronParameter end
abstract type AbstractSynapse end
abstract type AbstractNeuron end

abstract type AbstractSparseSynapse <: AbstractSynapse end
abstract type AbstractNormalization <: AbstractSynapse end

Spiketimes = Vector{Vector{Float32}}
export Spiketimes
