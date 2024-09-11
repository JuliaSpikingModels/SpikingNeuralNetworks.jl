"""
    Abstract type for normalization parameters.
"""
abstract type NormParam end

"""
    MultiplicativeNorm{FT = Int32} <: NormParam

This struct holds the parameters for multiplicative normalization. 
It includes a timescale τ (default 0.0) and an operator (default multiplication).
"""
MultiplicativeNorm

@snn_kw struct MultiplicativeNorm{FT = Int32} <: NormParam
    τ::Float32 = 0.0f0
    operator::Function = *
end

"""
    AdditiveNorm{FT = Float32} <: NormParam

This struct holds the parameters for additive normalization. 
It includes a timescale τ (default 0.0) and an operator (default addition).
"""
AdditiveNorm

@snn_kw struct AdditiveNorm{FT = Float32} <: NormParam
    τ::Float32 = 0.0f0
    operator::Function = +
end

"""
    SynapseNormalization{VFT = Vector{Float32}, VIT = Vector{Int32}, MFT = Matrix{Float32}}

A struct that holds parameters for synapse normalization, including:
- param: Normalization parameter, can be either MultiplicativeNorm or AdditiveNorm.
- t: A vector of integer values representing time points.
- W0: A vector of initial weights before simulation.
- W1: A vector of weights during the simulation.
- μ: A vector of mean synaptic weights.
- records: A dictionary for storing additional data.
"""
SynapseNormalization

@snn_kw struct SynapseNormalization{
    VFT = Vector{Float32},
    VIT = Vector{Int32},
    MFT = Matrix{Float32},
}
    param::NormParam = MultiplicativeNorm()
    t::VIT = [0, 1]
    W0::VFT = [0.0f0]
    W1::VFT = [0.0f0]
    μ::VFT = [0.0f0]
    records::Dict = Dict()
end

"""
    SynapseNormalization(N; param, kwargs...)

Constructor function for the SynapseNormalization struct.
- N: The number of synapses.
- param: Normalization parameter, can be either MultiplicativeNorm or AdditiveNorm.
- kwargs: Other optional parameters.
Returns a SynapseNormalization object with the specified parameters.
"""
function SynapseNormalization(N; param, kwargs...)
    W0 = zeros(Float32, N)
    W1 = zeros(Float32, N)
    μ = zeros(Float32, N)
    SynapseNormalization(; @symdict(param, W0, W1, μ)..., kwargs...)
end