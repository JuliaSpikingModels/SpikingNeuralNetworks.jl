abstract type SpikingSynapseParameter <: AbstractSynapseParameter end
abstract type PlasticityVariables end
struct no_STDPParameter <: SpikingSynapseParameter end
struct no_PlasticityVariables <: PlasticityVariables end

function get_variables(param::no_STDPParameter, Npre, Npost)
    return no_PlasticityVariables()
end

@snn_kw struct STDPParameter{FT = Float32} <: SpikingSynapseParameter
    τpre::FT = 20ms
    τpost::FT = 20ms
    Wmax::FT = 0.01
    ΔApre::FT = 0.01 * Wmax
    ΔApost::FT = -ΔApre * τpre / τpost * 1.05
end

@snn_kw struct STDPVariables{VFT = Vector{Float32}, IT =  Int} <: PlasticityVariables
    ## Plasticity variables
    Npost::IT
    Npre::IT
    tpre::VFT = zeros(Npre) # presynaptic spiking time
    tpost::VFT = zeros(Npost) # postsynaptic spiking time
    Apre::VFT = zeros(Npre) # presynaptic trace
    Apost::VFT = zeros(Npost) # postsynaptic trace
end

function get_variables(param::STDPParameter, Npre, Npost)
    return STDPVariables(Npre=Npre, Npost=Npost)
end

@snn_kw struct vSTDPParameter{FT = Float32} <: SpikingSynapseParameter
    A_LTD::FT = 8 * 10e-5pA / mV
    A_LTP::FT = 14 * 10e-5pA / (mV * mV)
    θ_LTD::FT = -70mV
    θ_LTP::FT = -49mV
    τu::FT = 20ms
    τv::FT = 7ms
    τx::FT = 15ms
    Wmax::FT = 30.0pF
    Wmin::FT = 0.1pF
end

@snn_kw struct vSTDPVariables{VFT = Vector{Float32}, IT= Int} <: PlasticityVariables
    ## Plasticity variables
    Npost::IT
    Npre::IT
    u::VFT = zeros(Npost) # presynaptic spiking time
    v::VFT = zeros(Npost) # postsynaptic spiking time
    x::VFT = zeros(Npost) # postsynaptic spiking time
end

function get_variables(param::vSTDPParameter, Npre, Npost)
    return vSTDPVariables(Npre=Npre, Npost=Npost)
end

abstract type iSTDPParameter <: SpikingSynapseParameter end

@snn_kw struct iSTDPParameterRate{FT = Float32} <: iSTDPParameter
    η::FT = 0.1pA
    r::FT = 3Hz
    τy::FT = 50ms
    Wmax::FT = 243pF
    Wmin::FT = 0.01pF
end

@snn_kw mutable struct iSTDPParameterPotential{FT = Float32} <: iSTDPParameter
    η::FT = 0.01pA
    v0::FT = -50mV
    τy::FT = 200ms
    Wmax::FT = 243pF
    Wmin::FT = 0.01pF
end

@snn_kw struct iSTDPVariables{VFT = Vector{Float32}, IT=Int} <: PlasticityVariables
    ## Plasticity variables
    Npost::IT
    Npre::IT
    tpost::VFT = zeros(Npost) # postsynaptic spiking time 
    tpre::VFT = zeros(Npre) # presynaptic spiking time
end

function get_variables(param::T, Npre, Npost) where T <: iSTDPParameter
    return iSTDPVariables(Npre=Npre, Npost=Npost)
end

# vSTDPParameter_dendrites = 
#     A_LTD::FT = 8 * 10e-5pA / mV
#     A_LTP::FT = 14 * 10e-5pA / (mV * mV)
#     θ_LTD::FT = -70mV
#     θ_LTP::FT = -49mV
#     τu::FT = 20ms
#     τv::FT = 7ms
#     τx::FT = 15ms
#     Wmax::FT = 50.0pF
#     Wmin::FT = 0.01pF
# end
