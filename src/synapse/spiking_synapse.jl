abstract type SpikingSynapseParameter end
abstract type SpikingSynapse end

@snn_kw struct STDPParameter{FT = Float32} <: SpikingSynapseParameter
    τpre::FT = 20ms
    τpost::FT = 20ms
    Wmax::FT = 0.01
    ΔApre::FT = 0.01 * Wmax
    ΔApost::FT = -ΔApre * τpre / τpost * 1.05
end

@snn_kw struct vSTDPParameter{FT = Float32} <: SpikingSynapseParameter
    A_LTD::FT = 8 *10e-5pA/mV
    A_LTP::FT = 14 *10e-5pA/(mV*mV)
    θ_LTD::FT = -70mV
    θ_LTP::FT = -49mV
    τu::FT = 10ms
    τv::FT = 7ms
    τx::FT = 15ms
    Wmax::FT = 21.4pA
    Wmin::FT = 1.78pA
end

@snn_kw struct iSTDPParameter{FT = Float32} <: SpikingSynapseParameter
    η::FT = 1pA
    r₀::FT = 3Hz
    τy::FT = 20ms
    Wmax::FT = 243pA
    Wmin::FT = 48.7pA
end

struct no_STDPParameter <:SpikingSynapseParameter end

@snn_kw mutable struct no_STDP{
VIT = Vector{Int32},
    VFT = Vector{Float32},
    VBT = Vector{Bool},
} <:SpikingSynapse
    param::SpikingSynapseParameter = no_STDPParameter()
    rowptr::VIT # row pointer of sparse W
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    J::VIT      # presynaptic index of W
    index::VIT  # index mapping: W[index[i]] = Wt[i], Wt = sparse(dense(W)')
    W::VFT  # synaptic weight
    fireJ::VBT # presynaptic firing
    g::VFT # postsynaptic conductance
    records::Dict = Dict()
end

@snn_kw mutable struct STDP{
    VIT = Vector{Int32},
    VFT = Vector{Float32},
    VBT = Vector{Bool},
} <:SpikingSynapse
    param::STDPParameter = STDPParameter()
    rowptr::VIT # row pointer of sparse W
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    J::VIT      # presynaptic index of W
    index::VIT  # index mapping: W[index[i]] = Wt[i], Wt = sparse(dense(W)')
    W::VFT  # synaptic weight
    tpre::VFT = zero(W) # presynaptic spiking time
    tpost::VFT = zero(W) # postsynaptic spiking time
    Apre::VFT = zero(W) # presynaptic trace
    Apost::VFT = zero(W) # postsynaptic trace
    fireI::VBT # postsynaptic firing
    fireJ::VBT # presynaptic firing
    g::VFT # postsynaptic conductance
    records::Dict = Dict()
end

@snn_kw mutable struct vSTDP{
    VIT = Vector{Int32},
    VFT = Vector{Float32},
    VBT = Vector{Bool},
} <:SpikingSynapse
    param::vSTDPParameter = vSTDPParameter()
    rowptr::VIT # row pointer of sparse W
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    J::VIT      # presynaptic index of W
    index::VIT  # index mapping: W[index[i]] = Wt[i], Wt = sparse(dense(W)')
    W::VFT  # synaptic weight
    u::VFT = zero(W) # presynaptic spiking time
    v::VFT = zero(W) # postsynaptic spiking time
    x::VFT = zero(W) # postsynaptic spiking time
    vpost::VFT = zero(W)
    fireJ::VBT # presynaptic firing
    g::VFT # postsynaptic conductance
    records::Dict = Dict()
end

@snn_kw mutable struct iSTDP{
    VIT = Vector{Int32},
    VFT = Vector{Float32},
    VBT = Vector{Bool},
} <:SpikingSynapse
    param::iSTDPParameter = iSTDPParameter()
    rowptr::VIT # row pointer of sparse W
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    J::VIT      # presynaptic index of W
    index::VIT  # index mapping: W[index[i]] = Wt[i], Wt = sparse(dense(W)')
    W::VFT  # synaptic weight
    yᴱ::VFT = zero(W) # presynaptic spiking time
    yᴵ::VFT = zero(W) # presynaptic spiking time
    fireI::VBT 
    fireJ::VBT # presynaptic firing
    g::VFT # postsynaptic conductance
    records::Dict = Dict()
end

"""
[Spking Synapse](https://brian2.readthedocs.io/en/2.0b4/resources/tutorials/2-intro-to-brian-synapses.html)
"""
SpikingSynapse

function SpikingSynapse(pre, post, sym; σ = 0.0, p = 0.0, kwargs...)
    w = σ * sprand(post.N, pre.N, p) # Construct a random sparse vector with length post.N, pre.N and density p
    w[findall(w.>0)] .=σ # make all the weights the same
    rowptr, colptr, I, J, index, W = dsparse(w) # Get info about the existing connections
    # rowptr: row pointer
    # colptr: column pointer
    # I: postsynaptic index of W
    # J: presynaptic index of W
    fireI, fireJ, vpost = post.fire, pre.fire, post.v
    # fireI: Stored spikes postsynaptic neuron
    # fireJ: Stored spikes presynaptic neuron
    g = getfield(post, sym)
    # g: get variable with symbol sym
    param = haskey(kwargs, :param) ? kwargs[:param] : nothing
    if isa(param, STDPParameter) 
        return STDP(;
            @symdict(rowptr, colptr, I, J, index, W, fireI, fireJ, g)...,
            kwargs...,
        )
    elseif isa(param, vSTDPParameter) 
        return vSTDP(;
            @symdict(rowptr, colptr, I, J, index, W, fireJ, vpost, g)...,
            kwargs...,
        )
    elseif isa(param, iSTDPParameter)
        return iSTDP(;
            @symdict(rowptr, colptr, I, J, index, W, fireI, fireJ, g)...,
            kwargs...,
        )
    else
        return no_STDP(;
            @symdict(rowptr, colptr, I, J, index, W, g, fireJ)...,
            kwargs...,
        )
    end
end

function SpikingSynapse(;kwargs...)
    isa(kwargs.param, STDPParameter) && STDP(kwargs...)
    isa(kwargs.param, vSTDPParameter) && vSTDP(kwargs...)
    isa(kwargs.param, iSTDPParameter) && iSTDP(kwargs...)
    throw("No Synapse type defined")
end


function forward!(c::SpikingSynapse, param::SpikingSynapseParameter)
    @unpack colptr, I, W, fireJ, g = c
    @inbounds for j = 1:(length(colptr)-1)
        if fireJ[j]
            for s = colptr[j]:(colptr[j+1]-1)
                g[I[s]] += W[s]
            end
        end
    end
end

function plasticity!(
    c::SpikingSynapse,
    param::no_STDPParameter,
    dt::Float32,
    t::Float32,
)
end

function plasticity!(
    c::SpikingSynapse,
    param::vSTDPParameter,
    dt::Float32,
    t::Float32,
)
    @unpack rowptr, colptr, I, J, index, W, u, v, x, vpost, fireJ, g = c
    @unpack A_LTD, A_LTP, θ_LTD, θ_LTP, τu, τv, τx, Wmax, Wmin = param
    R(x) = x < 0 ? 0 : x

    @inbounds for j = 1:(length(colptr)-1) # Iterate over all columns, j: presynaptic neuron
        x[j] += dt * (-x[j] + fireJ[j]) / τx # presynaptic neuron

        for s = colptr[j]:(colptr[j+1]-1) # Iterate over all values in column j, s: postsynaptic neuron connected to j
            u[s] += dt * (-u[s] + vpost[s]) / τu # postsynaptic neuron
            v[s] += dt * (-v[s] + vpost[s]) / τv # postsynaptic neuron
            
            # W[s] += - A_LTD * fireJ[j] * R(u[s] - θ_LTD)
            # + A_LTP * x[j] * R(vpost[s] - θ_LTP) * R(v[s] - θ_LTD)
            W[s] = clamp(W[s] - A_LTD * fireJ[j] * R(u[s] - θ_LTD)
            + A_LTP * x[j] * R(vpost[s] - θ_LTP) * R(v[s] - θ_LTD), Wmin, Wmax)
        end
    end

    @inbounds for i = 1:(length(rowptr)-1) # Iterate over all rows, i: postsynaptic neuron
        # Scaling
        _pre = rowptr[i]:rowptr[i+1]-1 # all presynaptic neurons connected to neuron i
        W0 = sum(W[_pre])/length(W[_pre])
        for st = rowptr[i]:(rowptr[i+1]-1) # presynaptic indeces to which neuron i connects
            s = index[st]
            W[s] -= W0
        end
    end
end

function plasticity!(
    c::SpikingSynapse,
    param::iSTDPParameter,
    dt::Float32,
    t::Float32,
)
    @unpack rowptr, colptr, I, J, index, W, yᴱ, yᴵ, fireI, fireJ, g = c
    @unpack η, r₀, τy, Wmax, Wmin = param

    @inbounds for j = 1:(length(colptr)-1) # presynaptic indeces j
        yᴱ[j] += dt * (-yᴱ[j] + fireJ[j]) / τy

        if fireJ[j] # presynaptic neuron
            for s = colptr[j]:(colptr[j+1]-1) # postsynaptic indeces to which neuron j connects
                W[s] = clamp(W[s] + η * (yᴱ[j] - 2 * r₀ * τy), Wmin, Wmax)
            end
        end
    end
    @inbounds for i = 1:(length(rowptr)-1) # postsynaptic indeces i
        yᴵ[i] += dt * (-yᴵ[i] + fireI[i]) / τy

        if fireI[i] # postsynaptic neuron
            for st = rowptr[i]:(rowptr[i+1]-1) # presynaptic indeces to which neuron i connects
                s = index[st]
                W[s] = clamp(W[s] + η * yᴵ[i], Wmin, Wmax)
            end
        end
     end
end

function plasticity!(
    c::SpikingSynapse,
    param::STDPParameter,
    dt::Float32,
    t::Float32,
)
    @unpack rowptr, colptr, I, J, index, W, tpre, tpost, Apre, Apost, fireI, fireJ, g = c
    @unpack τpre, τpost, Wmax, ΔApre, ΔApost = param
    @inbounds for j = 1:(length(colptr)-1)
        if fireJ[j]
            for s = colptr[j]:(colptr[j+1]-1)
                Apre[s] *= exp32(-(t - tpre[s]) / τpre)
                Apost[s] *= exp32(-(t - tpost[s]) / τpost)
                Apre[s] += ΔApre
                tpre[s] = t
                W[s] = clamp(W[s] + Apost[s], 0.0f0, Wmax)
            end
        end
    end
    @inbounds for i = 1:(length(rowptr)-1)
        if fireI[i]
            for st = rowptr[i]:(rowptr[i+1]-1)
                s = index[st]
                Apre[s] *= exp32(-(t - tpre[s]) / τpre)
                Apost[s] *= exp32(-(t - tpost[s]) / τpost)
                Apost[s] += ΔApost
                tpost[s] = t
                W[s] = clamp(W[s] + Apre[s], 0.0f0, Wmax)
            end
        end
    end
end

