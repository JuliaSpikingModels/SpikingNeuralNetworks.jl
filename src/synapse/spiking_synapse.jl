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
    Wmax::FT = 21.4pF
    Wmin::FT = 1.78pF
end

@snn_kw struct iSTDPParameter{FT = Float32} <: SpikingSynapseParameter
    η::FT = 1pA
    r₀::FT = 3Hz
    τy::FT = 20ms
    Wmax::FT = 243pF
    Wmin::FT = 48.7pF
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
    W0::VFT = deepcopy(W)
    u::VFT = zeros(length(colptr)) # presynaptic spiking time
    v::VFT = zeros(length(colptr)) # postsynaptic spiking time
    x::VFT = zeros(length(colptr)) # postsynaptic spiking time
    vpost::VFT = zeros(length(colptr))
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
    yᴱ::VFT = zeros(length(colptr)) # presynaptic spiking time
    yᴵ::VFT = zeros(length(rowptr)) # presynaptic spiking time
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
    w[findall(w.>0)] .= σ # make all the weights the same
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
    @inbounds for j = 1:(length(colptr)-1) # Iterate over all columns, j: presynaptic neuron
        if fireJ[j] # if presynaptic neuron fired, then
            for s = colptr[j]:(colptr[j+1]-1) # Iterate over all values in column j, s: postsynaptic neuron connected to j
                g[I[s]] += W[s] # update the conductance of the postsynaptic neuron s
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
    @unpack rowptr, colptr, I, J, index, W, W0, u, v, x, vpost, fireJ, g = c
    @unpack A_LTD, A_LTP, θ_LTD, θ_LTP, τu, τv, τx, Wmax, Wmin = param
    R(x) = x < 0 ? eltype(x)(0) : x

    # @inbounds @fastmath
    for j = 1:(length(colptr)-1) # Iterate over all columns, j: presynaptic neuron
        x[j] += dt * (-x[j] + fireJ[j]) / τx # presynaptic neuron

        for s = colptr[j]:(colptr[j+1]-1) # Iterate over all values in column j, s: postsynaptic neuron connected to j
            u[I[s]] += dt * (-u[I[s]] + vpost[I[s]]) / τu # postsynaptic neuron
            v[I[s]] += dt * (-v[I[s]] + vpost[I[s]]) / τv # postsynaptic neuron
            
            W[s] += dt * (- A_LTD * fireJ[j] * R(u[I[s]] - θ_LTD)
            + A_LTP * x[j] * R(vpost[I[s]] - θ_LTP) * R(v[I[s]] - θ_LTD))
        end
    end

    if (t % 20) == 0 
        # @inbounds @fastmath
        for i = 1:(length(rowptr)-1) # Iterate over all rows, i: postsynaptic neuron
            _pretopost = index[rowptr[i]:rowptr[i+1]-1] # all presynaptic neurons connected to neuron i
            W[_pretopost] .+= sum(W0[_pretopost].-W[_pretopost])./length(_pretopost) 
            # W[_pretopost] .+= mean(W0[_pretopost].-W[_pretopost]) 
            # @debug i, sum(W[_pretopost]), sum(W0[_pretopost])
        end
    end

    W[:] = clamp.(W[:], Wmin, Wmax)
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
                W[s] = W[s] + η * (yᴱ[j] - 2 * r₀ * τy)
            end
        end
    end
    @inbounds for i = 1:(length(rowptr)-1) # postsynaptic indeces i
        yᴵ[i] += dt * (-yᴵ[i] + fireI[i]) / τy

        if fireI[i] # postsynaptic neuron
            for st = rowptr[i]:(rowptr[i+1]-1) # presynaptic indeces to which neuron i connects
                s = index[st]
                W[s] = W[s] + η * yᴵ[i]
            end
        end
    end
    
    W[:] = clamp.(W[:], Wmin, Wmax)
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

