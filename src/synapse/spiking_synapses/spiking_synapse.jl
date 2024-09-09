abstract type SpikingSynapse end

struct no_STDPParameter <: SpikingSynapseParameter end

@snn_kw mutable struct STDP{VIT = Vector{Int32},VFT = Vector{Float32},VBT = Vector{Bool}} <:
                       SpikingSynapse
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
} <: SpikingSynapse
    param::vSTDPParameter = vSTDPParameter()
    rowptr::VIT # row pointer of sparse W
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    J::VIT      # presynaptic index of W
    index::VIT  # index mapping: W[index[i]] = Wt[i], Wt = sparse(dense(W)')
    W::VFT  # synaptic weight
    W0::VFT = deepcopy(W)
    u::VFT = zeros(length(colptr) - 1) # presynaptic spiking time
    v::VFT = zeros(length(colptr) - 1) # postsynaptic spiking time
    x::VFT = zeros(length(colptr) - 1) # postsynaptic spiking time
    vpost::VFT = zeros(length(colptr) - 1)
    fireJ::VBT # presynaptic firing
    g::VFT # postsynaptic conductance
    records::Dict = Dict()
end

@snn_kw mutable struct iSTDP{
    VIT = Vector{Int32},
    VFT = Vector{Float32},
    VBT = Vector{Bool},
} <: SpikingSynapse
    param::iSTDPParameter = iSTDPParameterRate()
    rowptr::VIT # row pointer of sparse W
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    J::VIT      # presynaptic index of W
    index::VIT  # index mapping: W[index[i]] = Wt[i], Wt = sparse(dense(W)')
    W::VFT  # synaptic weight
    tpost::VFT = zeros(length(rowptr) - 1) # postsynaptic spiking time 
    tpre::VFT = zeros(length(colptr) - 1) # presynaptic spiking time
    fireI::VBT
    fireJ::VBT # presynaptic firing
    g::VFT # postsynaptic conductance
    records::Dict = Dict()
end

@snn_kw mutable struct no_STDP{
    VIT = Vector{Int32},
    VFT = Vector{Float32},
    VBT = Vector{Bool},
} <: SpikingSynapse
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
# function forward!(c::InhSynapseTripod, param::SpikingSynapseParameter)
#     @unpack colptr, I, W, fireJ, g, αs = c
#     @inbounds @simd for j in 1:(length(colptr)-1)
#         if fireJ[j]
#             for s in colptr[j]:(colptr[j+1]-1)
#                 g[I[s], :] .+= W[s] .* αs
#             end
#         end
#     end
# end

"""
[Spking Synapse](https://brian2.readthedocs.io/en/2.0b4/resources/tutorials/2-intro-to-brian-synapses.html)
"""
SpikingSynapse

function SpikingSynapse(pre, post, sym; σ = 0.0, p = 0.0, kwargs...)
    w = σ * sprand(post.N, pre.N, p) # Construct a random sparse vector with length post.N, pre.N and density p
    w[findall(w .> 0)] .= σ # make all the weights the same
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
        return no_STDP(; @symdict(rowptr, colptr, I, J, index, W, g, fireJ)..., kwargs...)
    end
end

function SpikingSynapse(; kwargs...)
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

export SpikingSynapse
