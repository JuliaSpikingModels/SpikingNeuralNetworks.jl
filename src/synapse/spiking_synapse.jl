abstract type AbstractSpikingSynapse <: AbstractSparseSynapse end

@snn_kw mutable struct SpikingSynapse{
    VIT = Vector{Int32}, 
    VFT = Vector{Float32}, 
    VBT = Vector{Bool},
    } <: AbstractSpikingSynapse
    param::SpikingSynapseParameter = no_STDPParameter()
    plasticity::PlasticityVariables = no_PlasticityVariables()
    rowptr::VIT # row pointer of sparse W
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    J::VIT      # presynaptic index of W
    index::VIT  # index mapping: W[index[i]] = Wt[i], Wt = sparse(dense(W)')
    W::VFT  # synaptic weight
    fireI::VBT # postsynaptic firing
    fireJ::VBT # presynaptic firing
    v_post::VFT 
    g::VFT  # rise conductance
    αs::VFT = []
    receptors::VIT = []
    records::Dict = Dict()
end

function SpikingSynapse(pre, post, sym; σ = 0.0, p = 0.0, w = nothing, kwargs...)
    if isnothing(w)
        w = σ * sprand(post.N, pre.N, p) # Construct a random sparse matrix with dimensions post.N x pre.N and density p
    else
        w = sparse(w)
    end
    w[diagind(w)] .= 0 
    @assert size(w) == (post.N, pre.N)
    
    rowptr, colptr, I, J, index, W = dsparse(w)
    fireI, fireJ, v_post = post.fire, pre.fire, post.v
    g = getfield(post, sym)
    
    # set the paramter for the synaptic plasticity
    param = haskey(kwargs, :param) ? kwargs[:param] : no_STDPParameter()
    plasticity = get_variables(param, pre.N, post.N)
    
    # Construct the SpikingSynapse instance
    return SpikingSynapse(; param=param, plasticity=plasticity, g=g, @symdict(rowptr, colptr, I, J, index, W, fireI, fireJ, v_post)..., kwargs...)
end

function TripodSynapse(
    pre,
    post,
    target::Symbol,
    type::Symbol;
    w = nothing,
    σ = 0.0,
    p = 0.0,
    kwargs...,
)
    # Create the sparse matrix
    if w === nothing
        w = σ * sprand(post.N, pre.N, p)
    else
        w = sparse(w)
    end
    #no autapsis
    w[diagind(w)] .= 0
    rowptr, colptr, I, J, index, W = dsparse(w)
    fireI, fireJ = post.fire, pre.fire
    v_post = getfield(post, Symbol("v_$target"))

    # Get the parameters for post-synaptic cell
    @unpack dend_syn = post
    @unpack soma_syn = post
    if Symbol(type) == :exc
        receptors = target == "s" ? [1] : [1, 2]
        g = view(getfield(post, Symbol("h_$target")), :, receptors) 
        αs = [post.dend_syn[i].α for i in eachindex(receptors)]
    elseif Symbol(type) == :inh
        receptors = target == "s" ? [2] : [3, 4]
        g = view(getfield(post, Symbol("h_$target")), :, receptors)
        αs = [post.dend_syn[i].α for i in eachindex(receptors)]
    else
        throw(ErrorException("Synapse type: $type not implemented"))
    end

    param = haskey(kwargs, :param) ? kwargs[:param] : no_STDPParameter()
    plasticity = get_variables(param, pre.N, post.N)

    SpikingSynapse(;
    plasticity= plasticity,
        @symdict(
            rowptr,
            colptr,
            I,
            J,
            index,
            receptors,
            W,
            g,
            αs,
            v_post,
            fireI,
            fireJ
        )...,
        kwargs...,
    )
end

function forward!(c::SpikingSynapse, param::SpikingSynapseParameter)
    @unpack colptr, I, W, fireJ, g, αs = c
    if isempty(αs)
        @inbounds for j ∈ eachindex(fireJ) # loop on presynaptic neurons
            if fireJ[j] # presynaptic fire
                @inbounds @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                    g[I[s]] += W[s] 
                end
            end
        end
    else
        @inbounds for j ∈ eachindex(fireJ) # loop on presynaptic neurons
            if fireJ[j] # presynaptic fire
                @inbounds @fastmath for a in eachindex(αs)
                    @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                        g[I[s], a] += W[s] * αs[a]
                    end
                end
            end
        end
    end
end
