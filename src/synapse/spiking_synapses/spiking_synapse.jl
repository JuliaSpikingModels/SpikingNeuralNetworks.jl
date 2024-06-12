
@snn_kw struct SynapseTripod{MFT=Matrix{Float32}, VIT=Vector{Int32},VFT=Vector{Float32},VBT=Vector{Bool}, S=Synapse, VRT= Vector{ Float32}}
	param::SpikingSynapseParameter=no_STDPParameter()
    rowptr::VIT # row pointer of sparse W
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    J::VIT      # presynaptic index of W
    index::VIT  # index mapping: W[index[i]] = Wt[i], Wt = sparse(dense(W)')

    W::VFT  # synaptic weight
    v_post::VFT
    g::MFT  # rise conductance
	# target receptors specific alpha
    αs::VRT
    receptors::VIT
	# Plasticity variables
    tpre::VFT = zero(W)  # presynaptic spiking time
    tpost::VFT = zero(W) # postsynaptic spiking time
    Apre::VFT = zero(W)  # presynaptic trace
    Apost::VFT = zero(W) # postsynaptic trace
    fireI::VBT # postsynaptic firing
    fireJ::VBT # presynaptic firing
    records::Dict = Dict()
end

"""
[Spiking Synapse](https://brian2.readthedocs.io/en/2.0b4/resources/tutorials/2-intro-to-brian-synapses.html)
"""
function SynapseTripod(pre, post, target::String, type::String; w = nothing, σ = -1.0, p = 0.0, kwargs...)
	# Get the parameters for post-synaptic cell
	@unpack dend_syn = post
	@unpack soma_syn = post
	# Create the sparse matrix
	if w === nothing
        w = σ * sprand(post.N, pre.N, p)
    else
        w = sparse(w)
    end
    rowptr, colptr, I, J, index, W = dsparse(w)
    fireI, fireJ = post.fire, pre.fire

	# Get the target conductance (pointer to array)
    g = getfield(post, Symbol("h_$target"))
    v_post = getfield(post, Symbol("v_$target"))
	if target =="s"
		if type =="exc"
			receptors = [1]
		elseif type =="inh"
			receptors = [2]
		end
        αs = [post.soma_syn[i].α for i in eachindex(receptors)]
	else
		# param = deepcopy(dend_syn)
		if type =="exc"
			receptors = [1,2]
		elseif type =="inh"
			receptors = [3,4]
		end
        αs = [post.dend_syn[i].α for i in eachindex(receptors)]
	end
    SynapseTripod(;@symdict(rowptr, colptr, I, J, index, receptors, W,  g,  αs, v_post, fireI, fireJ)..., kwargs...)
end

function forward!(c::SynapseTripod, param::SpikingSynapseParameter)
    @unpack colptr, I, W, fireJ,  g = c
    @unpack receptors, αs = c
    @fastmath for r in eachindex(receptors) 
         @inbounds @simd for j in 1:(length(colptr) - 1)
            if fireJ[j]
                for s in colptr[j]:(colptr[j+1] - 1)
                    g[I[s],receptors[r]] +=  W[s]*αs[r]
                end
            end
        end
	end
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

export SynapseTripod, SpikingSynapse
