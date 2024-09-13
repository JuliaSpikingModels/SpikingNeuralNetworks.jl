abstract type SynapseTripod <: SpikingSynapse end

g_type = SubArray{
	Float32,
	2,
	Matrix{Float32},
	Tuple{Base.Slice{Base.OneTo{Int64}}, Vector{Int64}},
	false,
}

## Excitatory and inhibitory synapses for the Tripod
@snn_kw struct ExcSynapseTripod{
	MFT = g_type,
	FT = Float32,
	VIT = Vector{Int32},
	VFT = Vector{Float32},
	VBT = Vector{Bool},
	VRT = Vector{Float32},
} <: SynapseTripod
	param::SpikingSynapseParameter = no_STDPParameter()
	t::VIT = [0, 1]
	rowptr::VIT # row pointer of sparse W
	colptr::VIT # column pointer of sparse W
	I::VIT      # postsynaptic index of W
	J::VIT      # presynaptic index of W
	index::VIT  # index mapping: W[index[i]] = Wt[i], Wt = sparse(dense(W)')
	W::VFT  # synaptic weight
	v_post::VFT                              #! This will be that target
	g::MFT  # rise conductance               #! This will be the rise conductance
	# target receptors specific alpha
	αs::VRT
	receptors::VIT

	# Plasticity variables
	fireI::VBT # postsynaptic firing
	fireJ::VBT # presynaptic firing
	u::VFT = zeros(length(rowptr) - 1) # postsynaptic spiking time 
	v::VFT = zeros(length(rowptr) - 1) # postsynaptic spiking time 
	x::VFT = zeros(length(colptr) - 1) # presynaptic spiking time
	records::Dict = Dict()
end

@snn_kw struct InhSynapseTripod{
	MFT = g_type,
	FT = Float32,
	VIT = Vector{Int32},
	VFT = Vector{Float32},
	VBT = Vector{Bool},
	VRT = Vector{Float32},
	SPT <: SpikingSynapseParameter,
} <: SynapseTripod
	param::SpikingSynapseParameter = no_STDPParameter()
	t::VIT = [0, 1]
	rowptr::VIT # row pointer of sparse W
	colptr::VIT # column pointer of sparse W
	I::VIT      # postsynaptic index of W
	J::VIT      # presynaptic index of W
	index::VIT  # index mapping: W[index[i]] = Wt[i], Wt = sparse(dense(W)')
	W::VFT  # synaptic weight
	v_post::VFT                              #! This will be that target
	g::MFT  # rise conductance               #! This will be the synaptic conductance
	# target receptors specific alpha
	αs::VRT
	receptors::VIT

	# Plasticity variables
	# tpre::VFT = zero(W)  # presynaptic spiking time
	# tpost::VFT = zero(W) # postsynaptic spiking time
	tpost::VFT = zeros(length(rowptr) - 1) # postsynaptic spiking time 
	tpre::VFT = zeros(length(colptr) - 1) # presynaptic spiking time
	fireI::VBT # postsynaptic firing
	fireJ::VBT # presynaptic firing
	records::Dict = Dict()
end

"""
[Spiking Synapse](https://brian2.readthedocs.io/en/2.0b4/resources/tutorials/2-intro-to-brian-synapses.html)
"""
function SynapseTripod(
	pre,
	post,
	target::Union{String, Symbol},
	type::Union{String, Symbol};
	w = nothing,
	σ = 0.0,
	p = 0.0,
	kwargs...,
)
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
	v_post = getfield(post, Symbol("v_$target"))


	if Symbol(type) == :exc
		receptors = target == "s" ? [1] : [1, 2]
		g = view(getfield(post, Symbol("h_$target")), :, receptors) # switch this!
		αs = [post.dend_syn[i].α for i in eachindex(receptors)]
		ExcSynapseTripod(;
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
				fireJ,
			)...,
			kwargs...,
		)
	elseif Symbol(type) == :inh
		receptors = target == "s" ? [2] : [3, 4]
		g = view(getfield(post, Symbol("h_$target")), :, receptors)
		αs = [post.dend_syn[i].α for i in eachindex(receptors)]
		InhSynapseTripod(;
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
	else
		throw(ErrorException("Synapse type: $type not implemented"))
	end
end

function SynapseTwoDend(
	pre,
	post,
	type::Union{String, Symbol};
	w = nothing,
	σ = 0.0,
	p = 0.0,
	kwargs...,
)
	synapses = []
	for (n, d) in enumerate([:d1, :d2])
		w_d = isnothing(w) ? nothing : w[:, n, :]
		push!(
			synapses,
			SNN.SynapseTripod(pre, post, d, type, w = w_d, σ = σ, p = p, kwargs...),
		)
	end
	return synapses
end



function forward!(c::SynapseNormalization, param::NormParam) end

function forward!(c::SynapseTripod, param::SpikingSynapseParameter)
	@unpack colptr, I, W, fireJ, g, αs = c
	@inbounds for j ∈ eachindex(fireJ) # loop on presynaptic neurons
		if fireJ[j] # presynaptic fire
			@simd for a in eachindex(αs)
				for s ∈ colptr[j]:(colptr[j+1]-1)
					g[I[s], a] += W[s] * αs[a]
				end
			end
		end
	end
end


export ExcSynapseTripod, InhSynapseTripod
