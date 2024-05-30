@snn_kw struct SpikingSynapseTripod{MFT=Matrix{Float32}, VIT=Vector{Int32},VFT=Vector{Float32},VBT=Vector{Bool}}
	param::Synapse
    rowptr::VIT # row pointer of sparse W
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    J::VIT      # presynaptic index of W
    index::VIT  # index mapping: W[index[i]] = Wt[i], Wt = sparse(dense(W)')

    W::VFT  # synaptic weight
    v_post::VFT
    g::MFT  # rise conductance
	# target receptors specification
    receptors::Vector{Tuple{Int64, AbstractReceptor}}
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
    end
    rowptr, colptr, I, J, index, W = dsparse(w)
    fireI, fireJ = post.fire, pre.fire
	# Get the target conductance (pointer to array)
    g = getfield(post, Symbol("h_$target"))
    v_post = getfield(post, Symbol("v_$target"))

	if target =="s"
		param = soma_syn
		if type =="exc"
			receptors = [(1, soma_syn.AMPA)]
		elseif type =="inh"
			receptors = [(2, soma_syn.GABAa)]
		end
	else
		param = dend_syn
		if type =="exc"
			receptors = [(1, dend_syn.AMPA),(2, dend_syn.NMDA)]
		elseif type =="inh"
			receptors = [(3, dend_syn.GABAa),(4, dend_syn.GABAb)]
		end
	end
    SpikingSynapseTripod(;@symdict(rowptr, colptr, I, J, index, W,  g,  receptors, v_post, fireI, fireJ, param)..., kwargs...)
end

function forward!(c::SpikingSynapseTripod, param::Synapse)
    @unpack colptr, I, W, fireJ,  g, receptors = c
    #update synapse
    for (n,rec) in receptors
        @inbounds for j in 1:(length(colptr) - 1)
            if fireJ[j]
                for s in colptr[j]:(colptr[j+1] - 1)
                    g[I[s],n] += W[s]*rec.α
                end
            end
        end
	end
end

function plasticity!(c::SpikingSynapse, param::SpikingSynapseParameter, dt::Float32, t::Float32)
    @unpack rowptr, colptr, I, J, index, W, tpre, tpost, Apre, Apost, fireI, fireJ, g = c
    @unpack τpre, τpost, Wmax, ΔApre, ΔApost = param
    @inbounds for j in 1:(length(colptr) - 1)
        if fireJ[j]
            for s in colptr[j]:(colptr[j+1] - 1)
                Apre[s] *= exp32(- (t - tpre[s]) / τpre)
                Apost[s] *= exp32(- (t - tpost[s]) / τpost)
                Apre[s] += ΔApre
                tpre[s] = t
                W[s] = clamp(W[s] + Apost[s], 0f0, Wmax)
            end
        end
    end
    @inbounds for i in 1:(length(rowptr) - 1)
        if fireI[i]
            for st in rowptr[i]:(rowptr[i+1] - 1)
                s = index[st]
                Apre[s] *= exp32(- (t - tpre[s]) / τpre)
                Apost[s] *= exp32(- (t - tpost[s]) / τpost)
                Apost[s] += ΔApost
                tpost[s] = t
                W[s] = clamp(W[s] + Apre[s], 0f0, Wmax)
            end
        end
    end
end