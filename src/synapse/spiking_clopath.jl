@snn_kw struct SpikingSynapseClopath{MFT=Matrix{Float32}, VIT=Vector{Int32},VFT=Vector{Float32},VBT=Vector{Bool}}
    param::AbstractSynapse
    rowptr::VIT # row pointer of sparse W
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    J::VIT      # presynaptic index of W
    index::VIT  # index mapping: W[index[i]] = Wt[i], Wt = sparse(dense(W)')
    W::VFT  # synaptic weight
    v_post::VFT
    g::MFT  # rise conductance
	c_index::VIT # index mapping for receptor conductance in the cell
    receptors::Vector{Symbol}
    tpre::VFT = zero(W)  # presynaptic spiking time
    tpost::VFT = zero(W) # postsynaptic spiking time
    Apre::VFT = zero(W)  # presynaptic trace
    Apost::VFT = zero(W) # postsynaptic trace
    fireI::VBT # postsynaptic firing
    fireJ::VBT # presynaptic firing
    records::Dict = Dict()
end

"""
[Spking Synapse](https://brian2.readthedocs.io/en/2.0b4/resources/tutorials/2-intro-to-brian-synapses.html)
"""
SpikingSynapseClopath

function SpikingSynapseClopath(pre, post, target::String, receptors::Tuple{Vector{Symbol}, Vector{Int64}}; σ = -1.0, p = 0.0, kwargs...)
    w = σ * sprand(post.N, pre.N, p)
    rowptr, colptr, I, J, index, W = dsparse(w)
    fireI, fireJ = post.fire, pre.fire
    g = getfield(post, Symbol("h_$target"))
    v_post = getfield(post, Symbol("v_$target"))
	c_index = receptors[2]
	receptors = receptors[1]
    SpikingSynapseClopath(;@symdict(rowptr, colptr, I, J, index, W, fireI, fireJ, g,  receptors, c_index, v_post)..., kwargs...)
end

function forward!(c::SpikingSynapseClopath, param::AbstractSynapse)
    @unpack colptr, I, W, fireJ,  g, receptors, c_index = c
    #update synapse
    for (n,rec) in enumerate([getfield(param, r) for r in receptors ])
		n = c_index[n]
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
