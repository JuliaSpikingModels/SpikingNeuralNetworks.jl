struct FLSparseSpikingSynapseParameter
end

@snn_kw mutable struct FLSparseSpikingSynapse{VFT=Vector{Float32},FT=Float32}
    param::FLSparseSpikingSynapseParameter = FLSparseSynapseParameter()
    colptr::Vector{Int32} # column pointer of sparse W
    I::Vector{Int32}      # postsynaptic index of W
    W::VFT  # synaptic weight
    rI::VFT # postsynaptic rate
    rJ::VFT # presynaptic rate
    hJ::VFT  # pre-synptic double exponential
    g::VFT  # postsynaptic conductance
    P::VFT  # <rᵢrⱼ>⁻¹
    q::VFT  # P * r
    u::VFT # force weight
    w::VFT # output weight
    f::FT = 0 # postsynaptic traget
    z::FT = 0.5randn()  # output z ≈    f
    records::Dict = Dict()
end

"""
[Force Learning Sparse Spiking Synapse](Nicola and Clopath)
"""
FLSparseSynapse

function FLSparseSpikingSynapse(pre, post; σ = 1.5, p = 0.0, α = 1, kwargs...)
    w = σ * 1 / √(p * pre.N) * sprandn(post.N, pre.N, p)
    rowptr, colptr, I, J, index, W = dsparse(w)
    rI, rJ, g = post.r, pre.r, post.g
    P = α .* (I .== J)
    q = zeros(post.N)
    u = 2rand(post.N) - 1
    w = 1 / √post.N * (2rand(post.N) - 1)
    FLSparseSynapse(;@symdict(colptr, I, W, rI, rJ, g, P, q, u, w)..., kwargs...)
end


function forward!(c::FLSparseSpikingSynapse, param::FLSparseSpikingSynapseParameter)
    @unpack W, rI, rJ, g, P, q, u, w, f, z = c
    @unpack τr, τd = param
    c.z = dot(w, rI)
    g .= c.z .* u
    fill!(q, zero(Float32))
    # the sparse matrix is CSC.
    # each pre-synaptic neuron correspond to a column/
    # each pre-synaptic conductance is on J too
    @inbounds for j in 1:(length(colptr) - 1)
        if fireJ[j]
            hJ[i] += 1/(τr*τd)
        end
        ## presynaptic rate decay
        hJ[j] -= hJ[j]/τr
        rJ[j] -= rJ[j]/τd + hJ[j]

        rJj = rJ[j]
        for s = colptr[j]:(colptr[j+1] - 1)
            i = I[s]
            q[i] += P[s] * rJj
            g[i] += W[s] * rJj
        end
    end
end

function plasticity!(c::FLSparseSpikingSynapse, param::FLSparseSpikingSynapseParameter, dt::Float32, t::Float32)
    @unpack rI, P, q, w, f, z = c

    # 1 / (1 + rI P rJ) it is not in the paper
    C = 1 / (1 + dot(q, rI))

    # e- = C * (f-z)
    # w = w + e⁻ * q
    BLAS.axpy!(C * (f - z), q, w)

    # P = P + - C* q * q' = P - C * P rJ * rI * P
    # P[s] is the running estimate of the inverse correlation matrix of firing rates
    @inbounds for j in 1:(length(colptr) - 1)
        for s in colptr[j]:(colptr[j+1] - 1)
            P[s] += -C * q[I[s]] * q[j]
        end
    end
end
