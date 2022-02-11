struct FLSynapseParameter
end

@snn_kw mutable struct FLSynapse{MFT=Matrix{Float32},VFT=Vector{Float32},FT=Float32}
    param::FLSynapseParameter = FLSynapseParameter()
    W::MFT  # synaptic weight
    rI::VFT # postsynaptic rate
    rJ::VFT # presynaptic rate
    g::VFT  # postsynaptic conductance
    P::MFT  # <rᵢrⱼ>⁻¹
    q::VFT  # P * r
    u::VFT # force weight
    w::VFT # output weight
    f::FT = 0 # postsynaptic traget
    z::FT = 0.5randn()  # output z ≈ f
    records::Dict = Dict()
end

"""
[Force Learning Full Synapse](http://www.theswartzfoundation.org/docs/Sussillo-Abbott-Coherent-Patterns-August-2009.pdf)
"""
FLSynapse

function FLSynapse(pre, post; σ = 1.5, p = 0.0, α = 1, kwargs...)
    rI, rJ, g = post.r, pre.r, post.g
    W = σ * 1 / √pre.N * randn(post.N, pre.N) # normalized recurrent weight
    w = 1 / √post.N * (2rand(post.N) .- 1) # initial output weight
    u = 2rand(post.N) .- 1 # initial force weight
    P = α * I(post.N) # initial inverse of   = <rr'>
    q = zeros(post.N)
    FLSynapse(;@symdict(W, rI, rJ, g, P, q, u, w)..., kwargs...)
end

function forward!(c::FLSynapse, param::FLSynapseParameter)
    @unpack W, rI, rJ, g, P, q, u, w, z = c
    c.z = dot(w, rI)
    # @show z
    mul!(q, P, rJ)
    mul!(g, W, rJ)
    axpy!(c.z, u, g)
end

function plasticity!(c::FLSynapse, param::FLSynapseParameter, dt::Float32, t::Float32)
    @unpack rI, P, q, w, f, z = c
    #
    # 1 / (1 + rI P rJ) it is not in the paper
    C = 1 / (1 + dot(q, rI))

    # e- = C * (f-z)
    # w = w + e⁻ * q
    axpy!(C * (f - z), q, w)

    # ger(a,x,y,A = a*x*y' + A.
    # P = P + - C* q * q' = P - C * P rJ * rI * P
    BLAS.ger!(-C, q, q, P)
end
