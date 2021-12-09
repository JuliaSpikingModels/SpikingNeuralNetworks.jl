C     = 281pF        #(pF)
gL    = 40nS         #(nS) leak conductance #BretteGerstner2005 says 30 nS

@snn_kw struct AdExParameter{FT=Float32} <: AbstractIFParameter
	τm::FT = C/gL
    τe::FT = 5ms
    τi::FT = 10ms
    Vt::FT = -50mV
    Vr::FT = -70.6mV
    El::FT = Vr
	R::FT  = nS/gL
	ΔT::FT = 2mV
	τw::FT = 144ms
	a::FT = 4nS
	b::FT = 80.5nA
end

@snn_kw mutable struct AdEx{VFT=Vector{Float32},VBT=Vector{Bool}} <: AbstractIF
    param::AdExParameter = AdExParameter()
    N::Int32 = 100
    v::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    w::VFT = zeros(N)
    ge::VFT = zeros(N)
    gi::VFT = zeros(N)
    fire::VBT = zeros(Bool, N)
    θ::VFT = ones(N)*param.Vt
    I::VFT = zeros(N)
    records::Dict = Dict()
end

"""
    [Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
"""

function integrate!(p::AdEx, param::AdExParameter, dt::Float32)
    @unpack N, v, w, ge, gi, fire, I, θ = p
    @unpack τm, τe, τi, Vt, Vr, El, R, ΔT, τw, a, b = param
    @inbounds for i = 1:N
        v[i] += dt * (ge[i] + gi[i] - (v[i] - El) + ΔT*exp((v[i]-θ[i])/ΔT) -R*w[i] + I[i]) / τm
		w[i] += dt * (a*(v[i]-El) -w[i] )/τw
        ge[i] += dt * -ge[i] / τe
        gi[i] += dt * -gi[i] / τi
    end
    @inbounds for i = 1:N
        fire[i] = v[i] > 0.
        v[i] = ifelse(fire[i], Vr, v[i])
		w[i] = ifelse(fire[i], w[i]+b*τw, w[i])
    end
end


W = 10.
ga = zeros(1000)
gb = zeros(1000)
gb1 = zeros(1000)

ga[1] = W
gb[1] = 1.

dt = 0.01
for x in 2:1:1000
	ga[x] = ga[x-1]*exp(-dt)
	gb[x] = gb[x-1]*exp(-dt)
	gb1[x] = W*gb[x]
end


plot(ga)
plot!(gb1)
