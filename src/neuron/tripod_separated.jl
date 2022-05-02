@snn_kw struct AdExTripod{FT=Float32} <: AbstractIFParameter
    #Membrane parameters
    C::FT  = 281pF           # (pF) membrane timescale
    gl::FT = 40nS                # (nS) gl is the leaking conductance,opposite of Rm
    R::FT  = nS/gl*GΩ               # (GΩ) total membrane resistance
    τm::FT = C/gl                # (ms) C / gl
    Er::FT = -70.6mV          # (mV) resting potential
    # AdEx model
    Vr::FT = -70.6mV     # (mV) Reset potential of membrane
    Vt::FT   = -50mV          # (mv) Rheobase threshold
    ΔT::FT  = 2mV            # (mV) Threshold sharpness
    # Adaptation parameters
    τw::FT = 144ms          #ms adaptation current relaxing time
    a::FT  = 4nS            #nS adaptation current to membrane
    b::FT  = 80.5pA         #pA adaptation current increase due to spike
	# After spike timescales and membrane
	AP_membrane::Float32= 10f0mV
	BAP::Float32 = 1f0mV
	up::Int64 = 1ms
	idle::Int64 = 2ms
	dends::Int64 = 2
end

@with_kw struct PostSpike
      A::Float32
      τA::Float32
end
postspike = PostSpike(A=10, τA= 30ms)


@snn_kw struct DendriteParam{VFT=Vector{Float32}}
	N::Int32 = 100
    Er::VFT  = ones(N)*-70.6mV             # (mV) resting potential
    C::VFT   = ones(N)*10pF                 # (1/pF) membrane timescale
	g_ax::VFT= ones(N)*10nS				# (nS) axial conductance
	g_m::VFT = ones(N)*1nS
	l::VFT	 = ones(N)*150um				# μm distance from next compartment
	d::VFT	 = ones(N)*4um				# μm dendrite diameter
end

@snn_kw struct Dendrite{VIT=Vector{Int32}, VFT=Vector{Float32},VBT=Vector{Bool}}
	param::DendriteParam = DendriteParam()
	N::Int32 = 100
    v::VFT = param.Er[1] .+ rand(N) .* (-20mV - param.Er[1])
    I_syn::VFT = zeros(N)
    records::Dict = Dict()
end


@snn_kw struct Tripod{VIT=Vector{Int32}, VFT=Vector{Float32},VBT=Vector{Bool}} <: AbstractIF
    param::AdExTripod = AdExTripod()
    N::Int32 = 100
    v::VFT  = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    w::VFT = zeros(N)
    I_syn::VFT  = zeros(N)
    fire::VBT = zeros(Bool, N)
	after_spike::VFT = zeros(N)
    θ::VFT = ones(N)*param.Vt
	dends::Vector{Dendrite}
	postspike::PostSpike = postspike
    records::Dict = Dict()
end

"""
    Tripod neuron
"""

function integrate!(p::Tripod, param::AdExTripod, dt::Float32)
    @unpack N, v, w, I_syn, dends, fire, θ, after_spike, postspike = p
	@unpack Er, up, idle, BAP, AP_membrane, Vr, Vt, τw, a, b = param

    @inbounds for i = 1:N
		if after_spike[i] >idle/dt
	        v[i] = BAP
			## backpropagation effect
			for x in 1:param.dends
				dends[x].v[i] += dt*(BAP - dends[x].v[i])* dends[x].param.g_ax[i]/dends[x].param.C[i]
			end

		elseif after_spike[i] > 0
			vs[i] = Vr
			## apply currents
			for x in 1:param.dends
				dends[x].v[i] += dt*(Vr - dends[x].v[i])* dends[x].param.g_ax[i]/dends[x].param.C[i]
			end
		else
			c = 0.f0
			for x in 1:param.dends
				_c = ( v[i]- dends[x].v[i])* dends[x].param.g_ax[i]
				Δdends =dt* ((-dends[x].v[i] + dends[x].param.Er[i])*dends[x].param.g_m[i] - _c -dends[x].Isyn[i])/dends[x].param.C[i]
				c += _c
				dends[x].v[i] += Δdends
			end
			Δvs =dt* ΔvAdEx(v[i], w[i], θ[i], c, I_syn[i], param)
			Δws =dt* ΔwAdEx(v[i], w[i], param)
			v[i] += Δvs
			w[i] += Δws
		end
    end
    @inbounds for i = 1:N
		θ[i] -= dt* (θ[i] - Vt ) / postspike.τA
		after_spike[i] -= dt
		if after_spike[i] < 0.f0
	        fire[i] = vs[i] > θ[i] + 10.
			if fire[i]
				θ[i] += postspike.A
		        vs[i] = AP_membrane
				ws[i] += b*τw
				after_spike[i] = (up+idle)/dt
			end
		end
    end
end

@inline @fastmath function ΔvDend(v::Float32, axial::Float32, synaptic::Float32, pm::Dendrite)::Float32
	 return
end

@inline @fastmath function ΔvAdEx(v::Float32, w::Float32, θ::Float32, axial::Float32, synaptic::Float32, AdEx::AdExTripod)::Float32
	return ( AdEx.gl *( -v +AdEx.Er +  AdEx.ΔT * exp32(1/AdEx.ΔT*(v-θ))) - w  - synaptic - axial)/AdEx.C ## external currents
end

@inline @fastmath function ΔwAdEx(v::Float32, w::Float32,AdEx::AdExTripod)::Float32
	return ( AdEx.a *(v - AdEx.Er) - w)/ AdEx.τw
end
