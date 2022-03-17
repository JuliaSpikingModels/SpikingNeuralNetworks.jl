# Get synapses from SNNUtils
dend_syn= Synapse(EyalGluDend, MilesGabaDend)
soma_syn= Synapse(GluSoma, MilesGabaSoma)

@snn_kw struct AdExTripod{FT=Float32} <: AbstractIFParameter
    #Membrane parameters
    C::FT  = 281pF           # (pF) membrane timescale
    gl::FT = 40nS                # (nS) gl is the leaking conductance,opposite of Rm
    R::FT  = nS/gl*GΩ               # (GΩ) total membrane resistance
    τm::FT = C/gl                # (ms) C / gl
    Er::FT = -70.6mV          # (mV) resting potential
    # AdEx model
    Vr::FT = -70.6mV     # (mV) Reset potential of membrane
    Vt::FT   = -50.4mV          # (mv) Rheobase threshold
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
end


@with_kw struct PostSpike
      A::Float32
      τA::Float32
end


@snn_kw struct Dendrite{FT=Float32}
    Er::Float32= - 70.6mV             # (mV) resting potential
    C::Float32 = 10pF                 # (1/pF) membrane timescale
	gax::Float32= 10nS				# (nS) axial conductance
	gm::Float32 = 1nS
	l::Float32	=150um				# μm distance from next compartment
	d::Float32	=4um				# μm dendrite diameter
end


@snn_kw mutable struct Tripod{MFT= Matrix{Float32}, VIT=Vector{Int32}, VFT=Vector{Float32},VBT=Vector{Bool}} <: AbstractIF
    param::AdExTripod = AdExTripod()
	## These are compulsory parameters
	soma_syn::Synapse = soma_syn
	dend_syn::Synapse = dend_syn
	d1::Vector{Dendrite}
	d2::Vector{Dendrite}
	##
    N::Int32 = 100
    v_s::VFT  = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    w_s::VFT = zeros(N)
    v_d1::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    v_d2::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
	#Synapses
    g_s::MFT  = zeros(N,2)
    g_d1::MFT = zeros(N,4)
    g_d2::MFT = zeros(N,4)
    h_s::MFT  = zeros(N,2)
    h_d1::MFT = zeros(N,4)
    h_d2::MFT = zeros(N,4)
	## Debug to be removed later
	# i1::VFT = zeros(N)
	# i2::VFT = zeros(N)
	# is::VFT = zeros(N)
	# c1::VFT = zeros(N)
	# c2::VFT = zeros(N)
	# Δs::VFT = zeros(N)
	# Δd1::VFT = zeros(N)
	# Δd2::VFT = zeros(N)
	## till here
    fire::VBT = zeros(Bool, N)
	after_spike::VFT = zeros(Int, N)
	postspike::PostSpike = PostSpike(A=10, τA= 30ms)

    θ::VFT = ones(N)*param.Vt
    records::Dict = Dict()
end

"""
    Tripod neuron
"""
function integrate!(p::Tripod, param::AdExTripod, dt::Float32)
    @unpack N, v_s, w_s, v_d1, v_d2, g_s, g_d1, g_d2, h_s, h_d1, h_d2, d1, d2, fire, θ, after_spike, postspike = p
	dend_receptors = [getfield(p.dend_syn, r) for r in [:AMPA, :NMDA, :GABAa, :GABAb] ]
	soma_receptors = [getfield(p.soma_syn, r) for r in [:AMPA, :GABAa] ]
	@unpack Er, up, idle, BAP, AP_membrane, Vr, Vt, τw, a, b = param
	## Update all synaptic conductance
	# for dendrites
    for (n,rec) in enumerate(dend_receptors)
        @unpack gsyn, α, E_rev, τr, τd = rec
	    @inbounds @fastmath for i = 1:N
			g_d1[i,n] = exp32(-dt/τd)*(g_d1[i,n] + dt*h_d1[i,n])
			h_d1[i,n] = exp32(-dt/τr)*(h_d1[i,n])
			g_d2[i,n] = exp32(-dt/τd)*(g_d2[i,n] + dt*h_d2[i,n])
			h_d2[i,n] = exp32(-dt/τr)*(h_d2[i,n])
		end
	end
	# for soma
    for (n,rec) in enumerate(soma_receptors)
        @unpack gsyn, α, E_rev, τr, τd = rec
	    @inbounds @fastmath for i = 1:N
			g_s[i,n] = exp32(-dt/τd)*(g_s[i,n] + dt*h_s[i,n])#),13500)
			h_s[i,n] = exp32(-dt/τr)*(h_s[i,n])
		end
	end

	Δv_temp = zeros(Float32, 3)
	Δv = zeros(Float32, 3)
    @inbounds for i = 1:N
		if after_spike[i] >idle
	        v_s[i] = BAP
			## backpropagation effect
			c1 = (BAP - v_d1[i])* d1[i].gax
			c2 = (BAP - v_d2[i])* d2[i].gax
			## apply currents
			v_d1[i] += dt*c1/d1[i].C
			v_d2[i] += dt*c2/d2[i].C

		elseif after_spike[i] > 0
			v_s[i] = Vr
			c1 = (Vr - v_d1[i])* d1[i].gax/100 #
			c2 = (Vr - v_d2[i])* d2[i].gax/100 #
			## apply currents
			v_d1[i] += dt*c1/d1[i].C
			v_d2[i] += dt*c2/d2[i].C
		else
			## Heun integration
			update_tripod(p, Δv,i, param, 0.f0)
			Δv_temp .+= Δv
			update_tripod(p, Δv,i, param, dt)
			Δv_temp .+= Δv
			# @show Δv_temp
			v_s[i]  += 0.5*dt * Δv_temp[1]
			v_d1[i] += 0.5*dt * Δv_temp[2]
			v_d2[i] += 0.5*dt * Δv_temp[3]
			w_s[i] += dt*ΔwAdEx(v_s[i], w_s[i], param)
		end
    end
    @inbounds for i = 1:N
		θ[i] -= dt* (θ[i] - Vt ) / postspike.τA
		after_spike[i] -= 1
		if after_spike[i] < 0
			## spike ?
			if v_s[i] > θ[i] + 10.
		        fire[i] = true
				θ[i] += postspike.A
		        v_s[i] = AP_membrane
				w_s[i] += b*τw
				after_spike[i] = (up+idle)/dt
			end
		end
    end
end

function update_tripod(p::Tripod, Δv::Vector{Float32}, i::Int64, param::AdExTripod, dt::Float32)
		@unpack v_d1, v_d2, v_s, w_s, g_s, g_d1, g_d2, d1, d2, θ = p
		# @unpack i1, i2,is, c1, c2, Δs,Δd2,Δd1,   θ = p

		#compute axial currents
		c1 = (- (v_d1[i]+Δv[2]*dt) + (v_s[i]+Δv[1]*dt))* d1[i].gax
		c2 = (- (v_d2[i]+Δv[3]*dt) + (v_s[i]+Δv[1]*dt))* d2[i].gax

		# compute synaptic currents
		i1 = syn_current(v_d1[i]+Δv[2]*dt, g_d2[i,:], p.dend_syn)
		i2 = syn_current(v_d2[i]+Δv[3]*dt, g_d2[i,:], p.dend_syn)
		is = syn_current_soma(v_s[i]+Δv[1]*dt,g_s[i,:], p.soma_syn)

		#cap currents
		i1 = min(abs(i1[1]),500)*sign(i1)
		i2 = min(abs(i2[1]),500)*sign(i2)
		is = min(abs(is[1]),500)*sign(is)

		Δv[1] = ΔvAdEx(v_s[i] +Δv[1]*dt, w_s[i], θ[i], +c1+c2, is , param)
		Δv[2] = ΔvDend(v_d1[i]+Δv[2]*dt, -c1, i1, d1[i])
		Δv[3] = ΔvDend(v_d2[i]+Δv[3]*dt, -c2, i2, d2[i])

		# Δs[i] = Δv[1]
		# Δd1[i] = Δv[2]
		# Δd2[i] = Δv[2]
end

@inline @fastmath function ΔvDend(v::Float32, axial::Float32, synaptic::Float32, pm::Dendrite)::Float32
	 return  ((-v + pm.Er)*pm.gm - synaptic -axial)/pm.C
end

@inline @fastmath function ΔvAdEx(v::Float32, w::Float32, θ::Float32, axial::Float32, synaptic::Float32, AdEx::AdExTripod)::Float32
	return ( AdEx.gl *( -v +AdEx.Er +  AdEx.ΔT * exp32(1/AdEx.ΔT*(v-θ))) - w  - synaptic - axial)/AdEx.C ## external currents
end

@inline @fastmath function ΔwAdEx(v::Float32, w::Float32,AdEx::AdExTripod)::Float32
	return ( AdEx.a *(v - AdEx.Er) - w)/ AdEx.τw
end

@inline function NMDA_nonlinear(NMDA::ReceptorVoltage, v::Float32)::Float32

		 return (1+(NMDA.mg/NMDA.b)*exp32(NMDA.k*(v)))^-1 ##NMDA
 end

@inline function syn_current(v::Float32, g::Vector{Float32}, syn::Synapse)
	return (syn.AMPA.gsyn * g[1]+
			syn.NMDA.gsyn * g[2]*NMDA_nonlinear(syn.NMDA, v))*(v)+
			syn.GABAa.gsyn *(v- syn.GABAa.E_rev)* g[3]	+
			syn.GABAb.gsyn *(v- syn.GABAb.E_rev)* g[4]
	end

@inline function syn_current_soma(v::Float32, g::Vector{Float32}, syn::Synapse)
	return (syn.AMPA.gsyn * g[1]*v)+syn.GABAa.gsyn *(v- syn.GABAa.E_rev)* g[2]
end

# function update_tripod(p::Tripod, Δv::Vector{Float32}, i::Int64, param::AdExTripod, dt::Float32)
# 		@unpack v_d1, v_d2, v_s, w_s, g_s, g_d1, g_d2, d1, d2, Δs,Δd2,Δd1,   θ = p
#
# 		c1 = (- (v_d1[i]+Δv[2]*dt) + (v_s[i]+Δv[1]*dt))* d1[i].gax
# 		c2 = (- (v_d2[i]+Δv[3]*dt) + (v_s[i]+Δv[1]*dt))* d2[i].gax
# 		i1 = syn_current(v_d1[i]+Δv[2]*dt, g_d2[i,:], p.dend_syn)
# 		i2 = syn_current(v_d2[i]+Δv[3]*dt, g_d2[i,:], p.dend_syn)
# 		is = syn_current_soma(v_s[i]+Δv[1]*dt,g_s[i,:], p.soma_syn)
# 		i1 = min(abs(i1[1]),1300)*sign(i1[i])
# 		i2 = min(abs(i2[1]),1300)*sign(i2[i])
# 		is = min(abs(is[1]),1300)*sign(is[i])
# 		Δv[1] = ΔvAdEx(v_s[i] +Δv[1]*dt, w_s[i], θ[i], +c1+c2, is , param)
# 		Δv[2] = ΔvDend(v_d1[i]+Δv[2]*dt, -c1, i1, d1[i])
# 		Δv[3] = ΔvDend(v_d2[i]+Δv[3]*dt, -c2, i2, d2[i])
# 		Δs[i] = Δv[1]
# 		Δd1[i] = Δv[2]
# 		Δd2[i] = Δv[2]
# end
#
