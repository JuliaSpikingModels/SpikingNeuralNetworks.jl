# Get synapses from SNNUtils


@snn_kw struct AdExTripod{FT <: Float32} <: AbstractIFParameter
    #Membrane parameters
    C::FT = 281pF           # (pF) membrane timescale
    gl::FT = 40nS                # (nS) gl is the leaking conductance,opposite of Rm
    R::FT = nS / gl * GΩ               # (GΩ) total membrane resistance
    τm::FT = C / gl                # (ms) C / gl
    Er::FT = -70.6mV          # (mV) resting potential
    # AdEx model
    Vr::FT = -55.6mV     # (mV) Reset potential of membrane
    Vt::FT = -50.4mV          # (mv) Rheobase threshold
    ΔT::FT = 2mV            # (mV) Threshold sharpness
    # Adaptation parameters
    τw::FT = 144ms          #ms adaptation current relaxing time
    a::FT = 4nS            #nS adaptation current to membrane
    b::FT = 80.5pA         #pA adaptation current increase due to spike
    # After spike timescales and membrane
    AP_membrane::Float32 = 10.0f0mV
    BAP::Float32 = 1.0f0mV
    up::Int64 = 1ms
    idle::Int64 = 2ms
end


@snn_kw struct PostSpike{FT <:Float32}
    A::FT
    τA::FT
end


@snn_kw struct Dendrite{FT =  Float32}
    Er::FT = -70.6mV             # (mV) resting potential
    C::FT = 10pF                 # (1/pF) membrane timescale
    gax::FT = 10nS# (nS) axial conductance
    gm::FT = 1nS
    l::FT = 150um# μm distance from next compartment
    d::FT = 4um# μm dendrite diameter
end


@snn_kw struct Tripod{MFT = Matrix{Float32},VIT = Vector{Int32},VFT = Vector{Float32},VBT = Vector{Bool}, VDT=Vector{Dendrite}, ST=SynapseArray, NMDAT=NMDAVoltageDependency, PST=PostSpike} <: AbstractIF
    param::AdExTripod = AdExTripod()
    ## These are compulsory parameters
    N::Int32 = 100
    NMDA::NMDAT =  NMDAVoltageDependency(mg=Mg_mM, b=nmda_b, k=nmda_k)
    soma_syn::SynapseArray
    dend_syn::SynapseArray
    ##
    # dendrites
    gax1::VFT=zeros(N)
    gax2::VFT=zeros(N)
    cd1::VFT=zeros(N)
    cd2::VFT=zeros(N)
    gm1::VFT=zeros(N)
    gm2::VFT=zeros(N)
    # Synapses
    τr::SArray{Tuple{4,2},Float32}
    τd::SArray{Tuple{4,2},Float32}
    gsyn::SArray{Tuple{4,2},Float32}
    E_rev::SArray{Tuple{4,2},Float32}
    rtypes::SArray{Tuple{4,2},String}
    ##
    # Membrane potential and adaptation
    v_s::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    w_s::VFT = zeros(N)
    v_d1::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    v_d2::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    ##
    g_s::MFT = zeros(N, 4)
    g_d1::MFT = zeros(N, 4)
    g_d2::MFT = zeros(N, 4)
    h_s::MFT = zeros(N, 4)
    h_d1::MFT = zeros(N, 4)
    h_d2::MFT = zeros(N, 4)
    # Spike model and threshold
    fire::VBT = zeros(Bool, N)
    after_spike::VFT = zeros(Int, N)
    postspike::PST = PostSpike(A=10, τA=30ms)
    θ::VFT = ones(N) * param.Vt
    records::Dict = Dict()
    Δv::MArray{Tuple{3},Float32}=MArray{Tuple{3}}([0.f0, 0.f0, 0.f0])
    Δv_temp::MArray{Tuple{3},Float32}=MArray{Tuple{3}}([0.f0, 0.f0, 0.f0])
    cs::MArray{Tuple{2},Float32}=MArray{Tuple{2}}([0.f0, 0.f0])
    is::MArray{Tuple{3},Float32}=MArray{Tuple{3}}([0.f0, 0.f0, 0.f0])
end

function TripodPopulation(;N::Int, d1::Vector{Dendrite}, d2::Vector{Dendrite}, soma_syn::Synapse, dend_syn::Synapse, NMDA::NMDAVoltageDependency)::Tripod
    gax1 = [d.gax for d in d1]
    gax2 = [d.gax for d in d2]
    cd1 = [d.C for d in d1]
    cd2 = [d.C for d in d2]
    gm1 = [d.gm for d in d1]
    gm2 = [d.gm for d in d2]
    dend_syn = SNN.synapsearray(dend_syn)
    soma_syn = SNN.synapsearray(soma_syn)
    τr = SNN.SArray{Tuple{4,2}}(hcat([[1/r.τr for r in dend_syn], [1/r.τr for r in soma_syn]]...))
    τd = SNN.SArray{Tuple{4,2}}(hcat([[1/r.τd for r in dend_syn], [1/r.τd for r in soma_syn]]...))
    gsyn = SNN.SArray{Tuple{4,2}}(hcat([[r.gsyn for r in dend_syn], [r.gsyn for r in soma_syn]]...))
    E_rev = SNN.SArray{Tuple{4,2}}(hcat([[r.E_rev for r in dend_syn], [r.E_rev for r in soma_syn]]...))
    rtypes = SNN.SArray{Tuple{4,2}}(hcat([[r.type for r in dend_syn], [r.type for r in soma_syn]]...))
    return Tripod(N=N, τr=τr, τd=τd, gsyn=gsyn, rtypes=rtypes, E_rev=E_rev, NMDA=NMDA, gax1=gax1, gax2=gax2, cd1=cd1, cd2=cd2, gm1=gm1, gm2=gm2, soma_syn=soma_syn, dend_syn=dend_syn)
end

#const dend_receptors::SVector{Symbol,3} = [:AMPA, :NMDA, :GABAa, :GABAb]
# const soma_receptors::Vector{Symbol} = [:AMPA, :GABAa]
const soma_rr = SA[:AMPA,:GABAa]
const dend_rr = SA[:AMPA,:NMDA,:GABAa,:GABAb]

function integrate!(p::Tripod, param::AdExTripod, dt::Float32)
    @unpack N, v_s, w_s, v_d1, v_d2, g_s, g_d1, g_d2, h_s, h_d1, h_d2, fire, θ, after_spike, postspike, Δv, Δv_temp = p
    @unpack Er, up, idle, BAP, AP_membrane, Vr, Vt, τw, a, b = param
    @unpack gax1, gax2, gm1, gm2, cd1, cd2 = p
    @unpack τr, τd = p

    ## Update all synaptic conductance
    # for dendrites
    @inbounds for n in axes(τr,1)
        @fastmath @simd for i = 1:N
                # dendrites x = 1
                g_d1[i, n] = exp32(-dt * τr[n,1]) * (g_d1[i, n] + dt * h_d1[i, n])
                h_d1[i, n] = exp32(-dt * τr[n,1]) * (h_d1[i, n])
                g_d2[i, n] = exp32(-dt * τd[n,1]) * (g_d2[i, n] + dt * h_d2[i, n])
                h_d2[i, n] = exp32(-dt * τr[n,1]) * (h_d2[i, n])
                #soma x = 2
                g_s[i, n] =  exp32(-dt * τd[n,2]) * (g_s[i, n] + dt * h_s[i, n])
                h_s[i, n] =  exp32(-dt * τr[n,2]) * (h_s[i, n])
        end
    end

    # update the neurons
    @inbounds @fastmath @simd for i = 1:N
        if after_spike[i] > idle
            v_s[i] = BAP
            ## backpropagation effect
            c1 = (BAP - v_d1[i]) * gax1[i] / 100 #
            c2 = (BAP - v_d2[i]) * gax2[i] / 100 #
            ## apply currents
            v_d1[i] += dt * c1 / cd1[i]
            v_d2[i] += dt * c2 / cd2[i]
        elseif after_spike[i] > 0
            v_s[i] = Vr
            c1 = (Vr - v_d1[i]) * gax1[i] / 100 #
            c2 = (Vr - v_d2[i]) * gax2[i] / 100 #
            ## apply currents
            v_d1[i] += dt * c1 / cd1[i]
            v_d2[i] += dt * c2 / cd2[i]
        else
            ## Heun integration
            for _i in 1:3
                Δv_temp[_i] = 0.f0
                Δv[_i] = 0.f0
            end
            update_tripod!(p, Δv, i, param, 0.0f0)
            for _i in 1:3
                Δv_temp[_i] = Δv[_i]
            end
            update_tripod!(p, Δv, i, param, dt)
            v_s[i]  += 0.5 * dt * (Δv_temp[1] +Δv[1])
            v_d1[i] += 0.5 * dt * (Δv_temp[2] +Δv[2])
            v_d2[i] += 0.5 * dt * (Δv_temp[3] +Δv[3])
            w_s[i] += dt * ΔwAdEx(v_s[i], w_s[i], param)
        end
    end

    # reset firing
    fire .= false
    @inbounds @fastmath @simd for i = 1:N
        θ[i] -= dt * (θ[i] - Vt) / getfield(postspike,:τA)
        after_spike[i] -= 1
        if after_spike[i] < 0
            ## spike ?
            if v_s[i] > θ[i] + 10.f0
                fire[i] = true
                θ[i] += getfield(postspike,:A)
                v_s[i] = AP_membrane
                w_s[i] += b ##  *τw
                after_spike[i] = (up + idle) / dt
            end
        end
    end
    return 
end

function update_tripod!(p::Tripod, Δv::MVector, i::Int64, param::AdExTripod, dt::Float32)

    @inbounds @fastmath begin

    @unpack v_d1, v_d2, v_s, w_s, g_s, g_d1, g_d2, θ = p
    @unpack gax1, gax2, gm1, gm2, cd1, cd2 = p
    # @unpack soma_syn, dend_syn, NMDA= p
    @unpack gsyn, E_rev, rtypes, NMDA = p
    @unpack is, cs = p

    #compute axial currents
    cs[1] = -( (v_d1[i] + Δv[2] * dt) - (v_s[i] + Δv[1] * dt)) * gax1[i]
    cs[2] = -( (v_d2[i] + Δv[3] * dt) - (v_s[i] + Δv[1] * dt)) * gax2[i]

    @simd for _i in 1:3
        is[_i] = 0.f0
    end
    @simd for r in axes(gsyn,1)
        if rtypes[r,1] =="nmda"
            is[2] += gsyn[r,1] * g_d1[i, r]*(v_d1[i] + Δv[2] * dt - E_rev[r,1]) *(1.f0+(NMDA.mg/NMDA.b)*exp32(NMDA.k*(v_d1[i] + Δv[2] * dt)))^-1
            is[3] += gsyn[r,1] * g_d2[i, r]*(v_d2[i] + Δv[3] * dt - E_rev[r,1]) *(1.f0+(NMDA.mg/NMDA.b)*exp32(NMDA.k*(v_d2[i] + Δv[3] * dt)))^-1
            is[1] += gsyn[r,2] * g_s[i, r]*(v_s[i] + Δv[1] * dt - E_rev[r,2]) 
        else
            is[2] += gsyn[r,1] * g_d1[i, r]*(v_d1[i] + Δv[2] * dt - E_rev[r,1])
            is[3] += gsyn[r,1] * g_d2[i, r]*(v_d2[i] + Δv[3] * dt - E_rev[r,1])
            # (gsyn[n,2]==0) && (continue)
            is[1] += gsyn[r,2] *  g_s[i, r]*(v_s[i] +  Δv[1] * dt - E_rev[r,2]) 
        end
    end
    @simd for _i in 1:3
        is[_i] = clamp(is[_i], -500,500)
    end

    # update membrane potential
    Δv[1] = ΔvAdEx(v_s[i] + Δv[1] * dt, w_s[i], θ[i], + sum(cs), is[1], param)
    Δv[2] = ((-(v_d1[i] + Δv[2] * dt) + param.Er) * gm1[i] - is[2] + cs[1]) / cd1[i] 
    Δv[3] = ((-(v_d2[i] + Δv[3] * dt) + param.Er) * gm2[i] - is[3] + cs[2]) / cd2[i] 
    end

end


@inline
@fastmath function ΔvAdEx(v::Float32, w::Float32, θ::Float32, axial::Float32, synaptic::Float32, AdEx::AdExTripod)::Float32
    return 1/ AdEx.C * (
        AdEx.gl * (
                (-v + AdEx.Er) + 
                AdEx.ΔT * exp32(1 / AdEx.ΔT * (v - θ))
                ) 
                - w 
                - synaptic 
                - axial
        ) 
end ## external currents

@inline
@fastmath function ΔwAdEx(v::Float32, w::Float32, AdEx::AdExTripod)::Float32
    return (AdEx.a * (v - AdEx.Er) - w) / AdEx.τw
end

export  Tripod, TripodPopulation, Dendrite, PostSpike
