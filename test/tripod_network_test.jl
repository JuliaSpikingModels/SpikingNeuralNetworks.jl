using Random, Statistics, StatsBase
using BenchmarkTools

function test_network()
    Random.seed!(123)
    network = let
        NE = 400
        NI = 100
        d1 = [SNN.Dendrite(; create_dendrite(l = rand(150:1:350) * um)...) for n = 1:NE]
        d2 = [SNN.Dendrite(; create_dendrite(l = rand(150:1:350))...) for n = 1:NE]
        E = SNN.TripodNeurons(
            N = NE,
            d1 = d1,
            d2 = d2,
            soma_syn = Synapse(DuarteGluSoma, MilesGabaSoma),
            dend_syn = Synapse(EyalGluDend, MilesGabaDend),
            NMDA = SNN.EyalNMDA,
            param = SNN.AdExTripod(Vr = -50),
        )
        I1 = SNN.IF(; N = NI ÷ 2, param = SNN.IFParameter(τm = 7ms, El = -55mV))
        I2 = SNN.IF(; N = NI ÷ 2, param = SNN.IFParameter(τm = 20ms, El = -55mV))
        E_to_I1 = SNN.SpikingSynapse(E, I1, :ge, p = 0.2, σ = 15.0)
        E_to_I2 = SNN.SpikingSynapse(E, I2, :ge, p = 0.2, σ = 15.0)
        I2_to_E = SNN.SynapseTripod(
            I2,
            E,
            :d1,
            :inh,
            p = 0.2,
            σ = 5.0,
            param = SNN.iSTDPParameterPotential(v0 = -50mV),
        )
        I1_to_E = SNN.SynapseTripod(
            I1,
            E,
            "s",
            :inh,
            p = 0.2,
            σ = 5.0,
            param = SNN.iSTDPParameterRate(r = 10Hz),
        )
        E_to_E_d1 = SNN.SynapseTripod(
            E,
            E,
            :d1,
            :exc,
            p = 0.2,
            σ = 30,
            param = SNN.vSTDPParameter(),
        )
        E_to_E_d2 = SNN.SynapseTripod(
            E,
            E,
            :d2,
            :exc,
            p = 0.2,
            σ = 30,
            param = SNN.vSTDPParameter(),
        )
        pop = dict2ntuple(@strdict E I1 I2)
        syn = dict2ntuple(@strdict E_to_E_d1 E_to_E_d2 I1_to_E I2_to_E E_to_I1 E_to_I2)
        recurrent_norm_d1 = SNN.SynapseNormalization(
            E,
            [E_to_E_d1],
            param = SNN.MultiplicativeNorm(τ = 100ms),
        )
        recurrent_norm_d2 = SNN.SynapseNormalization(
            E,
            [E_to_E_d2],
            param = SNN.MultiplicativeNorm(τ = 100ms),
        )
        norm = dict2ntuple(@strdict d1 = recurrent_norm_d1 d2 = recurrent_norm_d2)
        (pop = pop, syn = syn, norm = norm)
    end

    # background
    noise = SNN.TripodBalancedNoise(network.pop.E, σ_s = 5.0, v0 = -60mV, ν_E = 100Hz)

    populations = [network.pop..., noise.pop...]
    synapses = [network.syn..., noise.syn..., network.norm...]

    #
    SNN.clear_records([network.pop...])
    SNN.train!(populations, synapses, duration = 5000ms)
    SNN.monitor(network.pop.E, [:v_d1, :v_s, :fire])
    SNN.monitor(network.pop.I1, [:fire])
    SNN.monitor(network.pop.I2, [:fire])
    SNN.sim!(populations, synapses, duration = 1000ms)
    WI1 = network.syn.I1_to_E.W
    WI2 = network.syn.I2_to_E.W
    WEd1 = network.syn.E_to_E_d1.W
    WEd2 = network.syn.E_to_E_d2.W
    spikes = SNN.spiketimes(network.pop.E)

    timing = @belapsed train!($populations, $synapses, duration = 500ms)
    data = @strdict WI1 WI2 WEd2 WEd1 spikes timing

    # tagsave(datadir("network_test", "TripodNetwork.jld2"), data, safe=true)
    ## Test with previous results
    data_new = dict2ntuple(data)
    data_old = dict2ntuple(DrWatson.load(datadir("network_test", "TripodNetwork.jld2")))

    @info "Previous timing: $(data_old.timing)"
    @info "Present timing: $(data_new.timing)"
    @assert sum(data_new.WI1 .- data_old.WI1) ≈ 0
    @assert sum(data_new.WI2 .- data_old.WI2) ≈ 0
    @assert sum(data_new.WEd2 .- data_old.WEd2) ≈ 0
    @assert sum(data_new.WEd1 .- data_old.WEd1) ≈ 0
    @info "The simulation reproduces original data"
    return true
end
