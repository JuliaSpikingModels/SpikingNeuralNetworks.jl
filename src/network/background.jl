using DrWatson
"""
	TripodBackground(Tripod_pop; N_E = 1000, N_I = 250, ν_E = 50Hz, ν_I = 50Hz, r0 = 10Hz, v0_d1 = -50mV, v0_d2 = -50mV, σ_s = 0.5f0)

Create a background feed for a population of Tripod neurons.

# Arguments
- `Tripod_pop`: the population of Tripod neurons.

# Keyword Arguments
- `N_E`: Number of excitatory Poisson neurons (default: 1000).
- `N_I`: Number of inhibitory Poisson neurons (default: 250).
- `ν_E`: Firing rate of excitatory Poisson neurons (default: 50Hz).
- `ν_I`: Firing rate of inhibitory Poisson neurons (default: 50Hz).
- `r0`: Base rate for STDP (default: 10Hz).
- `v0_d1`: Reversal potential for inhibitory synaptic inputs on dendrite 1(default: -50mV).
- `v0_d2`: Reversal potential for inhibitory synaptic inputs on dendrite 2 (default: -50mV).
- `σ_s`: Standard deviation for the weight distribution of excitatory synapses onto the soma (default: 0.5).

# Returns
- A tuple of two dictionaries: 
	- `back_syn`: contains the synapses created.
	- `back_pop`: contains the populations created.

"""
function TripodBalancedNoise(
    Tripod_pop;
    N_E = 1000,
    N_I = 250,
    ν_E = 20Hz,
    ν_I = 50Hz,
    r0 = 3Hz,
    v0 = -50mV,
    σ_s = 0.5f0,
)
    I = SNN.Poisson(N = N_I, param = SNN.PoissonParameter(rate = ν_I))
    E = SNN.Poisson(N = N_E, param = SNN.PoissonParameter(rate = ν_E))
    inh_d1 = SNN.SynapseTripod(
        I,
        Tripod_pop,
        "d1",
        "inh",
        p = 0.2,
        σ = 1,
        param = SNN.iSTDPParameterPotential(v0 = v0),
    )
    inh_d2 = SNN.SynapseTripod(
        I,
        Tripod_pop,
        "d2",
        "inh",
        p = 0.2,
        σ = 1,
        param = SNN.iSTDPParameterPotential(v0 = v0),
    )
    inh_s = SNN.SynapseTripod(
        I,
        Tripod_pop,
        "s",
        "inh",
        p = 0.4,
        σ = 1,
        param = SNN.iSTDPParameterRate(r = r0),
    )
    exc_d1 = SNN.SynapseTripod(E, Tripod_pop, "d1", "exc", p = 0.2, σ = 1.0)
    exc_d2 = SNN.SynapseTripod(E, Tripod_pop, "d2", "exc", p = 0.2, σ = 1.0)
    exc_s = SNN.SynapseTripod(E, Tripod_pop, "s", "exc", p = 0.2, σ = σ_s)

    synapses = dict2ntuple(@strdict inh_d1 inh_d2 inh_s exc_d1 exc_d2 exc_s)
    populations = dict2ntuple(@strdict I E)
    return (syn = synapses, pop = populations)
end

function TripodExcNoise(Tripod_pop; N_E = 1000, ν_s = 200Hz, ν_d = 200Hz, σ_s = 2.0f0)
    Ed = SNN.Poisson(N = N_E, param = SNN.PoissonParameter(rate = ν_d))
    Es = SNN.Poisson(N = N_E, param = SNN.PoissonParameter(rate = ν_s))
    exc_d1 = SNN.SynapseTripod(Ed, Tripod_pop, "d1", "exc", p = 0.2, σ = 1.0)
    exc_d2 = SNN.SynapseTripod(Ed, Tripod_pop, "d2", "exc", p = 0.2, σ = 1.0)
    exc_s = SNN.SynapseTripod(Es, Tripod_pop, "s", "exc", p = 0.2, σ = σ_s)
    synapses = dict2ntuple(@strdict exc_d1 exc_d2 exc_s)
    populations = dict2ntuple(@strdict Ed Es)
    return (syn = synapses, pop = populations)
end


function BalancedNoise(pop; N_E = 1000, N_I = 250, ν_E = 50Hz, ν_I = 50Hz, r0 = 10Hz)
    I = SNN.Poisson(N = N_I, param = SNN.PoissonParameter(rate = ν_I))
    E = SNN.Poisson(N = N_E, param = SNN.PoissonParameter(rate = ν_E))
    inh = SNN.SpikingSynapse(
        I,
        pop,
        :gi,
        p = 0.2,
        σ = 1,
        param = SNN.iSTDPParameterRate(r = r0),
    )
    exc = SNN.SpikingSynapse(E, pop, :ge, p = 0.2, σ = 1.0)

    synapses = dict2ntuple(@strdict exc inh)
    populations = dict2ntuple(@strdict I E)
    return (syn = synapses, pop = populations)
end

function ExcNoise(pop; N_E = 100, ν_E = 20Hz, name = "E")
    E = SNN.Poisson(N = N_E, param = SNN.PoissonParameter(rate = ν_E))
    exc = SNN.SpikingSynapse(E, pop, :ge, p = 0.2, σ = 2.0)
    synapses = dict2ntuple(Dict(Symbol("exc_$name") => exc))
    populations = dict2ntuple(Dict(Symbol("E_$name") => E))
    return (syn = synapses, pop = populations)
end
