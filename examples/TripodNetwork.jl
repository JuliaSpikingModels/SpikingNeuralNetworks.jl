using DrWatson
using Plots
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Random, Statistics, StatsBase
using Statistics, SparseArrays
using StatsPlots
using ProgressBars

# %% [markdown]
# Network
network = let
	N = 200
	d1 = [SNN.Dendrite(; SNNUtils.create_dendrite(l = rand(350:1:350) * um)...) for n in 1:N]
	d2 = [SNN.Dendrite(; SNNUtils.create_dendrite(l = 0um)...) for n in 1:N]
	E = SNN.TripodNeurons(N = N, d1 = d1, d2 = d2,
		soma_syn = Synapse(DuarteGluSoma, MilesGabaSoma),
		dend_syn = Synapse(EyalGluDend, MilesGabaDend),
		NMDA = SNN.EyalNMDA,
		param = SNN.AdExTripod(b = 0.0f0, Vr = -50))
	I1 = SNN.IF(; N = 20, param = SNN.IFParameter(τm = 7ms, El = -55mV))
	I2 = SNN.IF(; N = 20, param = SNN.IFParameter(τm = 20ms, El = -55mV))
	E_to_I1 = SNN.SpikingSynapse(E, I1, :ge, p = 0.2, σ = 5.0)
	E_to_I2 = SNN.SpikingSynapse(E, I2, :ge, p = 0.2, σ = 5.0)
	I2_to_E = SNN.SynapseTripod(I2, E, "d1", "inh", p = 0.2, σ = 10, param = SNN.iSTDPParameterPotential(v0 = -50mV))
	I1_to_E = SNN.SynapseTripod(I1, E, "s", "inh", p = 0.2, σ = 10, param = SNN.iSTDPParameterRate(r = 10Hz))
	recurrent_norm = SNN.SynapseNormalization(N, param = SNN.MultiplicativeNorm(τ = 100ms))
	E_to_E = SNN.SynapseTripod(E, E, "d1", "exc", p = 0.2, σ = 30, param = SNN.vSTDPParameter(), normalize = recurrent_norm)
	pop = dict2ntuple(@strdict E I1 I2)
	syn = dict2ntuple(@strdict E_to_E I1_to_E I2_to_E E_to_I1 E_to_I2)
	(pop = pop, syn = syn)
end

# background
@unpack back_syn, back_pop = SNN.TripodBackground(network.pop.E, σ_s = 5.0, v0_d1 = -60mV, v0_d2 = -60mV, ν_E = 80Hz)

populations = [network.pop..., back_pop...]
synapses = [network.syn..., back_syn...]

SNN.clear_records([network.pop...])

@profview SNN.train!(populations, synapses, duration = 2000ms)

SNN.monitor(network.pop.E, [:v_d1, :v_s, :fire])
SNN.monitor(network.pop.I1, [:fire])
SNN.monitor(network.pop.I2, [:fire])

using BenchmarkTools
@btime SNN.sim!(populations, synapses, duration = 1000ms)

@profview SNN.sim!(populations, synapses, duration = 1000ms)



SNN.raster([network.pop...])
