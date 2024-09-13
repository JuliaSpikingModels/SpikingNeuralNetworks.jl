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
	NE = 400
	NI = 100
	d1 = [SNN.Dendrite(; SNNUtils.create_dendrite(l = rand(150:1:350) * um)...) for n in 1:NE]
	d2 = [SNN.Dendrite(; SNNUtils.create_dendrite(l = rand(150:1:350))...) for n in 1:NE]
	E = SNN.TripodNeurons(N = NE, d1 = d1, d2 = d2,
		soma_syn = Synapse(DuarteGluSoma, MilesGabaSoma),
		dend_syn = Synapse(EyalGluDend, MilesGabaDend),
		NMDA = SNN.EyalNMDA,
		param = SNN.AdExTripod(Vr = -50))
	I1 = SNN.IF(; N = NI÷2, param = SNN.IFParameter(τm = 7ms, El = -55mV))
	I2 = SNN.IF(; N = NI÷2, param = SNN.IFParameter(τm = 20ms, El = -55mV))
	E_to_I1 = SNN.SpikingSynapse(E, I1, :ge, p = 0.2, σ = 5.0)
	E_to_I2 = SNN.SpikingSynapse(E, I2, :ge, p = 0.2, σ = 5.0)
	I2_to_E = SNN.SynapseTripod(I2, E, "d1", "inh", p = 0.2, σ = 5., param = SNN.iSTDPParameterPotential(v0 = -50mV))
	I1_to_E = SNN.SynapseTripod(I1, E, "s", "inh", p = 0.2, σ = 5., param = SNN.iSTDPParameterRate(r = 10Hz))
	recurrent_norm_d1 = SNN.SynapseNormalization(E.N, param = SNN.MultiplicativeNorm(τ = 100ms))
	recurrent_norm_d2 = SNN.SynapseNormalization(E.N, param = SNN.MultiplicativeNorm(τ = 100ms))
	E_to_E_d1 = SNN.SynapseTripod(E, E, "d1", "exc", p = 0.2, σ = 30, param = SNN.vSTDPParameter(), normalize = recurrent_norm_d1)
	E_to_E_d2 = SNN.SynapseTripod(E, E, "d2", "exc", p = 0.2, σ = 30, param = SNN.vSTDPParameter(), normalize = recurrent_norm_d2)
	pop = dict2ntuple(@strdict E I1 I2)
	syn = dict2ntuple(@strdict E_to_E_d1 E_to_E_d2 I1_to_E I2_to_E E_to_I1 E_to_I2)
	norm = dict2ntuple(@strdict recurrent_norm_d1 recurrent_norm_d2)
	(pop = pop, syn = syn, norm=norm)
end

# background
@unpack back_syn, back_pop = SNN.TripodBackground(network.pop.E, σ_s = 5.0, v0_d1 = -60mV, v0_d2 = -60mV, ν_E = 100Hz)

populations = [network.pop..., back_pop...]
synapses = [network.syn..., back_syn..., network.norm...]

##
SNN.clear_records([network.pop...])
SNN.train!(populations, synapses, duration = 5000ms)
SNN.monitor(network.pop.E, [:v_d1, :v_s, :fire])
SNN.monitor(network.pop.I1, [:fire])
SNN.monitor(network.pop.I2, [:fire])

# using BenchmarkTools
# @btime SNN.sim!(populations, synapses, duration = 1000ms)
# @profview SNN.sim!(populations, synapses, duration = 1000ms)
SNN.sim!(populations, synapses, duration = 1000ms)

SNN.raster([network.pop...])
savefig(plotsdir("example_raster.pdf"))
