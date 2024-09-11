# This script simulates a Spiking Neural Network (SNN) using the SNNUtils and SpikingNeuralNetworks packages in Julia.
# Packages like DrWatson and Plots are also used for project management and data visualization respectively. 

# The network consists of Tripod neurons, Poisson neurons representing excitatory and inhibitory populations, 
# and synapses connecting these neuron populations.

# The script begins by importing necessary libraries and setting the units for physical quantities using the SNN.@load_units command.

# Two different types of dendrites (d1 and d2) are created for the Tripod neurons with varying properties. 
# The length of dendrite d1 is randomly chosen between 250 to 350 um while d2 has length 0um.

# The Tripod_pop object combines the neurons and their attributes including the type of synapse on soma and dendrites,
# NMDA channel parameters, and certain neuron-specific parameters like reset potential and adaptation current b.

# The network contains 1000 excitatory (E) and 200 inhibitory (I) Poisson neurons. 
# These neurons fire at a constant rate defined by ν_E and ν_I respectively. 
# The initial firing rate r0 for the synapses and the initial membrane potentials v0_d1 and v0_d2 for the dendrites are also set.

# Synapses are created between the Poisson neurons and the Tripod neurons. 
# This includes inhibitory synapses to the dendrite d1 and to the soma "s", as well as excitatory synapses to dendrite d1 and soma "s".
# The strength of synapses (σ) and connection probability (p) are defined during creation of these synapses.

# The neural network is trained for a given duration using the SNN.train! function. 
# After training, the population firing and voltages are monitored using the SNN.monitor function, 
# and then the network is simulated for a defined duration. 

# The final output of the script are plots of the voltages across different compartments of the Tripod neurons.

using DrWatson
using Plots
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Random, Statistics, StatsBase
using Statistics, SparseArrays

# %% [markdown]
# Create vectors of dendritic parameters and the Tripod model
N = 1
d1 = [SNN.Dendrite(; SNNUtils.create_dendrite(l = rand(250:1:350) * um)...) for n in 1:N]
d2 = [SNN.Dendrite(; SNNUtils.create_dendrite(l = 0um)...) for n in 1:N]
Tripod_pop = SNN.TripodNeurons(N = N, d1 = d1, d2 = d2,
	soma_syn = Synapse(DuarteGluSoma, MilesGabaSoma),
	dend_syn = Synapse(EyalGluDend, MilesGabaDend),
	NMDA = SNN.EyalNMDA,
	param = SNN.AdExTripod(b = 0.0f0, Vr = -50))

d2[1]
# background
N_E = 1000
N_I = 200
ν_E = 50Hz
ν_I = 50Hz
r0 = 10Hz
v0_d1 = -40mV
v0_d2 = -40mV
σ_s = 1.5f0

I = SNN.Poisson(N = N_I, param = SNN.PoissonParameter(rate = ν_E))
E = SNN.Poisson(N = N_E, param = SNN.PoissonParameter(rate = ν_I))
inh_d1 = SNN.SynapseTripod(
	I,
	Tripod_pop,
	"d1",
	"inh",
	p = 0.2,
	σ = 1,
	param = SNN.iSTDPParameterPotential(v0 = v0_d1),
)
inh_s = SNN.SynapseTripod(
	I,
	Tripod_pop,
	"s",
	"inh",
	p = 0.1,
	σ = 1,
	param = SNN.iSTDPParameterRate(r = r0),
)
exc_d1 = SNN.SynapseTripod(E, Tripod_pop, "d1", "exc", p = 0.2, σ = 15.0)
exc_s = SNN.SynapseTripod(E, Tripod_pop, "s", "exc", p = 0.2, σ = σ_s)

synapses = [inh_d1, inh_s, exc_d1, exc_s]
populations = [Tripod_pop, I, E]

SNN.train!(populations, synapses, duration = 5000ms)
##
SNN.monitor(Tripod_pop, [:fire, :v_d1, :v_s, :v_d2])
SNN.sim!(populations, synapses, duration = 10000ms)
SNN.vecplot(Tripod_pop, [:v_d2, :v_d1, :v_s])
