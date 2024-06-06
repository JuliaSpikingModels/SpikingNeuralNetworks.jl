# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
# ---

# %%
##
using DrWatson
using Plots
@quickactivate "SpikingNeuralNetworks"
using SNNUtils
using SpikingNeuralNetworks
SNN.@load_units;


# %% [markdown]
# Create vectors of dendritic parameters and the Tripod model

# %%
N = 100
d1 = [SNN.Dendrite(;SNNUtils.create_dendrite(rand(150:1:400))...) for n in 1:N]
d2 = [SNN.Dendrite(;SNNUtils.create_dendrite(rand(150:1:400))...) for n in 1:N]
E = SNN.Tripod(N=N, d1=d1, d2=d2)  #is it one soma with 2*200 dendrites??

# %% [markdown]
# Create the input spike trains

# %%
inputs = [
    SNN.Poisson(N = 1000,param=SNN.PoissonParameter(rate=5Hz)),
    SNN.Poisson(N = 1000,param=SNN.PoissonParameter(rate=5Hz))
    ]
projections = []
for d in ["d1", "d2"]
    push!(projections,SpikingNeuralNetworks.SynapseTripod(inputs[1], E, d, "exc", p=0.2, σ=4.5))
    push!(projections,SpikingNeuralNetworks.SynapseTripod(inputs[2], E, d, "inh", p=0.2, σ=7.))
end



# %% [markdown]
# Create the recurrent connections
# %%
recurrent_d1 =  SpikingNeuralNetworks.SynapseTripod(E, E, "d1", "exc", p=0.2, σ=1.)
recurrent_d2 =  SpikingNeuralNetworks.SynapseTripod(E, E, "d2", "exc", p=0.2, σ=1.)

# %% [markdown]
# Select the variables to monitor
# %%

# initialize simulation with 1s, without recordings
SNN.sim!([E,inputs...],[projections... ];dt=0.1ms, duration=1000ms)

# always record firing rate
SNN.monitor(E, [:fire]) 

# record membranes of 20 neurons
mont_idx = rand(1:N,20)
mont = [(s,mont_idx) for s in [:v_s, :v_d1,:v_d2]]
SNN.monitor(E, mont)

# add recordings if necessary
all_membrane = [:v_s, :v_d1,:v_d2,]
all_conductances = [:h_s, :g_d2, :g_d1, :after_spike,:w_s]

##

# %% [markdown]
# Run simulation
# %%

SNN.sim!([E,inputs...],[projections... ];dt=0.1ms, duration=1000ms)
SNN.raster([E])

# %%
plot(SNN.vecplot(E, :v_s),SNN.vecplot(E, :v_d2), SNN.vecplot(E,:v_d1), layout =(3,1))
# plot(SNN.vecplot(E, :g_d1),SNN.vecplot(E, :g_d2))
# E.records[:g_d2][end] 

# Benchmarking
using BenchmarkTools
@benchmark  SNN.sim!([E,inputs...],[projections... ];dt=0.1ms, duration=1000ms)

