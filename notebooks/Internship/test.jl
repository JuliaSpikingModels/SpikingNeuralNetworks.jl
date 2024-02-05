using Revise
using DrWatson
@quickactivate "SpikingNeuralNetworks"
using Plots
using SpikingNeuralNetworks

default(
    size = (800, 600),
    tickfontsize = 8,
    guidefontsize = 10,
    margin = 8Plots.mm,
    titlefontsize = 13,
    titlefontcolor = :teal,
    legend = true,
    palette = (:Dark2_5),
)

SNN.@load_units
