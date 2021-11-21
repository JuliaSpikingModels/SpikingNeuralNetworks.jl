using Plots
# using SpikingNeuralNetworks
include("../src/SpikingNeuralNetworks.jl")
SNN.@load_units

E = SNN.AdEx(;N = 1)
E.I = [21]
E.param

E.param.R
E.param.b
# TN.AdEx.Rm *TN.AdEx.b
##

SNN.monitor(E, [:v, :fire, :w])

SNN.sim!([E], []; duration = 700ms)
plot(SNN.vecplot(E, :w),SNN.vecplot(E, :v))
