using Plots
using SpikingNeuralNetworks
SNN.@load_units

E = SNN.AdEx(; N = 1, param = SNN.AdExParameter(; El = -49mV))
E.param

E.param.R
E.param.b
# TN.AdEx.Rm *TN.AdEx.b
##

SNN.monitor(E, [:v, :fire, :w])
SNN.sim!([E], []; duration = 700ms)
plot(SNN.vecplot(E, :w), SNN.vecplot(E, :v))

E.param.C
