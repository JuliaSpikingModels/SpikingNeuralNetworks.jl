using Plots
using SpikingNeuralNetworks
SNN.@load_units

E = SNN.AdEx(;N = 3200, param = SNN.AdExParameter(;El=-49mV))
I = SNN.IF(;N = 800, param = SNN.IFParameter(;El = -49mV))
EE = SNN.SpikingSynapse(E, E, :ge; σ = 60*0.27/10, p = 0.02)
EI = SNN.SpikingSynapse(E, I, :ge; σ = 60*0.27/10, p = 0.02)
IE = SNN.SpikingSynapse(I, E, :gi; σ = -20*4.5/10, p = 0.02)
II = SNN.SpikingSynapse(I, I, :gi; σ = -20*4.5/10, p = 0.02)
P = [E, I]
C = [EE, EI, IE, II]

SNN.monitor([E, I], [:fire, (:v, [1, 25,50, 100])])
# SNN.train!(P, C; duration = 1second)
SNN.monitor([E], [( :v, [1, 25,50, 100])])
SNN.sim!(P, C; duration = 1second)
SNN.raster(P)
SNN.vecplot(E,:v)
