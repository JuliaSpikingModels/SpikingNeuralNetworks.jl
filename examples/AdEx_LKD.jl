using Plots
using SpikingNeuralNetworks
SNN.@load_units
import SpikingNeuralNetworks: AdExParameter
using Statistics, Random


## LKD parameters
τm = 20ms
C = 300SNN.pF # Capacitance
R = τm / C
τre = 1ms # Rise time for excitatory synapses
τde = 6ms # Decay time for excitatory synapses
τri = 0.5ms # Rise time for inhibitory synapses # TODO: why does he use -?
τdi = 2ms # Decay time for inhibitory synapses

LKD_AdEx_exc =
    AdExParameter(τm = 20ms, τe = 6ms, τi = 2ms, Vt = -52mV, Vr = -60mV, El = -70mV, R = R, 
    b = 0.000805nA, τw = 150ms, τre = τre, τde = τde)
LKD_AdEx_inh =
    AdExParameter(τm = 20ms, τe = 6ms, τi = 2ms, Vt = -52mV, Vr = -60mV, El = -62mV, R = R, 
    b = 0.000805nA, τw = 150ms, τri = τri, τdi = τdi)

# inputs, the kHz is obtained by the N*ν, so doing the spikes (read eqs. 4 section)
N = 1000
νe = 4.5Hz # Rate of external input to E neurons
νi = 2.25Hz # Rate of external input to I neurons
p_in = 1.0 # 0.5 

σ_in_E = 1.78SNN.pF
σ_in_I = 48.7SNN.pF

σEE = 2.76SNN.pF # Initial E to E synaptic weight
σEI = 1.27SNN.pF # Synaptic weight from E to I
σIE = 48.7SNN.pF # Initial I to E synaptic weight
σII = 16.2SNN.pF # Synaptic weight from I to I
#

Input_E = SNN.Poisson(; N = N, param = SNN.PoissonParameter(; rate = νe))
Input_I = SNN.Poisson(; N = N, param = SNN.PoissonParameter(; rate = νi))
E = SNN.AdEx(; N = 4000, param = LKD_AdEx_exc)
I = SNN.AdEx(; N = 1000, param = LKD_AdEx_inh)

EE = SNN.SpikingSynapse(E, E, :ge; σ = σEE, p = 0.2)
EI = SNN.SpikingSynapse(E, I, :ge; σ = σEI, p = 0.2)
IE = SNN.SpikingSynapse(I, E, :gi; σ = σIE, p = 0.2)
II = SNN.SpikingSynapse(I, I, :gi; σ = σII, p = 0.2)
ProjE = SNN.SpikingSynapse(Input_E, E, :ge; σ = σ_in_E, p = p_in)
ProjI = SNN.SpikingSynapse(Input_I, I, :ge; σ = σ_in_I, p = p_in)

#3
P = [E, I, Input_E, Input_I]
C = [EE, EI, IE, II, ProjE, ProjI]

##
Random.seed!(23)
SNN.monitor([E, I], [:fire])
SNN.sim!(P, C; duration = 15second)
bar(sum(hcat(E.records[:fire]...) ./ 15, dims = 2)[:, 1])
SNN.raster([E, I], [9, 11] .* 10e3)

# SNN.monitor([E, I], [:ge, :gi, :v])

# plot(
#     [hcat(E.records[:ge]...)[123,:] ,
#     hcat(E.records[:gi]...)[123,:]]
# )

##
# Random.seed!(101)
# E = SNN.AdEx(;N = 100, param = AdExParameter(;El=-40mV))
# EE = SNN.SpikingSynapse(E, E, :ge; σ=10, p = 0.02)
# EG = SNN.SpikeRateSynapse(E, G; σ = 1., p = 1.0)
# SNN.monitor(E, [:fire])
# SNN.sim!(P, C; duration = 4second)
# SNN.raster([E], [900, 1000])
# plot!(xlims=(100,1000))
    