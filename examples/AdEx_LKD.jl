using Plots
using SpikingNeuralNetworks
SNN.@load_units
import SpikingNeuralNetworks: AdExParameter
using Statistics, Random


## LKD parameters
τm = 20ms
C = 300SNN.pF
R = τm/C

LKD_AdEx_exc = AdExParameter(τm=20ms, τe=6ms, τi=2ms, Vt=-52mV, Vr=-60mV, El=-70mV, R=R, )
LKD_AdEx_inh = AdExParameter(τm=20ms, τe=6ms, τi=2ms, Vt=-52mV, Vr=-60mV, El=-62mV, R=R, )

# inputs, the kHz is obtained by the N*ν, so doing the spikes (read eqs. 4 section)
N = 1000
νe=4.5Hz
νi=2.5Hz
p_in = 0.5
σ_in = 1.50
C = 300SNN.pF

σEE= 2.76/C*SNN.pF
σEI= 10.27/C*SNN.pF
σIE= 48.7/C*SNN.pF
σII= 16.2/C*SNN.pF
#

Input_E = SNN.Poisson(;N=N,param=SNN.PoissonParameter(;rate=νe))
Input_I = SNN.Poisson(;N=N,param=SNN.PoissonParameter(;rate=νi))
E = SNN.AdEx(; N = 2000, param = LKD_AdEx_exc)
I = SNN.AdEx(; N = 500, param = LKD_AdEx_inh)

EE = SNN.SpikingSynapse(E, E, :ge; σ = σEE, p = 0.2)
EI = SNN.SpikingSynapse(E, I, :ge; σ = σEI, p = 0.2)
IE = SNN.SpikingSynapse(I, E, :gi; σ = σIE, p = 0.2)
II = SNN.SpikingSynapse(I, I, :gi; σ = σII, p = 0.2)
ProjE = SNN.SpikingSynapse(Input_E,E, :ge; σ=σ_in, p=p_in)
ProjI = SNN.SpikingSynapse(Input_I,I, :ge; σ=σ_in, p=p_in)

#3
P = [E, I, Input_E, Input_I]
C = [EE, EI, IE, II, ProjE, ProjI]
# C = [Proj]
##
Random.seed!(23)
SNN.monitor([E, I], [:fire])
SNN.sim!(P, C; duration = 15second)
bar(sum(hcat(E.records[:fire]...)./15, dims=2)[:,1])
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
