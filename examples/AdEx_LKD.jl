using Plots
using SpikingNeuralNetworks
SNN.@load_units
import SpikingNeuralNetworks: AdExParameter, IFParameter
using Statistics, Random
using Logging

## LKD parameters
τre = 1ms # Rise time for excitatory synapses
τde = 6ms # Decay time for excitatory synapses
τri = 0.5ms # Rise time for inhibitory synapses 
τdi = 2ms # Decay time for inhibitory synapses

LKD_AdEx_exc = 
    AdExParameter(τm = 20ms, Vt = -52mV, Vr = -60mV, El = -70mV, R = 20ms/300SNN.pF, ΔT = 2mV, τw = 150ms, a = 4nS,
    b = 0.805SNN.pA, τabs = 1ms, τre = τre, τde = τde, τri = τri, τdi = τdi, E_i = -75mV, E_e = 0mV, At = 10mV, τT = 30ms)

LKD_IF_inh =
    IFParameter(τm = 20ms, Vt = -52mV, Vr = -60mV, El = -62mV, R = 20ms/300SNN.pF, ΔT = 2mV, τre = τre, τde = τde, τri = τri, τdi = τdi, 
    E_i = -75mV, E_e = 0mV, τabs = 1ms)

# inputs, the kHz is obtained by the N*ν, so doing the spikes (read eqs. 4 section)
N = 1000
νe = 4.5Hz # Rate of external input to E neurons
νi = 2.25Hz # Rate of external input to I neurons
p_in = 0.2 # 0.5 
σ_in_E = 1.78SNN.pF

σEE = 2.76SNN.pF # Initial E to E synaptic weight
σEI = 1.27SNN.pF # Synaptic weight from E to I
σIE = 48.7SNN.pF # Initial I to E synaptic weight
σII = 16.2SNN.pF # Synaptic weight from I to I

Input_E = SNN.Poisson(; N = N, param = SNN.PoissonParameter(; rate = νe))
Input_I = SNN.Poisson(; N = N, param = SNN.PoissonParameter(; rate = νi))
E = SNN.AdEx(; N = 4000, param = LKD_AdEx_exc)
I = SNN.IF(; N = 1000, param = LKD_IF_inh)

# EE = SNN.SpikingSynapse(E, E, :ge; σ = σEE, p = 0.2, param=SNN.vSTDPParameter()) 
# EI = SNN.SpikingSynapse(E, I, :ge; σ = σEI, p = 0.2)
# IE = SNN.SpikingSynapse(I, E, :gi; σ = σIE, p = 0.2, param=SNN.iSTDPParameter())
# II = SNN.SpikingSynapse(I, I, :gi; σ = σII, p = 0.2)

ProjE = SNN.SpikingSynapse(Input_E, E, :ge; σ = σ_in_E, p = p_in)
ProjI = SNN.SpikingSynapse(Input_I, I, :ge; σ = σ_in_E, p = p_in)

#
P = [E, I, Input_E, Input_I]
C = [EE, EI, IE, II, ProjE, ProjI]

#
Random.seed!(28)
SNN.monitor([E, I], [:fire])
SNN.train!(P, C; duration = 10ms)

# debuglogger = ConsoleLogger(stderr, Logging.Debug)
# with_logger(debuglogger) do
#     @debug "Start the simulation"
#     # SNN.sim!(P, C; duration = 1second)
#     SNN.train!(P, C; duration = 1second)
# end
# ##
# bar(sum(hcat(E.records[:fire]...) ./ 1, dims = 2)[:, 1])
# SNN.raster([E, I], [9, 11] .* 10e3)
SNN.raster([E, I], [0, 1] .* 10e1)
    
