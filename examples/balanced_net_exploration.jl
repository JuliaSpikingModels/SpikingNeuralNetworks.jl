using Revise
using DrWatson
@quickactivate "SpikingNeuralNetworks"
using SpikingNeuralNetworks
SNN.@load_units
import SpikingNeuralNetworks: AdExParameter, IFParameter
using Statistics, Random
using Plots

## Neuron parameters
τm = 20ms
C = 300SNN.pF # Capacitance
R = τm / C
τre = 1ms # Rise time for excitatory synapses
τde = 6ms # Decay time for excitatory synapses
τri = 0.5ms # Rise time for inhibitory synapses 
τdi = 2ms # Decay time for inhibitory synapses

# Input and synapse paramater
N = 1000
νe = 4.5Hz # Rate of external input to E neurons 
νi = 2.25Hz # Rate of external input to I neurons 
p_in = 1.0 #1.0 # 0.5 
σ_in_E = 1.78SNN.pF

σEE = 2.76SNN.pF # Initial E to E synaptic weight
σIE = 48.7SNN.pF # Initial I to E synaptic weight
σEI = 1.27SNN.pF # Synaptic weight from E to I
σII = 16.2SNN.pF # Synaptic weight from I to I

Random.seed!(28)
duration = 700ms
pltdur = 70e1

LKD_AdEx_exc = 
    AdExParameter(τm = 20ms, Vt = -52mV, Vr = -60mV, El = -70mV, R = R, ΔT = 2mV, a=4nS,
    b = 0.805SNN.pA, τabs = 1ms, τw = 150ms, τre = τre, τde = τde, τri = τri, τdi = τdi, At = 10mV, τT = 30ms, E_i=-75mV, E_e = 0mV) #  0.000805nA

E = SNN.AdEx(; N = 100, param = LKD_AdEx_exc)

Input_E = SNN.Poisson(; N = N, param = SNN.PoissonParameter(; rate = νe))
ProjE = SNN.SpikingSynapse(Input_E, E, :ge; σ = σ_in_E, p = p_in) # connection from input to E

LKD_IF_inh =
    IFParameter(τm = 20ms, Vt = -52mV, Vr = -60mV, El = -62mV, R = R, τre = τre, τde = τde, τri = τri, τdi = τdi, E_i = -75mV, E_e=0mV)

I = SNN.IF(; N = 100, param = LKD_IF_inh)

EE = SNN.SpikingSynapse(E, E, :ge; σ = σEE, p = 0.2, param=SNN.vSTDPParameter()) 
EI = SNN.SpikingSynapse(E, I, :ge; σ = σEI, p = 0.2)
IE = SNN.SpikingSynapse(I, E, :gi; σ = σIE, p = 0.2, param=SNN.iSTDPParameter())
II = SNN.SpikingSynapse(I, I, :gi; σ = σII, p = 0.2)

Input_I = SNN.Poisson(; N = N, param = SNN.PoissonParameter(; rate = νi))
ProjI = SNN.SpikingSynapse(Input_I, I, :ge; σ = σ_in_E, p = p_in)

#
P = [E, I, Input_E, Input_I]
C = [EE, II, EI, IE, ProjE, ProjI]

#
SNN.monitor([E, I], [:fire]) 
SNN.sim!(P, C; duration = duration)
SNN.raster([E, I], [0, 1] .* pltdur)

function firing_frequency(E, I, bin_width, bin_edges)
    E_neuron_spikes = map(sum, E.records[:fire])./length(E.records[:fire]) # avg spikes at each time step
    I_neuron_spikes = map(sum, I.records[:fire])./length(E.records[:fire]) # avg spikes at each time step

    # Count the number of spikes in each bin
    E_bin_count = [sum(E_neuron_spikes[i:i+bin_width-1]) for i in bin_edges]
    I_bin_count = [sum(I_neuron_spikes[i:i+bin_width-1]) for i in bin_edges]

    return E_bin_count, I_bin_count
end 

frequencies = [1Hz 10Hz 100Hz]
E_bin_counts = []
I_bin_counts = []
bin_edges = []
bin_width = 10  # in milliseconds

num_bins = Int(length(E.records[:fire]) / bin_width)
bin_edges = 1:bin_width:(num_bins * bin_width)

for νi in frequencies
    E = SNN.AdEx(; N = 100, param = LKD_AdEx_exc)
    I = SNN.IF(; N = 100, param = LKD_IF_inh)

    EE = SNN.SpikingSynapse(E, E, :ge; σ = σEE, p = 0.2, param=SNN.vSTDPParameter()) 
    EI = SNN.SpikingSynapse(E, I, :ge; σ = σEI, p = 0.2)
    IE = SNN.SpikingSynapse(I, E, :gi; σ = σIE, p = 0.2, param=SNN.iSTDPParameter())
    II = SNN.SpikingSynapse(I, I, :gi; σ = σII, p = 0.2)

    Input_E = SNN.Poisson(; N = N, param = SNN.PoissonParameter(; rate = νe))
    ProjE = SNN.SpikingSynapse(Input_E, E, :ge; σ = σ_in_E, p = p_in) # connection from input to E  
    Input_I = SNN.Poisson(; N = N, param = SNN.PoissonParameter(; rate = νi))
    ProjI = SNN.SpikingSynapse(Input_I, I, :ge; σ = σ_in_E, p = p_in)

    #
    P = [E, I, Input_E, Input_I]
    C = [EI, IE, ProjE, ProjI]

    #
    SNN.monitor([E, I], [:fire]) 
    SNN.sim!(P, C; duration = duration)
    
    E_bin_count, I_bin_count = firing_frequency(E, I, bin_width, bin_edges)
    push!(E_bin_counts, E_bin_count)
    push!(I_bin_counts, I_bin_count)
end

# Create a new plot or use an existing plot if it exists
plot(xlabel="Time bins", size=(800, 800), ylabel="Firing frequency (spikes/$(bin_width) ms)", 
    xtickfontsize=6, ytickfontsize=6, yguidefontsize=6, xguidefontsize=6, titlefontsize=7,
    legend=:bottomright, layout=(length(frequencies), 1), title=["νi = $(νi*1000)Hz" for νi in frequencies])

# Plot excitatory neurons
plot!(bin_edges, E_bin_counts, label="Excitatory neurons")
# Plot inhibitory neurons
plot!(bin_edges, I_bin_counts, label="Inhibitory neurons")
