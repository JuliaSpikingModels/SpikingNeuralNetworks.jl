##
using DrWatson
@quickactivate "eqs_network"
using Plots
using SNNUtils
using SpikingNeuralNetworks
SNN.@load_units
##
##
# Create vectors of dendritic parameters
# Set the synapses (this could be done also in the )
N = 1
d1 = [SNN.Dendrite(;SNNUtils.create_dendrite(l)...) for l in rand(350:400, N)]
d2 = [SNN.Dendrite(;SNNUtils.create_dendrite(l)...) for l in rand(150:300, N)]
SNN.Dendrite(;SNNUtils.create_dendrite(100)...)
E = SNN.Tripod(N=N, d1=d1, d2=d2)  #is it one soma with 2*200 dendrites??
inputs = [
    SNN.Poisson(N = 100,param=SNN.PoissonParameter(rate=.5)),
    SNN.Poisson(N = 100,param=SNN.PoissonParameter(rate=.05))
    ]
# synapses = []
for d in ["d1", "d2"]
    push!(synapses,SpikingNeuralNetworks.SynapseTripod(inputs[1], E, d, "exc", p=1., σ=50.))
    push!(synapses,SpikingNeuralNetworks.SynapseTripod(inputs[2], E, d, "inh", p=1.0, σ=10.))
end

synapses

recurrent =  SpikingNeuralNetworks.SynapseTripod(E, E, "d1", "exc", p=0.2, σ=0.)
#
SNN.monitor(E, [:v_s, :v_d1,:v_d2, :fire, :h_s, :g_d2, :g_d1, :after_spike])
mont_idx = [round(Int,i*N/10) for i=4:2:10]
mont_idx=[1]
# mont = [(s,mont_idx) for s in [:v_s, :v_d1,:v_d2]]
# SNN.monitor(E, mont)
SNN.monitor(E, [:fire])
#SNN.sim!([E,inputs...], [synapses...]; duration = 50ms)

E.v_d1
#
E.v_d1.=-50.f0
E.v_d2.=-50.f0
E.v_s .= -50.f0

recurrent.W
@error "Start sim"

##
SNN.integrate!(E, E.param, Float32(0.1))
# SNN.sim!([E,inputs...],[recurrent,synapses...];dt=0.1ms)

# plot(SNN.vecplot(E, :v_s),SNN.vecplot(E, :v_d2), SNN.vecplot(E,:v_d1), layout =(3,1))
# SNN.raster([E])
# plot(SNN.vecplot(E, :g_d1),SNN.vecplot(E, :g_d2))
# E.records[:g_d2][end]