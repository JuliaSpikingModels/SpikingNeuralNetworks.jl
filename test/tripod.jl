using SpikingNeuralNetworks
using Test
SNN.@load_units;


d1 = [SNN.Dendrite()]
d2 = [SNN.Dendrite()]
N = 1
E = SNN.Tripod(
    N=N, d1=d1, d2=d2, 
    soma_syn=SNN.synapsearray(Synapse(DuarteGluSoma, MilesGabaSoma), [1,3]),
    dend_syn = SNN.synapsearray(Synapse(EyalGluDend, MilesGabaDend)),
    param = SNN.AdExTripod(Er=-65mV))


SNN.synapsearray(Synapse(DuarteGluSoma, MilesGabaDend))
pre = (fire=falses(1), N=1)
w = 20*ones(1,1)
projection_exc_dend = SNN.SynapseTripod(pre, E, "d1", "exc", w=w)
projection_inh_dend = SNN.SynapseTripod(pre, E, "d1", "inh", w=w)
projection_exc_soma = SNN.SynapseTripod(pre, E, "s", "exc", w=w)
projection_inh_soma = SNN.SynapseTripod(pre, E, "s", "inh", w=w)


projections=[ 
projection_exc_dend ,
projection_inh_dend ,
# projection_exc_soma,
# projection_inh_soma,
]

projection_inh_dend.αs

SNN.synapsearray(Synapse(EyalGluDend, MilesGabaDend))
soma_syn=SNN.synapsearray(Synapse(DuarteGluSoma, MilesGabaSoma))
soma_syn[2].type

SNN.sim!([E], projections, duration=1000)

SNN.monitor(E, [:v_s, :v_d1, :g_d1, :fire, :g_s])
SNN.sim!([E], projections, duration=50)
for p in projections
    p.fireJ[1]= true
end
SNN.sim!([E], projections, duration=0.1f0)
for p in projections
    p.fireJ[1]= false
end
SNN.sim!([E], projections, duration=200)
using Plots
plot(
plot(vcat(E.records[:g_d1]...)[:,1:2], ylabel="g_d exc", labels=["AMPA" "NMDA"]),
plot(vcat(E.records[:g_d1]...)[:,3:4], ylabel="g_d inh", labels=["GABAa" "GABAb"]),
SNN.vecplot(E, :v_d1),
layout=(3,1),
)