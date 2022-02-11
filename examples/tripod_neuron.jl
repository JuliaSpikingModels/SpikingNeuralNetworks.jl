using Plots
using TripodSNN
using SpikingNeuralNetworks
SNN = SpikingNeuralNetworks
# SNN.@load_units
d1 = [SNN.Dendrite(;TripodSNN.create_dendrite(Float32(l))...) for l in rand(350:400, 200)]
d2 = [SNN.Dendrite(;TripodSNN.create_dendrite(Float32(l))...) for l in rand(150:400, 200)]

pyplot()
# SNN.DendriteParam(;)
# TripodSNN.create_dendrites(rand(150:400, 200))

##

using Random
Random.seed!(10)
ampa = ([:AMPA],[1])
gaba = ([:GABAa],[3])
E  = SNN.Tripod(N=1,pm1=d1, pm2=d2, param=SNN.AdExTripod(b=80SNN.nA))
inputs = [
    SNN.Poisson(N = 100,param=SNN.PoissonParameter(rate=1.40)),
    SNN.Poisson(N = 100,param=SNN.PoissonParameter(rate=1.5))
    ]
synapses = []
for d in ["d1", "d2"]
    push!(synapses, SNN.SpikingSynapseClopath(inputs[1], E, d, gaba, param=SNN.synapse_soma, p=1., σ=1.5))
    push!(synapses, SNN.SpikingSynapseClopath(inputs[2], E, d, ampa, param=SNN.synapse_soma, p=1., σ=5.9))
end
##
SNN.monitor(E, [:v_s, :v_d1,:v_d2, :fire, :h_s, :g_d2, :g_d1, :after_spike])
SNN.monitor(E, [
			:i1,
			:i2,
			:is,
			:c1,
			:c2,
			:Δs,
			:Δd1,
			:Δd2,
			])
SNN.sim!([E,inputs...], [synapses...]; duration = 1000SNN.ms)
plot(SNN.vecplot(E, :v_s),SNN.vecplot(E, :v_d2), SNN.vecplot(E,:v_d1), layout =(3,1))
##
data = vcat([get_recs(E, i, :i1, :c1, :Δd1 ) for i in 1:1000]...)
get_recs(p, i::Int, args...) =  hcat([SNN.getrecord(p,arg)[i][1] for arg in args]...)

plotly()
plot(data,xlims=(425,435), labels=["i1" "c1" "d1"], )
##
plot(
    plot([x[1] for x in SNN.getrecord(E, :g_d1)]),
    plot([x[1] for x in SNN.getrecord(E, :g_d2)]),
    plot([x[3] for x in SNN.getrecord(E, :g_d1)]),
    plot([x[3] for x in SNN.getrecord(E, :g_d2)])
    )
# SNN.vecplot(E, :Isyn_d1)

##
SNN.monitor(inputs[1],[:fire])
SNN.sim!([inputs[1]],[]; duration = 50SNN.ms)
SNN.raster([inputs[1]], ylims=(1,100), markersize=20)

using PyPlot
pygui(true)
show()