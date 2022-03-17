using Revise
using SNNUtils
using Plots
include("/home/cocconat/Documents/Research/phd_project/models/spiking/SpikingNeuralNetworks.jl/src/SpikingNeuralNetworks.jl")
using .SpikingNeuralNetworks
SNN = SpikingNeuralNetworks
SNN.@load_units
##
# Create vectors of dendritic parameters
d1 = [SNN.Dendrite(;SNNUtils.create_dendrite(l)...) for l in rand(350:400, 200)]
d2 = [SNN.Dendrite(;SNNUtils.create_dendrite(l)...) for l in rand(150:300, 200)]
# Set the synapses (this could be done also in the )
##
E = SNN.Tripod(N=1, d1=d1, d2=d2)
inputs = [
    SNN.Poisson(N = 100,param=SNN.PoissonParameter(rate=.06)),
    SNN.Poisson(N = 100,param=SNN.PoissonParameter(rate=.5))
    ]
synapses = []
for d in ["d1", "d2"]
    push!(synapses,SpikingNeuralNetworks.SynapseTripod(inputs[1], E, d, "exc", p=1., σ=50.))
    push!(synapses,SpikingNeuralNetworks.SynapseTripod(inputs[2], E, d, "inh", p=1.0, σ=10.))
end

recurrent =  SpikingNeuralNetworks.SynapseTripod(E, E, "d1", "exc", p=0.2, σ=0.)
# ##
SNN.monitor(E, [:v_s, :v_d1,:v_d2, :fire, :h_s, :g_d2, :g_d1, :after_spike])
# SNN.monitor(E, [
# 			:i1,
# 			:i2,
# 			:is,
# 			:c1,
# 			:c2,
# 			:Δs,
# 			:Δd1,
# 			:Δd2,
# 			])
SNN.sim!([E,inputs...], [synapses...]; duration = 1500ms)


plot(SNN.vecplot(E, :v_s),SNN.vecplot(E, :v_d2), SNN.vecplot(E,:v_d1), layout =(3,1))
##
d1[1]
# SNN.vecplot(E, :i1)
# ,SNN.vecplot(E, :v_d2), SNN.vecplot(E,:v_d1), layout =(3,1))
# ##
# data = vcat([get_recs(E, i, :i1, :c1, :Δd1 ) for i in 1:1000]...)
# get_recs(p, i::Int, args...) =  hcat([SNN.getrecord(p,arg)[i][1] for arg in args]...)
#
# plotly()
# plot(data,xlims=(425,435), labels=["i1" "c1" "d1"], )
# ##
# plot(
#     plot([x[1] for x in SNN.getrecord(E, :g_d1)]),
#     plot([x[1] for x in SNN.getrecord(E, :g_d2)]),
#     plot([x[3] for x in SNN.getrecord(E, :g_d1)]),
#     plot([x[3] for x in SNN.getrecord(E, :g_d2)])
#     )
# # SNN.vecplot(E, :Isyn_d1)
#
# ##
# SNN.monitor(inputs[1],[:fire])
# SNN.sim!([inputs[1]],[]; duration = 50SNN.ms)
# SNN.raster([inputs[1]], ylims=(1,100), markersize=20)
#
# using PyPlot
# pygui(true)
# show()
