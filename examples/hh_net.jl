using Plots
using SpikingNeuralNetworks
SNN.@load_units

E = SNN.HH(; N = 3200)
I = SNN.HH(; N = 800)
EE = SNN.SpikingSynapse(E, E, :ge; σ = 6nS, p = 0.02)
EI = SNN.SpikingSynapse(E, I, :ge; σ = 6nS, p = 0.02)
IE = SNN.SpikingSynapse(I, E, :gi; σ = 67nS, p = 0.02)
II = SNN.SpikingSynapse(I, I, :gi; σ = 67nS, p = 0.02)
P = [E, I]
C = [EE, EI, IE, II]

SNN.monitor(E, [(:v, [1, 10, 100])])
SNN.sim!(P, C; dt = 0.01ms, duration = 100ms)
SNN.vecplot(E, :v)

##

xx = 10 .^range(-6,4,11)
yy = 1 .^range(-6,1,8)
rectangle(x1,x2,y1,y2) = Shape([x1,x2,x2,x1], [y1,y1,y2,y2])
plot(xlims=(10e-6,10e4), ylims=(1e-6,1e1), scale=:log, xticks=(xx), size=(900,600), xlabel="Time (ms)", ylabel="Space (m)", margin=10Plots.mm)
default(lw=2, guidefontsize=18, tickfontsize=15, size=(800,600))
p =  plot!()
dir = "/Users/cocconat/Downloads/scales.pdf"
isdir(dir)
vline!([0.5])
savefig(p,dir)
# plot!(rectangle(1e-5,5e-3,1e-6,1e-5), alpha=0.5,c=:grey )
# plot!(rectangle(1e-5,5e-3,1e-6,1e-5), alpha=0.5,c=:grey )
# plot!(rectangle(1e-5,5e-3,1e-6,1e-5), alpha=0.5,c=:grey )
# plot!(rectangle(2e-4,8e-2,6e-6,2e-4), alpha=0.5,c=:grey)
# plot!(rectangle(8e-3,7e-1,6e-5,6e-4), alpha=0.5,c=:grey)
# plot!(rectangle(2e-2,1.5e-0,1e-4,8e-4), alpha=0.5,c=:grey)