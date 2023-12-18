using Plots
using SpikingNeuralNetworks
SNN.@load_units
import SpikingNeuralNetworks: AdExParameter
using Statistics, Rando##

function initialize()
    E = SNN.AdEx(; N = 2000, param = AdExParameter(El = -35mV))
    I = SNN.IF(; N = 500, param = SNN.IFParameter())
    EE = SNN.SpikingSynapse(E, E, :ge; σ = 20, p = 0.02)
    EI = SNN.SpikingSynapse(E, I, :ge; σ = 40, p = 0.02)
    IE = SNN.SpikingSynapse(I, E, :gi; σ = -50, p = 0.02)
    II = SNN.SpikingSynapse(I, I, :gi; σ = -10, p = 0.02)
    P = [E, I]
    C = [EE, EI, IE, II]
    return P, C
end

##
rI = []
rE = []
Irange = 0:5:70
for x in Irange
    Random.seed!(10)
    P, C = initialize()
    E, I = P
    SNN.sim!(P, C; duration = 3second)
    SNN.monitor([E, I], [:fire, :v])
    I.I .= x
    SNN.sim!(P, C; duration = 1second)
    i = round(sum(mean(I.records[:fire])), digits = 2)
    e = round(sum(mean(E.records[:fire])), digits = 2)
    @info "E: $e Hz I: $i Hz"
    push!(rE, e)
    push!(rI, i)
end
plot(
    Irange,
    [rE, rI],
    label = ["Excitatory neurons" "Inhibitory neurons"],
    xlabel = "Input to I neurons (mA)",
    ylabel = "Firing rate (Hz)",
    lc = [:red :blue],
    lw = 4,
)
##

Random.seed!(10)
P, C = initialize()
E, I = P
SNN.monitor([E, I], [:fire, :v])
SNN.sim!(P, C; duration = 5second)
SNN.raster(P, [4 * 1000, 5 * 1000])
