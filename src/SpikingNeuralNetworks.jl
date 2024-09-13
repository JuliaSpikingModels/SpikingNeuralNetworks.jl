module SpikingNeuralNetworks


SNN = SpikingNeuralNetworks
export SNN

using LinearAlgebra
using SparseArrays
using Requires
using UnPack
using Random
using Logging
using StaticArrays
using ProgressBars

include("unit.jl")
include("main.jl")
include("util.jl")
include("structs.jl")
include("analysis.jl")

include("neuron/if.jl")
include("neuron/if_constant_input.jl")
include("neuron/if2.jl")
include("neuron/AdEx.jl")
include("neuron/AdEx_constant_input.jl")
include("neuron/noisy_if.jl")
include("neuron/poisson.jl")
include("neuron/iz.jl")
include("neuron/hh.jl")
include("neuron/rate.jl")

include("synapse/rate_synapse.jl")
include("synapse/fl_synapse.jl")
include("synapse/fl_sparse_synapse.jl")
include("synapse/pinning_synapse.jl")
include("synapse/pinning_sparse_synapse.jl")
include("synapse/spike_rate_synapse.jl")
include("synapse/spiking_synapses/base.jl")
include("neuron/Tripod.jl")

include("network/background.jl")
include("network/inputs.jl")



function __init__()
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("plot.jl")
end

end
