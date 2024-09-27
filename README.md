# SpikingNeuralNetworks

## Installation

```julia
using Pkg
pkg"dev SpikingNeuralNetworks"
```

## Documentation

Spiking Neural Network library for Julia.

The library allows us to define and simulate models from computational neuroscience easily. 
The library exposes two functions:

```julia
function sim!(p::Vector{AbstractNeuron}, c::Vector{AbstractSynapse}, duration<:Real) end
function train!(p::Vector{AbstractSynapse}, c:Vector{AbstractSynapse}, duration<:Real) end
```

The functions support simulation with and without neural plasticity; the model is defined within the arguments passed to the functions. 
Models are composed of 'AbstractNeuron' and 'AbstractSynapse' arrays. 

Any elements of `AbstractNeuron` must implement the method: 
```julia
function integrate!(p, p.param, dt) end
```

Conversely, elements of `AbstractSynapse` must implement the methods: 

```julia
function forward!(p, p.param) end
function plasticity!(c, c.param, dt) end
```

The library is rich in examples of common neuron models that can be used as a basis. 

In the notebook folder, there is a tutorial about how to use SparseMatrices in the SNN framework.
