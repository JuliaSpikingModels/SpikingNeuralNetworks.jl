function plasticity!(c::AbstractSparseSynapse, param::no_STDPParameter, dt::Float32) end

"""
    plasticity!(c::AbstractSparseSynapse, param::iSTDPParameterRate, dt::Float32)

Performs the synaptic plasticity calculation based on the inihibitory spike-timing dependent plasticity (iSTDP) model from Vogels (2011). 
The function updates synaptic weights `W` of each synapse in the network according to the firing status of pre and post-synaptic neurons.
This is an in-place operation that modifies the input `AbstractSparseSynapse` object `c`.

# Arguments
- `c::AbstractSparseSynapse`: The current spiking synapse object which contains data structures to represent the synapse network.
- `param::iSTDPParameterRate`: Parameters needed for the iSTDP model, including learning rate `η`, target rate `r`, STDP time constant `τy`, maximal and minimal synaptic weight (`Wmax` and `Wmin`).
- `dt::Float32`: The time step for the numerical integration.

# Algorithm
- For each pre-synaptic neuron, if it fires, it reduces the synaptic weight by an amount proportional to the difference between the target rate and the actual rate and increases the inhibitory term, 
  otherwise the inhibitory term decays exponentially over time with a time constant `τy`.
- For each post-synaptic neuron, if it fires, it increases the synaptic weight by an amount proportional to the pre-synaptic trace and increases the excitatory term,
  otherwise the excitatory term decays exponentially over time with a time constant `τy`.
- The synaptic weights are bounded by `Wmin` and `Wmax`.
"""
##
function plasticity!(c::AbstractSparseSynapse, param::iSTDPParameterRate, dt::Float32)
    @unpack rowptr, colptr, index, I, J, W, tpost, tpre, fireI, fireJ, g = c
    @unpack η, r, τy, Wmax, Wmin = param

    # @inbounds 
    # if pre-synaptic inhibitory neuron fires
    for j in eachindex(fireJ) # presynaptic indices j
        tpre[j] += dt * (-tpre[j]) / τy
        if fireJ[j] # presynaptic neuron
            tpre[j] += 1
            for st = colptr[j]:(colptr[j+1]-1)
                W[st] = clamp(W[st] + η * (tpost[I[st]] - 2 * r * Hz * τy), Wmin, Wmax)
            end
        end
    end
    # if post-synaptic excitatory neuron fires
    # @inbounds 
    for i in eachindex(fireI) # postsynaptic indices i
        tpost[i] += dt * (-tpost[i]) / τy
        if fireI[i] # postsynaptic neuron
            tpost[i] += 1
            for st = rowptr[i]:(rowptr[i+1]-1) ## 
                st = index[st]
                W[st] = clamp(W[st] + η * tpre[J[st]], Wmin, Wmax)
            end
        end
    end
end

"""
    plasticity!(c::AbstractSparseSynapse, param::iSTDPParameterRate, dt::Float32)

Performs the synaptic plasticity calculation based on the inihibitory spike-timing dependent plasticity (iSTDP) model from Vogels (2011) adapted to control the membrane potential. 
The function updates synaptic weights `W` of each synapse in the network according to the firing status of pre and post-synaptic neurons.
This is an in-place operation that modifies the input `AbstractSparseSynapse` object `c`.

# Arguments
- `c::AbstractSparseSynapse`: The current spiking synapse object which contains data structures to represent the synapse network.
- `param::iSTDPParameterVoltage`: Parameters needed for the iSTDP model, including learning rate `η`, target membrane potenital `v0`, STDP time constant `τy`, maximal and minimal synaptic weight (`Wmax` and `Wmin`).
- `dt::Float32`: The time step for the numerical integration.

# Algorithm
- For each pre-synaptic neuron, if it fires, it reduces the synaptic weight by an amount proportional to the difference between the target membrane potential and the actual one. 
- For each pre-synaptic neuron, if it fires, increases the inhibitory term, otherwise the inhibitory term decays exponentially over time with a time constant `τy`.
- For each post-synaptic neuron, if it fires, it increases the synaptic weight by an amount proportional to the pre-synaptic trace and increases the excitatory term, otherwise the excitatory term decays exponentially over time with a time constant `τy`.
- The synaptic weights are bounded by `Wmin` and `Wmax`.
"""
function plasticity!(c::AbstractSparseSynapse, param::iSTDPParameterPotential, dt::Float32)
    @unpack rowptr, colptr, index, I, J, W, v_post, tpost, tpre, fireI, fireJ, g = c
    @unpack η, v0, τy, Wmax, Wmin = param

    # @inbounds 
    # if pre-synaptic inhibitory neuron fires
    @fastmath @inbounds for j in eachindex(fireJ) # presynaptic indices j
        tpre[j] += dt * (-tpre[j]) / τy
        if fireJ[j] # presynaptic neuron
            tpre[j] += 1
            for st = colptr[j]:(colptr[j+1]-1)
                W[st] = clamp(W[st] + η * (tpost[I[st]] - v0), Wmin, Wmax)
            end
        end
    end
    # if post-synaptic excitatory neuron fires
    @fastmath @inbounds for i in eachindex(fireI) # postsynaptic indices i
        # trace of the membrane potential
        tpost[i] += dt * -(tpost[i] - v_post[i]) / τy
        if fireI[i] # postsynaptic neuron
            for st = rowptr[i]:(rowptr[i+1]-1) ## 
                st = index[st]
                W[st] = clamp(W[st] + η * tpre[J[st]], Wmin, Wmax)
            end
        end
    end
end



#
# function plasticity!(
#     c::AbstractSparseSynapse,
#     param::STDPParameter,
#     dt::Float32,
# )
#     @unpack rowptr, colptr, I, J, index, W, tpre, tpost, Apre, Apost, fireI, fireJ, g = c
#     @unpack τpre, τpost, Wmax, ΔApre, ΔApost = param
#     @inbounds for j = 1:(length(colptr)-1)
#         if fireJ[j]
#             for s = colptr[j]:(colptr[j+1]-1)
#                 Apre[s] *= exp32(-(t - tpre[s]) / τpre)
#                 Apost[s] *= exp32(-(t - tpost[s]) / τpost)
#                 Apre[s] += ΔApre
#                 tpre[s] = t
#                 W[s] = clamp(W[s] + Apost[s], 0.0f0, Wmax)
#             end
#         end
#     end
#     @inbounds for i = 1:(length(rowptr)-1)
#         if fireI[i]
#             for st = rowptr[i]:(rowptr[i+1]-1)
#                 s = index[st]
#                 Apre[s] *= exp32(-(t - tpre[s]) / τpre)
#                 Apost[s] *= exp32(-(t - tpost[s]) / τpost)
#                 Apost[s] += ΔApost
#                 tpost[s] = t
#                 W[s] = clamp(W[s] + Apre[s], 0.0f0, Wmax)
#             end
#         end
#     end
# end

"""
    plasticity!(c::AbstractSparseSynapse, param::vSTDPParameter, dt::Float32)

Perform update of synapses using plasticity rules based on the Spike Timing Dependent Plasticity (STDP) model.
This function updates pre-synaptic spike traces and post-synaptic membrane traces, and modifies synaptic weights using vSTDP rules.

# Arguments
- `c::AbstractSparseSynapse`: The spiking synapse to be updated.
- `param::vSTDPParameter`: Contains STDP parameters including A_LTD, A_LTP, θ_LTD, θ_LTP, τu, τv, τx, Wmax, Wmin.
    - `A_LTD`: Long Term Depression learning rate.
    - `A_LTP`: Long Term Potentiation learning rate.
    - `θ_LTD`: LTD threshold.
    - `θ_LTP`: LTP threshold.
    - `τu, τv, τx`: Time constants for different variables in STDP.
    - `Wmax, Wmin`: Maximum and minimum synaptic weight.
- `dt::Float32`: Time step for simulation.

In addition to these, the function uses normalization where the operator can be multiplicative or additive as defined by `c.normalize.param.operator`.
The `operator` is applied when updating the synaptic weights. The frequency of normalization is controlled by `τ`, 
where if `τ > 0.0f0` then normalization will occur at intervals approximately equal to `τ`.

After all updates, the synaptic weights are clamped between `Wmin` and `Wmax`.

"""
function plasticity!(c::AbstractSparseSynapse, param::vSTDPParameter, dt::Float32)
    @unpack rowptr, colptr, I, J, index, W, u, v, x, v_post, fireI, fireJ, g, index = c
    @unpack A_LTD, A_LTP, θ_LTD, θ_LTP, τu, τv, τx, Wmax, Wmin = param
    R(x::Float32) = x < 0.0f0 ? 0.0f0 : x

    # update pre-synaptic spike trace
    @simd for j in eachindex(fireJ) # Iterate over all columns, j: presynaptic neuron
        @inbounds @fastmath x[j] += dt * (-x[j] + fireJ[j]) / τx
    end


    @inbounds @fastmath for i in eachindex(fireI) # Iterate over postsynaptic neurons
        u[i] += dt * (-u[i] + v_post[i]) / τu # postsynaptic neuron
        v[i] += dt * (-v[i] + v_post[i]) / τv # postsynaptic neuron
        ltd_u = u[i] - θ_LTD
        ltd_v = v[i] - θ_LTD
        ltp = v_post[i] - θ_LTP
        # @simd for s = colptr[j]:(colptr[j+1]-1) 

        @simd for s = rowptr[i]:(rowptr[i+1]-1)
            j = J[index[s]]
            if ltd_u > 0.0f0 && fireJ[j]
                W[index[s]] += -A_LTD * ltd_u
            end
            if ltp > 0.0f0 && ltd_v > 0.0f0
                W[index[s]] += A_LTP * x[j] * ltp * ltd_v
            end
        end
    end

    @inbounds @simd for i in eachindex(W)
        W[i] = clamp.(W[i], Wmin, Wmax)
    end
end



    # @inbounds @fastmath @simd for i in eachindex(fireI) # Iterate over postsynaptic neurons
    #     u[i] += dt * (-u[i] + v_post[i]) / τu # postsynaptic neuron
    #     v[i] += dt * (-v[i] + v_post[i]) / τv # postsynaptic neuron
    # end
    # Threads.@threads for j in eachindex(fireJ) # Iterate over presynaptic neurons
    #     # @simd for s = colptr[j]:(colptr[j+1]-1) 
    #     @inbounds @fastmath @simd for s = colptr[j]:(colptr[j+1]-1)
    #         i = I[s] # get_postsynaptic cell
    #         ltd_u = R(u[i] - θ_LTD)
    #         if fireJ[j] && ltd_u >0
    #             W[s] += -A_LTD * ltd_u
    #         end
    #         ltd_v = v[i] - θ_LTD
    #         ltp = v_post[i] - θ_LTP
    #         if ltp > 0.0f0 && ltd_v > 0.0f0
    #             W[index[s]] += A_LTP * x[j] * ltp * ltd_v
    #         end
    #     end
    # end
