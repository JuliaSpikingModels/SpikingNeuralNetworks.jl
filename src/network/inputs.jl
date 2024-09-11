"""
    set_input_rate!(inputs_pop, n, rate, baseline = 5Hz)

Set the input rate for each index of `inputs_pop`. If the index matches the given `n`, 
the corresponding rate will be set to `rate`. All other indices will have their rates 
set to `baseline`.

# Arguments
- `inputs_pop`: The population of inputs.
- `n`: The specific index in `inputs_pop` for which the rate should be set to `rate`.
- `rate`: The desired rate for the nth input.
- `baseline`: The rate to set for all other inputs. Defaults to `5Hz`.

# Usage

`set_input_rate!(inputs_pop, 3, 50Hz)`

This will set the rate of the 3rd input to `50Hz`, and all others to `5Hz`.
"""
function set_input_rate!(inputs_pop, n, rate, baseline = 5Hz)
    for i in eachindex(inputs_pop)
        if i == n
            inputs_pop[i].rate .= rate
        else
            inputs_pop[i].rate .= baseline
        end
    end
end


export set_input_rate!