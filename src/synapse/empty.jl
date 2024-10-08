
@snn_kw struct EmptyParam
    type::Symbol = :empty
end

@snn_kw struct EmptySynapse <: AbstractSynapse
    param::EmptyParam=EmptyParam()
    records::Dict= Dict()
end

function forward!(p::EmptySynapse, param::EmptyParam)
end