mutable struct Tensor
    val::AbstractArray  # Value of the Tensor
    parents::Vector     # Who created this Tensor
    chain_rules::Vector # Chain Rule to get global gradients wrt parents

    # Constructor to initialise the Tensor (for leaf nodes)
    function Tensor(val::Union{Number, AbstractArray})
        if typeof(val) <: Number
            val = [val][:,:] # Storing Scalar as an Array
            new(val, [], [])
        else
            new(val, [], [])
        end
    end

    # Constructor to create Tensor using operations (for intermediate nodes)
    function Tensor(val::Union{Number, AbstractArray}, parents::Vector{Tensor}, chain_rules::Vector)
        if typeof(val) <: Number
            val = [val][:,:] # Storing Scalar as an Array
            new(val, parents, chain_rules)
        else
            new(val, parents, chain_rules)
        end
    end
end