mutable struct Tensor
    val::AbstractArray  # Value of the Tensor
    parents::Vector     # Who created this Tensor
    chain_rules::Vector # Chain Rule to get global gradients wrt parents

    # Constructor to initialise the Tensor (for leaf nodes)
    function Tensor(val::Number)
        val = [val][:,:] # Storing Scalar as an Array
        new(val, [], [])
    end

    function Tensor(val::AbstractArray)
        new(val, [], [])
    end


    # Constructor to create Tensor using operations (for intermediate nodes)
    function Tensor(val::Number, parents::Vector{Tensor}, chain_rules::Vector)
        val = [val][:,:] # Storing Scalar as an Array
        new(val, parents, chain_rules)
    end

    function Tensor(val::AbstractArray, parents::Vector{Tensor}, chain_rules::Vector)
        new(val, parents, chain_rules)
    end
end