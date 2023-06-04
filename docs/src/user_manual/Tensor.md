# `Tensor.jl`

```julia
mutable struct Tensor
    val::AbstractArray  # Value of the Tensor
    parents::Vector     # Who created this Tensor
    chain_rules::Vector # Chain Rule to get global gradients wrt parents

    # Constructor to initialise the Tensor (for leaf nodes)
    function Tensor(val::Number) # If input is a scalar
        val = [val][:,:] # Storing Scalar as an Array
        new(val, [], [])
    end

    function Tensor(val::AbstractArray) # If input is a matrix
        new(val, [], [])
    end


    # Constructor to create Tensor using operations (for intermediate nodes)
    function Tensor(val::Number, parents::Vector{Tensor}, chain_rules::Vector) # If input is a number
        val = [val][:,:] # Storing Scalar as an Array
        new(val, parents, chain_rules)
    end

    function Tensor(val::AbstractArray, parents::Vector{Tensor}, chain_rules::Vector) # If input is a matrix
        new(val, parents, chain_rules)
    end
end
```