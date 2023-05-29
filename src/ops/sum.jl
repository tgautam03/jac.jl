function Base.sum(T::Tensor)
    val = sum(T.val)
    parents = [T]

    dT = global_grad::Tensor -> global_grad #.* Tensor(ones(size(T.val)))
    chain_rules = [dT]

    return Tensor(val, parents, chain_rules)
end

# Sum along an axis
function Base.sum(T::Tensor, dims::Int)
    val = sum(T.val, dims=dims)
    parents = [T]

    # Chain Rule Functions defining global gradients wrt parents
    dT = global_grad::Tensor -> global_grad #.* Tensor(ones(size(T.val)))
    chain_rules = [dT]

    return Tensor(val, parents, chain_rules)
end