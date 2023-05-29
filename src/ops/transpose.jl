function Base.transpose(T::Tensor)
    val = transpose(T.val)
    parents = [T]

    dT = global_grad::Tensor -> transpose(global_grad)
    chain_rules = [dT]

    return Tensor(val, parents, chain_rules)
end