# Subtraction
function Base.:-(T1::Tensor, T2::Tensor)
    val = T1.val - T2.val
    parents = [T1, T2]

    # Chain Rule Functions defining global gradients wrt parents
    dT1 = global_grad::Tensor -> global_grad 
    dT2 = global_grad::Tensor -> global_grad .* (-1)
    chain_rules = [dT1, dT2]

    return Tensor(val, parents, chain_rules)
end