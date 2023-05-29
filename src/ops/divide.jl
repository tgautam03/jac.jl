# Matrix Scalar Division
function Base.:/(T1::Tensor, T2::Number)
    val = T1.val / T2
    parents = [T1]

    # Chain Rule Functions defining global gradients wrt parents
    dT1 = global_grad::Tensor -> global_grad .* (1/T2)
    chain_rules = [dT1]

    return Tensor(val, parents, chain_rules)
end