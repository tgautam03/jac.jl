# Multiply two Tensors
function Base.:*(T1::Tensor, T2::Tensor)
    val = T1.val * T2.val
    parents = [T1, T2]

    # Chain Rule Functions defining global gradients wrt parents
    dT1 = global_grad::Tensor -> global_grad * transpose(T2)
    dT2 = global_grad::Tensor -> transpose(T1) * global_grad 
    chain_rules = [dT1, dT2]

    return Tensor(val, parents, chain_rules)
end

# Elementwise Multiplication
function Base.broadcasted(::typeof(*), T1::Tensor, T2::Tensor)
    val = T1.val .* T2.val
    parents = [T1, T2]

    # Chain Rule Functions defining global gradients wrt parents
    dT1 = global_grad::Tensor -> global_grad .* T2
    dT2 = global_grad::Tensor -> global_grad .* T1
    chain_rules = [dT1, dT2]

    return Tensor(val, parents, chain_rules)
end

function Base.broadcasted(::typeof(*), T1::Tensor, T2::Number)
    val = T1.val .* T2
    parents = [T1]

    # Chain Rule Functions defining global gradients wrt parents
    dT1 = global_grad::Tensor -> global_grad .* T2
    chain_rules = [dT1]

    return Tensor(val, parents, chain_rules)
end

function Base.broadcasted(::typeof(*), T1::Number, T2::Tensor)
    val = T1 .* T2.val
    parents = [T2]

    # Chain Rule Functions defining global gradients wrt parents
    dT2 = global_grad::Tensor -> global_grad .* T1
    chain_rules = [dT2]

    return Tensor(val, parents, chain_rules)
end