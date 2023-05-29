# Addition
function Base.:+(T1::Tensor, T2::Tensor)
    val = T1.val + T2.val
    parents = [T1, T2]

    # Chain Rule Functions defining global gradients wrt parents
    dT1 = global_grad::Tensor -> global_grad 
    dT2 = global_grad::Tensor -> global_grad
    chain_rules = [dT1, dT2]

    return Tensor(val, parents, chain_rules)
end

# Broadcast Addition
function Base.broadcasted(::typeof(+), T1::Tensor, T2::Tensor)
    @assert size(T2.val)[1] == 1 # Making sure that T2 is a row vector

    val = T1.val .+ T2.val
    parents = [T1, T2]

    # Chain Rule Functions defining global gradients wrt parents
    dT1 = global_grad::Tensor -> global_grad #.* Tensor(ones(size(T1.val)))
    dT2 = global_grad::Tensor -> sum(global_grad, 1)
    chain_rules = [dT1, dT2]

    return Tensor(val, parents, chain_rules)
end