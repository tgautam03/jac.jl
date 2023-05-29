using Test
using jac

function test_grad()
    a = Tensor(4)
    b = Tensor(3)
    c = a + b
    d = a * c

    ∂d_∂a = grad(d)[a]

    ∂2d_∂a2 = grad(∂d_∂a)[a]

    # println(∂d_∂a.val)
    # println(∂2d_∂a2.val)

    return ∂d_∂a.val[1,1] == 11 && ∂2d_∂a2.val[1,1] == 2
end


function test_linregress()
    # Getting Dataset
    n = 10
    x = rand(n, 1)
    w_true = rand(1, 1)
    b_true = rand(1, 1)
    y = x * w_true .+ b_true + rand(n, 1)*0.1

    # JAC 
    X = Tensor(x)
    Y = Tensor(y)

    W = Tensor(rand(1,1)*0.001)
    B = Tensor(rand(1,1)*0.001)

    Z = X * W .+ B
    z = x*W.val .+ B.val
    
    L = sum((Z - Y)^2, 1) / n
    l = sum((z - y).^2, dims=1) / n

    ∂L = grad(L)

    ∂L_∂W = ∂L[W]
    ∂l_∂W = sum(2 .* (z - y) .* x, dims=1) / n
    
    ∂L_∂B = ∂L[B]
    ∂l_∂B = sum(2 .* (z - y), dims=1) / n

    # println(∂L_∂W.val)
    # println(∂l_∂W)
    # println(∂L_∂B.val)
    # println(∂l_∂B)

    return ∂L_∂W.val ≈ ∂l_∂W && ∂L_∂B.val ≈ ∂l_∂B
end

# @test test_grad()
@test test_linregress()