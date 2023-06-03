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

    linear(X, W, B) = X * W .+ B
    mse(Z, Y) = sum((Z - Y)^2, 1) / size(Z.val)[1]

    for i=0:10000
        # Forward
        Z = linear(X, W, B)
        # Loss
        L = mse(Z, Y)
        # Gradients
        ∂L = grad(L)

        if i % 100 == 0
            println("Iteration $(i); MSE $(mse(linear(X, W, B), Y).val)")
        end

        # Updating Parameters
        W.val = W.val - 0.01 * ∂L[W].val
        B.val = B.val - 0.01 * ∂L[B].val
    end

    println(W.val)
    println(w_true)
    
    println(B.val)
    println(b_true)

    return true
end

# @test test_grad()
@test test_linregress()