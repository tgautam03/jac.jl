using Test
using jac

function test_scalar()
    x = Tensor(4)
    y = Tensor(3)
    
    f1 = x * y
    f2 = y^3
    f3 = f1 + f2
    f = f3^2

    gradients = grad(f)
    ∂f_∂x = gradients[x]
    ∂f_∂y = gradients[y]

    
    return ∂f_∂x.val[1,1] == 2*(x.val[1,1]*y.val[1,1] + y.val[1,1]^3)*y.val[1,1] && ∂f_∂y.val[1,1] == 2*(x.val[1,1]*y.val[1,1] + y.val[1,1]^3)*(x.val[1,1]+3*y.val[1,1]^2)
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

@test test_scalar()
# @test test_linregress()