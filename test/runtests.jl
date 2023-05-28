using Test
using jac

function test_Tensor()
    vec = rand(5,1)
    scalar = Tensor(5)
    vector = Tensor(vec)

    println(scalar)
    println(vector)

    cond1 = (scalar.val == [5][:,:])
    cond2 = (vector.val == vec)

    return cond1 && cond2
end

@test test_Tensor()