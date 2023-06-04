# An example involving Scalars

## Introduction

In this post, I'll be coding up the example discussed [here](https://github.com/tgautam03/jac.jl/blob/master/README.md). Just as a recap, we have a function $f(x,y)=(xy+y^3)^2$ and task is to get the partial derivatives with respect to $x$ and $y$. This same function can also be written as a series of simpler operations 

$$f_1 = x \cdot y \tag{1}$$

$$f_2 = y^3 \tag{2}$$

$$f_3 = f_1 + f_2 \tag{3}$$

$$f(x,y) = f_3^2 \tag{4}$$

Looking at these equations, we need a way to represent the value and derivative of each node (i.e. $x$, $y$, $f_1$, $\cdots$, $f(x,y)$). To facilitate this, I have to create a custom variable(ish) that has multiple attributes. An obvious choice here is to use a `mutable struct` (we need the ability to modify the attributes of this struct) and I'll name that `Tensor`. 

## `Tensor.jl`
```julia
mutable struct Tensor
    val::AbstractArray  # Value of the Tensor
    parents::Vector     # Who created this Tensor
    chain_rules::Vector # Chain Rule to get global gradients wrt parents
end
```

- `val` stores the value of the node or variable.
- `parents` is a list that will contain the other nodes that creates this node. It'll be empty for leaf nodes $x$ and $y$.
- `chain_rules` is a little complicated and I'll explain it later.

Now I need a way to initialise my variables $x$ and $y$. This is done using a constructor.

```julia
mutable struct Tensor
    val::AbstractArray  # Value of the Tensor
    parents::Vector     # Who created this Tensor
    chain_rules::Vector # Chain Rule to get global gradients wrt parents

    # Constructor to initialise the Tensor (for leaf nodes)
    function Tensor(val::Number)
        val = [val][:,:] # Storing Scalar as a 1 X 1 Array
        new(val, [], [])
    end
end
```

I also need a way to create variables from operations. I'll just add another constructor for that.

```julia
mutable struct Tensor
    val::AbstractArray  # Value of the Tensor
    parents::Vector     # Who created this Tensor
    chain_rules::Vector # Chain Rule to get global gradients wrt parents

    # Constructor to initialise the Tensor (for leaf nodes)
    function Tensor(val::Number)
        val = [val][:,:] # Storing Scalar as an Array
        new(val, [], [])
    end

    # Constructor to create Tensor using operations (for intermediate nodes)
    function Tensor(val::Number, parents::Vector{Tensor}, chain_rules::Vector)
        val = [val][:,:] # Storing Scalar as an Array
        new(val, parents, chain_rules)
    end
end
```

> Notice that chain rule is defined by the operations, so let's create the operations needed.

## Arithmetic Operations

### Multiply
```julia
function Base.:*(T1::Tensor, T2::Tensor)
    val = T1.val * T2.val
    parents = [T1, T2]

    # Chain Rule Functions defining global gradients wrt parents
    dT1 = global_grad::Tensor -> global_grad * transpose(T2)
    dT2 = global_grad::Tensor -> transpose(T1) * global_grad 
    chain_rules = [dT1, dT2]

    return Tensor(val, parents, chain_rules)
end
```

The multiply here is actually a matrix-matrix multiplication (which will work on scalars too as they're $1 \times 1$ matrices). That is why you see transpose in the chain rule. 

As side note, in vector calculus when two matrices are multiplied 

$$A = B \cdot C$$ 

the derivatives are

$$\frac{\partial A}{\partial B} = C^T, \ \frac{\partial A}{\partial C} = B^T$$

and the order of multiplication in the chain rule is also maintained, i.e. $C^T$ will be post multiplied (just as it's post multiplied in $B \cdot C$) and $B^T$ will be pre multiplied.

> This also needs defining the transpose operation on the `Tensor` which you can see [here](https://github.com/tgautam03/jac.jl/blob/master/src/ops/transpose.jl).

### Power
Similarly, I also defined the power operation the `Tensor`.

```julia
function Base.:^(T::Tensor, n::Number)
	val = T.val .^ n
	parents = [T]

	# Chain Rule Functions defining global gradients wrt parents
	dT = global_grad::Tensor -> global_grad .* n .* T ^ (n-1)
	chain_rules = [dT]

	return Tensor(val, parents, chain_rules)
end
```

### Addition
And the addition operation as well.

```julia
function Base.:+(T1::Tensor, T2::Tensor)
    val = T1.val + T2.val
    parents = [T1, T2]

    # Chain Rule Functions defining global gradients wrt parents
    dT1 = global_grad::Tensor -> global_grad 
    dT2 = global_grad::Tensor -> global_grad
    chain_rules = [dT1, dT2]

    return Tensor(val, parents, chain_rules)
end
```

Now, I can define the forward pass

```julia
x = Tensor(4)
y = Tensor(3)

f1 = x * y
f2 = y^3
f3 = f1 + f2
f = f3^2
```

## Evaluating Gradients
Let's add the backward pass capability

```julia
# Gradient of T wrt all nodes it depends upon
function grad(T::Tensor)
	gradients = Dict() # Dict to hold grads of T wrt all creators

	function compute(node::Tensor, global_grad::Tensor)
		for (parent, chain_rule) in zip(node.parents, node.chain_rules)
			# Chain Rule
			new_global_grad = chain_rule(global_grad)

			# Checking if grad present or not
			if haskey(gradients, parent)
				gradients[parent] += new_global_grad
			else
				gradients[parent] = new_global_grad
			end

			# Recusive call
			compute(parent, new_global_grad)
		end
	end

	# Output Node
	dT_dT = Tensor(ones(size(T.val)))
	compute(T, dT_dT)

	return gradients
end
```

Getting the partial derivatives using AD

```julia
x = Tensor(4)
y = Tensor(3)

# Forward Pass
f1 = x * y
f2 = y^3
f3 = f1 + f2
f = f3^2

# Backward Pass
gradients = grad(f)

# Collect Gradients
∂f_∂x = gradients[x]
∂f_∂y = gradients[y]
```