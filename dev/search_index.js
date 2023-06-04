var documenterSearchIndex = {"docs":
[{"location":"dev_blog/basic_scalar_example/#An-example-involving-Scalars","page":"An example involving Scalars","title":"An example involving Scalars","text":"","category":"section"},{"location":"dev_blog/basic_scalar_example/#Introduction","page":"An example involving Scalars","title":"Introduction","text":"","category":"section"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"In this post, I'll be coding up the example discussed here. Just as a recap, we have a function f(xy)=(xy+y^3)^2 and task is to get the partial derivatives with respect to x and y. This same function can also be written as a series of simpler operations ","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"f_1 = x cdot y tag1","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"f_2 = y^3 tag2","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"f_3 = f_1 + f_2 tag3","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"f(xy) = f_3^2 tag4","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"Looking at these equations, we need a way to represent the value and derivative of each node (i.e. x, y, f_1, cdots, f(xy)). To facilitate this, I have to create a custom variable(ish) that has multiple attributes. An obvious choice here is to use a mutable struct (we need the ability to modify the attributes of this struct) and I'll name that Tensor. ","category":"page"},{"location":"dev_blog/basic_scalar_example/#Tensor.jl","page":"An example involving Scalars","title":"Tensor.jl","text":"","category":"section"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"mutable struct Tensor\n    val::AbstractArray  # Value of the Tensor\n    parents::Vector     # Who created this Tensor\n    chain_rules::Vector # Chain Rule to get global gradients wrt parents\nend","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"val stores the value of the node or variable.\nparents is a list that will contain the other nodes that creates this node. It'll be empty for leaf nodes x and y.\nchain_rules is a little complicated and I'll explain it later.","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"Now I need a way to initialise my variables x and y. This is done using a constructor.","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"mutable struct Tensor\n    val::AbstractArray  # Value of the Tensor\n    parents::Vector     # Who created this Tensor\n    chain_rules::Vector # Chain Rule to get global gradients wrt parents\n\n    # Constructor to initialise the Tensor (for leaf nodes)\n    function Tensor(val::Number)\n        val = [val][:,:] # Storing Scalar as a 1 X 1 Array\n        new(val, [], [])\n    end\nend","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"I also need a way to create variables from operations. I'll just add another constructor for that.","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"mutable struct Tensor\n    val::AbstractArray  # Value of the Tensor\n    parents::Vector     # Who created this Tensor\n    chain_rules::Vector # Chain Rule to get global gradients wrt parents\n\n    # Constructor to initialise the Tensor (for leaf nodes)\n    function Tensor(val::Number)\n        val = [val][:,:] # Storing Scalar as an Array\n        new(val, [], [])\n    end\n\n    # Constructor to create Tensor using operations (for intermediate nodes)\n    function Tensor(val::Number, parents::Vector{Tensor}, chain_rules::Vector)\n        val = [val][:,:] # Storing Scalar as an Array\n        new(val, parents, chain_rules)\n    end\nend","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"Notice that chain rule is defined by the operations, so let's create the operations needed.","category":"page"},{"location":"dev_blog/basic_scalar_example/#Arithmetic-Operations","page":"An example involving Scalars","title":"Arithmetic Operations","text":"","category":"section"},{"location":"dev_blog/basic_scalar_example/#Multiply","page":"An example involving Scalars","title":"Multiply","text":"","category":"section"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"function Base.:*(T1::Tensor, T2::Tensor)\n    val = T1.val * T2.val\n    parents = [T1, T2]\n\n    # Chain Rule Functions defining global gradients wrt parents\n    dT1 = global_grad::Tensor -> global_grad * transpose(T2)\n    dT2 = global_grad::Tensor -> transpose(T1) * global_grad \n    chain_rules = [dT1, dT2]\n\n    return Tensor(val, parents, chain_rules)\nend","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"The multiply here is actually a matrix-matrix multiplication (which will work on scalars too as they're 1 times 1 matrices). That is why you see transpose in the chain rule. ","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"As side note, in vector calculus when two matrices are multiplied ","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"A = B cdot C","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"the derivatives are","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"fracpartial Apartial B = C^T  fracpartial Apartial C = B^T","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"and the order of multiplication in the chain rule is also maintained, i.e. C^T will be post multiplied (just as it's post multiplied in B cdot C) and B^T will be pre multiplied.","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"This also needs defining the transpose operation on the Tensor which you can see here.","category":"page"},{"location":"dev_blog/basic_scalar_example/#Power","page":"An example involving Scalars","title":"Power","text":"","category":"section"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"Similarly, I also defined the power operation the Tensor.","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"function Base.:^(T::Tensor, n::Number)\n\tval = T.val .^ n\n\tparents = [T]\n\n\t# Chain Rule Functions defining global gradients wrt parents\n\tdT = global_grad::Tensor -> global_grad .* n .* T ^ (n-1)\n\tchain_rules = [dT]\n\n\treturn Tensor(val, parents, chain_rules)\nend","category":"page"},{"location":"dev_blog/basic_scalar_example/#Addition","page":"An example involving Scalars","title":"Addition","text":"","category":"section"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"And the addition operation as well.","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"function Base.:+(T1::Tensor, T2::Tensor)\n    val = T1.val + T2.val\n    parents = [T1, T2]\n\n    # Chain Rule Functions defining global gradients wrt parents\n    dT1 = global_grad::Tensor -> global_grad \n    dT2 = global_grad::Tensor -> global_grad\n    chain_rules = [dT1, dT2]\n\n    return Tensor(val, parents, chain_rules)\nend","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"Now, I can define the forward pass","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"x = Tensor(4)\ny = Tensor(3)\n\nf1 = x * y\nf2 = y^3\nf3 = f1 + f2\nf = f3^2","category":"page"},{"location":"dev_blog/basic_scalar_example/#Evaluating-Gradients","page":"An example involving Scalars","title":"Evaluating Gradients","text":"","category":"section"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"Let's add the backward pass capability","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"# Gradient of T wrt all nodes it depends upon\nfunction grad(T::Tensor)\n\tgradients = Dict() # Dict to hold grads of T wrt all creators\n\n\tfunction compute(node::Tensor, global_grad::Tensor)\n\t\tfor (parent, chain_rule) in zip(node.parents, node.chain_rules)\n\t\t\t# Chain Rule\n\t\t\tnew_global_grad = chain_rule(global_grad)\n\n\t\t\t# Checking if grad present or not\n\t\t\tif haskey(gradients, parent)\n\t\t\t\tgradients[parent] += new_global_grad\n\t\t\telse\n\t\t\t\tgradients[parent] = new_global_grad\n\t\t\tend\n\n\t\t\t# Recusive call\n\t\t\tcompute(parent, new_global_grad)\n\t\tend\n\tend\n\n\t# Output Node\n\tdT_dT = Tensor(ones(size(T.val)))\n\tcompute(T, dT_dT)\n\n\treturn gradients\nend","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"Getting the partial derivatives using AD","category":"page"},{"location":"dev_blog/basic_scalar_example/","page":"An example involving Scalars","title":"An example involving Scalars","text":"x = Tensor(4)\ny = Tensor(3)\n\n# Forward Pass\nf1 = x * y\nf2 = y^3\nf3 = f1 + f2\nf = f3^2\n\n# Backward Pass\ngradients = grad(f)\n\n# Collect Gradients\n∂f_∂x = gradients[x]\n∂f_∂y = gradients[y]","category":"page"},{"location":"#JAC","page":"Home","title":"JAC","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"I would recommend going through the short theoretical introduction to Automatic Differentiation explained here before moving forward with the implementation details.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The documentation is divided into two broad sections:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Development: Here you'll find a more story like blogs showing the development of JAC over time (one step at a time).\nManual: Description of each struct, function, etc so that you don't have to dig through the code base to find how it's implemented.","category":"page"},{"location":"user_manual/Tensor/#Tensor.jl","page":"Tensor.jl","title":"Tensor.jl","text":"","category":"section"},{"location":"user_manual/Tensor/","page":"Tensor.jl","title":"Tensor.jl","text":"mutable struct Tensor\n    val::AbstractArray  # Value of the Tensor\n    parents::Vector     # Who created this Tensor\n    chain_rules::Vector # Chain Rule to get global gradients wrt parents\n\n    # Constructor to initialise the Tensor (for leaf nodes)\n    function Tensor(val::Number) # If input is a scalar\n        val = [val][:,:] # Storing Scalar as an Array\n        new(val, [], [])\n    end\n\n    function Tensor(val::AbstractArray) # If input is a matrix\n        new(val, [], [])\n    end\n\n\n    # Constructor to create Tensor using operations (for intermediate nodes)\n    function Tensor(val::Number, parents::Vector{Tensor}, chain_rules::Vector) # If input is a number\n        val = [val][:,:] # Storing Scalar as an Array\n        new(val, parents, chain_rules)\n    end\n\n    function Tensor(val::AbstractArray, parents::Vector{Tensor}, chain_rules::Vector) # If input is a matrix\n        new(val, parents, chain_rules)\n    end\nend","category":"page"}]
}
