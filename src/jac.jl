module jac
using Base

include("Tensor.jl")
export Tensor

include("autograd/grad.jl")
export grad

include("ops/utils.jl")
export length 

include("ops/multiply.jl")
export *

include("ops/divide.jl")
export /

include("ops/transpose.jl")
export transpose

include("ops/sum.jl")
export sum

include("ops/add.jl")
export +

include("ops/subtract.jl")
export -

include("ops/power.jl")
export ^


end # module jac
