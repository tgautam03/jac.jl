# Length of the array
function Base.length(T::Tensor)
    return prod(size(T.val))
end