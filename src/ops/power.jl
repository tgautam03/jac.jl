# Power Operation
function Base.:^(T::Tensor, n::Number)
	val = T.val .^ n
	parents = [T]

	# Chain Rule Functions defining global gradients wrt parents
	dT = global_grad::Tensor -> global_grad .* n .* T ^ (n-1)
	chain_rules = [dT]

	return Tensor(val, parents, chain_rules)
end