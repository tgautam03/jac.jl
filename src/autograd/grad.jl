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