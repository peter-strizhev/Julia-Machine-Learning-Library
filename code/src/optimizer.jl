module Optimizer

# Stochastic Gradient Descent (SGD) optimizer
struct SGD
    lr::Float64  # Learning rate
end

# Update rule for SGD
function update!(optimizer::SGD, layer, gradients_w, gradients_b)
    # Update weights and biases for each layer using the SGD rule
    layer.weights .-= optimizer.lr * gradients_w
    layer.bias .-= optimizer.lr * gradients_b
end


end  # module Optimizer
