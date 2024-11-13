module Optimizer

# Stochastic Gradient Descent (SGD) optimizer
mutable struct SGD
    lr::Float64  # Learning rate
end

# Update rule for SGD
function update!(optimizer::SGD, weights::Matrix{Float64}, gradients_w::Matrix{Float64}, biases::Vector{Float64}, gradients_b::Vector{Float64})
    weights .-= optimizer.lr * gradients_w
    biases .-= optimizer.lr * gradients_b
end

end  # module Optimizer