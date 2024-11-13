module Activations

# Relu activation function
function relu(x)
    return max.(0, x)
end

# Derivative of ReLU activation function
function drelu(x)
    return x .> 0
end

# Sigmoid activation function
function sigmoid(x)
    return 1.0 ./ (1.0 .+ exp.(-x))
end

# Derivative of the sigmoid activation function
function dsigmoid(x)
    s = sigmoid(x)
    return s .* (1 .- s)
end

# Tanh activation function
function tanh(x)
    return tanh.(x)
end

# Derivative of Tanh activation function
function dtanh(x)
    return 1 .- tanh.(x).^2
end

# Leaky ReLU activation function
function leaky_relu(x, alpha=0.001)
    return max.(alpha .* x, x)
end

# Derivative of Leaky ReLU activation function
function dleaky_relu(x, alpha=0.01)
    return (x .> 0) .+ (x .<= 0) .* alpha
end

# Swish activation function (Self-Gated)
function swish(x)
    return x .* sigmoid(x)
end

# Derivative of Swish activation function
function dswish(x)
    s = sigmoid(x)
    return s .+ x .* s .* (1 .- s)
end

# Mish activation function
function mish(x)
    return x .* tanh.(log1p.(exp.(x)))
end

# Derivative of Mish activation function
function dmish(x)
    return mish(x) .+ x .* (1 .- tanh.(log1p.(exp.(x))).^2) .* exp.(x) ./ (1 .+ exp.(x))
end

end  # module Activations
