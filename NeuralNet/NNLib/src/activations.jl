module Activations

function relu(x)
    return max.(0, x)
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

end  # module Activations
