module NeuralNetwork

using ..Activations

# Define the neural network model
struct NeuralNetworkModel
    layers::Vector{Matrix{Float64}}    # List of weight matrices for each layer
    biases::Vector{Vector{Float64}}    # List of bias vectors for each layer
    activations::Vector{Function}      # List of activation functions for each layer
end

# Initialize the neural network with random weights and biases
function initialize_network(layer_sizes::Vector{Int}, activations::Vector{Function}, precision)
    layers = [randn(layer_sizes[i+1], layer_sizes[i]) * precision for i in 1:length(layer_sizes)-1]
    biases = [randn(layer_sizes[i+1]) * precision for i in 1:length(layer_sizes)-1]
    NeuralNetworkModel(layers, biases, activations)
end

# Perform a forward pass through the network
function forward_pass(nn::NeuralNetworkModel, X::Matrix{Float64})
    activations = X
    activations_list = [X]  # Save activations for later use in backpropagation
    
    for (W, b, activation) in zip(nn.layers, nn.biases, nn.activations)
        # Ensure weight matrix W is transposed for correct matrix multiplication
        z = activations * W' .+ b'  # W' ensures correct dimensions for multiplication
        
        # Apply activation function
        activations = activation.(z)
        
        # Save activations for each layer
        push!(activations_list, activations)
    end
    
    # Return final activation (output of the network)
    return activations_list
end

# Backpropagation: compute gradients for weights and biases
function backward_pass(nn::NeuralNetworkModel, X::Matrix{Float64}, y::Matrix{Float64}, activations_list::Vector{Matrix{Float64}})
    # Initialize gradients
    gradients_w = []
    gradients_b = []

    # Compute the error at the output layer
    error = activations_list[end] .- y  # Error at the output layer
    delta = error .* Activations.dsigmoid(activations_list[end])  # Using sigmoid derivative for the output layer

    # Backpropagate the error to previous layers
    for i in length(nn.layers):-1:1
        # Compute gradients for weights and biases
        gradient_w = delta' * activations_list[i]  # Gradient for weights
        gradient_b = sum(delta, dims=1)            # Gradient for biases

        # Store gradients
        push!(gradients_w, gradient_w)
        push!(gradients_b, gradient_b)

        # Backpropagate to the previous layer (if not the first layer)
        if i > 1
            delta = (delta * nn.layers[i]) .* Activations.dsigmoid(activations_list[i])
        end
    end

    # Reverse the gradients list because we need to apply updates in the correct order
    return reverse(gradients_w), reverse(gradients_b)
end

end  # module NeuralNetwork
