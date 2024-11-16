module NeuralNetwork

using ..Activations

# Define the neural network model
mutable struct NeuralNetworkModel
    layers::Vector{Matrix{Float64}}    # List of weight matrices for each layer
    biases::Vector{Vector{Float64}}    # List of bias vectors for each layer
    activations::Vector{Function}      # List of activation functions for each layer
end

mutable struct RecurrentNeuralNetworkModel
    Wxh::Matrix{Float64}  # Input-to-hidden weights
    Whh::Matrix{Float64}  # Hidden-to-hidden weights
    bh::Vector{Float64}   # Hidden bias
    Why::Matrix{Float64}  # Hidden-to-output weights
    by::Vector{Float64}   # Output bias
    hidden_size::Int      # Number of hidden units
    input_size::Int       # Input feature size
    output_size::Int      # Output feature size
end

# Initialize the neural network with random weights and biases
function initialize_network(layer_sizes::Vector{Int}, activations::Vector{Function}, precision)
    layers = [randn(layer_sizes[i+1], layer_sizes[i]) * precision for i in 1:length(layer_sizes)-1]
    biases = [randn(layer_sizes[i+1]) * precision for i in 1:length(layer_sizes)-1]
    NeuralNetworkModel(layers, biases, activations)
end

function initialize_rnn(hidden_size::Int64, input_size::Int64, output_size::Int64, precision::Float64)
    # Initialize weights and biases with small random values
    Wxh = randn(hidden_size, input_size) * precision
    Whh = randn(hidden_size, hidden_size) * precision
    bh = zeros(hidden_size)

    Why = randn(output_size, hidden_size) * precision
    by = zeros(output_size)

    return RecurrentNeuralNetworkModel(Wxh, Whh, bh, Why, by, hidden_size, input_size, output_size)
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

function rnn_forward(model::RecurrentNeuralNetworkModel, inputs::Vector{Vector{Float64}}, h_prev::Vector{Float64})
    Wxh, Whh, bh, Why, by = model.Wxh, model.Whh, model.bh, model.Why, model.by
    hidden_size = model.hidden_size

    h_states = []  # Store hidden states
    outputs = []   # Store outputs

    h = h_prev
    for x in inputs
        h = tanh(Wxh * x + Whh * h + bh)
        push!(h_states, h)

        y = Why * h + by
        push!(outputs, y)
    end

    return outputs, h_states, h
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

function rnn_backward!(model::RecurrentNeuralNetworkModel, inputs::Vector{Vector{Float64}}, h_prev::Vector{Float64}, learning_rate::Float64)
    Wxh, Whh, bh, Why, by = model.Wxh, model.Whh, model.bh, model.Why, model.by

    # Initialize gradients
    dWxh = zeros(size(Wxh))
    dWhh = zeros(size(Whh))
    dbh = zeros(size(bh))
    dWhy = zeros(size(Why))
    dby = zeros(size(by))

    dh_next = zeros(size(h_prev))

    # Backpropagation through time
    for t in reverse(1:length(inputs))
        # Compute gradients (not detailed for brevity)
        # Update dWxh, dWhh, dbh, dWhy, dby, and dh_next
    end

    # Update parameters using gradient descent
    model.Wxh -= learning_rate * dWxh
    model.Whh -= learning_rate * dWhh
    model.bh -= learning_rate * dbh
    model.Why -= learning_rate * dWhy
    model.by -= learning_rate * dby
end

end  # module NeuralNetwork
