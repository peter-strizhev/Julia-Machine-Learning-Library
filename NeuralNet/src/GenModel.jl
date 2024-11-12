include("../NNLib/src/NNLib.jl")

using Random
using Statistics

# Generate random X values with 20 features for each sample (100 samples in this case)
X = rand(100, 1)

# Calculate Y using an unknown function (adjusted to match the input data)
y = 3 * sum(X.^2, dims=2) .+ 2 * sum(X, dims=2) .+ 5 .+ 0.1 * randn(100, 1)  # Add noise to the output

X = (X .- mean(X)) ./ std(X)
y = (y .- mean(y)) ./ std(y)

layer_sizes = [1, 4096, 1024, 1]

# Define the activation functions for the hidden layers (ReLU for both)
activations = [NNLib.Activations.relu, NNLib.Activations.relu, identity]  # Using identity for the output layer

# Initialize the network
model = NNLib.NeuralNetwork.initialize_network(layer_sizes, activations)

function mse_loss(predicted, target)
    return sum((predicted .- target).^2) / length(target)
end

optimizer = NNLib.Optimizer.SGD(0.01)

epochs = 3000
batch_size = 100

NNLib.Train.train!(model, X, y, optimizer, epochs, batch_size)

# Save the trained model
NNLib.SaveModel.save_model(model, "my_trained_model.jld2")