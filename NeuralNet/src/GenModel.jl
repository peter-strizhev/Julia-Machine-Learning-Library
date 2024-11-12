include("../NNLib/src/NNLib.jl")

using Random
using Statistics

point_count = 100

# Generate random X values with 20 features for each sample (100 samples in this case)
X = rand(point_count, 1)
X = (X .- mean(X)) ./ std(X)

# Calculate Y using an unknown function (adjusted to match the input data)
y = X.^2 .+ 0.1 * randn(point_count, 1)  # Adding noise to exponential data
y = (y .- mean(y)) ./ std(y)

layer_sizes = [1, 4096, 1024,  1]
activations = [NNLib.Activations.relu, NNLib.Activations.relu, identity]

# Initialize the network
model = NNLib.NeuralNetwork.initialize_network(layer_sizes, activations)
optimizer = NNLib.Optimizer.SGD(0.01)

# Set the training parameters
epochs = Inf
batch_size = 1000
target_loss = 0.015

# Additional parameters for dynamic learning rate decay
min_lr = 1e-6            # Minimum learning rate
decay_factor = 0.001     # Factor by which learning rate is reduced
patience = 2            # Number of epochs to wait without improvement before decaying the learning rate

# Train the model with the updated train! function
NNLib.Train.train!(model, X, y, optimizer, epochs, batch_size, target_loss, min_lr, decay_factor, patience)


# Save the trained model
NNLib.SaveModel.save_model(model, "my_trained_model.jld2")