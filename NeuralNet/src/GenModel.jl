include("../NNLib/src/NNLib.jl")

using Random
using Statistics

point_count = 500

# Generate random X values with 20 features for each sample (100 samples in this case)
X = rand(point_count, 1)
X = (X .- mean(X)) ./ std(X)

# Calculate Y using an unknown function (adjusted to match the input data)
y = X.^2 .+ 0.1 * randn(point_count, 1)  # Adding noise to exponential data
y = (y .- mean(y)) ./ std(y)

# layer_sizes = [1, 4096, 1024, 512, 1]
# activations = [NNLib.Activations.leaky_relu, NNLib.Activations.leaky_relu, NNLib.Activations.relu, identity]

layer_sizes = [1, 2048, 512, 1]
activations = [NNLib.Activations.relu, NNLib.Activations.relu, identity]

# Initialize the network
model = NNLib.NeuralNetwork.initialize_network(layer_sizes, activations)
optimizer = NNLib.Optimizer.SGD(0.001)

epochs = Inf
batch_size = 3000
target_loss = 0.01

NNLib.Train.train!(model, X, y, optimizer, epochs, batch_size, target_loss)

# Save the trained model
NNLib.SaveModel.save_model(model, "my_trained_model.jld2")