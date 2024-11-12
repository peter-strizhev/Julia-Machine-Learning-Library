include("../NNLib/src/NNLib.jl")

using Random
using Statistics

point_count = 100

# Generate random X values with 20 features for each sample (100 samples in this case)
X = rand(point_count, 1)
X = (X .- mean(X)) ./ std(X)

layer_sizes = [1, 128, 64, 32, 16, 1]
activations = [NNLib.Activations.relu, NNLib.Activations.leaky_relu, NNLib.Activations.relu, NNLib.Activations.relu, identity]

# Initialize the network
model = NNLib.NeuralNetwork.initialize_network(layer_sizes, activations)
optimizer = NNLib.Optimizer.SGD(0.001)

epochs = Inf
batch_size = 100

exponents = [2]
for exponent in exponents
    # Calculate Y using an unknown function (adjusted to match the input data)
    y = X.^exponent .+ 0.1 * randn(point_count, 1)  # Adding noise to exponential data
    y = (y .- mean(y)) ./ std(y)
    NNLib.Train.train!(model, X, y, optimizer, epochs, batch_size)
end

# Save the trained model
NNLib.SaveModel.save_model(model, "my_trained_model.jld2")