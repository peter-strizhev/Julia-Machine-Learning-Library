include("../NNLib/src/NNLib.jl")

using Random
using Statistics

point_count = 500

# Generate random X values with 20 features for each sample (100 samples in this case)
X = rand(point_count, 1)


X = (X .- mean(X)) ./ std(X)

layer_sizes = [1, 128, 256, 512, 128, 1]
activations = [NNLib.Activations.relu, NNLib.Activations.relu, NNLib.Activations.relu, NNLib.Activations.relu, identity]

# Initialize the network
model = NNLib.NeuralNetwork.initialize_network(layer_sizes, activations)

optimizer = NNLib.Optimizer.SGD(0.01)

epochs = 23000
batch_size = 1000

exponents = [2]

for exponent in exponents
    # Calculate Y using an unknown function (adjusted to match the input data)
    y = X.^exponent .+ 0.1 * randn(point_count, 1)  # Adding noise to exponential data

    y = (y .- mean(y)) ./ std(y)
    NNLib.Train.train!(model, X, y, optimizer, epochs, batch_size)
end

# Save the trained model
NNLib.SaveModel.save_model(model, "my_trained_model.jld2")