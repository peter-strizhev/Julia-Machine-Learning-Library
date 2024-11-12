# run_network.jl
include("../NNLib/src/NNLib.jl")
# using NNLib

# Initialize a neural network
layer_sizes = [10, 5, 2]
activations = [NNLib.Activations.relu, NNLib.Activations.sigmoid]
model = NNLib.NeuralNetwork.initialize_network(layer_sizes, activations)  # Calls your `initialize_network` function

# Generate dummy data
X = rand(100, 10)
y = rand(100, 2)

# Set optimizer and train
optimizer = NNLib.Optimizer.SGD(0.01)
NNLib.Train.train!(model, X, y, optimizer, 10, 10)