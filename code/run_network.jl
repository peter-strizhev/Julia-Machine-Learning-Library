# run_network.jl
include("MySimpleNN.jl")

using .MySimpleNN

# Initialize a neural network
layer_sizes = [10, 5, 2]
activations = [Activations.relu, Activations.sigmoid]
model = NeuralNetwork.initialize_network(layer_sizes, activations)

# Generate dummy data
X = rand(100, 10)
y = rand(100, 2)

# Set optimizer and train
optimizer = Optimizer.SGD(lr=0.01)
Train.train!(model, X, y, optimizer, epochs=10, batch_size=10)
