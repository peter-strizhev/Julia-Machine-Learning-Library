# run_network.jl
include("../NNLib/src/NNLib.jl")

using MLDatasets
using DataFrames
using Random
using Statistics
using LinearAlgebra
using Plots

# Load the Iris dataset
iris_data = MLDatasets.Iris()

X = iris_data.features  # Extract features
y = iris_data.targets   # Extract targets

X = Matrix(X)
X = (X .- mean(X, dims=1)) ./ std(X, dims=1)

label_map = Dict("Iris-setosa" => 0, "Iris-versicolor" => 1, "Iris-virginica" => 2)

y_int = [label_map[row[:class]] for row in eachrow(y)]

function one_hot_encode(labels, num_classes)
    return hcat([Float64.(I(num_classes)[label + 1, :]) for label in labels]...)
end

y_one_hot = one_hot_encode(y_int, 3)
y_one_hot = hcat(y_one_hot...)'

# Shuffle X and y together
shuffled_indices = shuffle(1:size(X, 1))
X_shuffled = X[shuffled_indices, :]
y_shuffled = y_one_hot[shuffled_indices, :]

# Split into training and testing sets
train_ratio = 0.8
split_idx = Int(round(train_ratio * size(X, 1)))

X_train, y_train = X_shuffled[1:split_idx, :], y_shuffled[1:split_idx, :]
X_test, y_test = X_shuffled[split_idx+1:end, :], y_shuffled[split_idx+1:end, :]

# Define the neural network architecture
input_size = 4
hidden_size = 4096
output_size = 3

layer_sizes = [input_size, hidden_size, output_size]
activations = [NNLib.Activations.relu, identity]  # Make sure there is one less activation than layers

# Initialize the model
model = NNLib.NeuralNetwork.initialize_network(layer_sizes, activations, 0.001)

# Set up the optimizer and training parameters
optimizer = NNLib.Optimizer.SGD(0.0001)
epochs = Inf
batch_size = 1000
target_loss = 0.05
min_lr = 1.0e-6
decay_factor = 0.99
patience = 1000

# Train the model on the Iris dataset
NNLib.Train.train!(model, X_train, y_train, optimizer, epochs, batch_size, target_loss, min_lr, decay_factor, patience)

# Generate predictions
predictions = NNLib.NeuralNetwork.forward_pass(model, X_test)

# Flatten y_test to a 1D vector with 4 elements
y_test = vec(y_test)

# Select the corresponding successful predictions and their X values
successful_predictions = predictions[end]

# Sort the X_test values in ascending order and rearrange corresponding y_test and predictions
sorted_indices = sortperm(X_test[:, 1])  # Sort based on the first column of X_test
X_test_sorted = X_test[sorted_indices, :]  # Reorder X_test
y_test_sorted = y_test[sorted_indices]  # Reorder y_test accordingly
successful_predictions_sorted = successful_predictions[sorted_indices]  # Reorder predictions accordingly

# Plot the true data points (y_test) and model predictions as continuous lines
scatter(X_test, y_test, label="Data Points", legend=:topleft)
plot!(X_test_sorted[:, 1], successful_predictions_sorted, label="Predictions", color=:red, linewidth=2)

# Save the plot as a PNG
png_path = "successful_predictions_plot.png"
savefig(png_path)
println("Plot saved as $png_path")
