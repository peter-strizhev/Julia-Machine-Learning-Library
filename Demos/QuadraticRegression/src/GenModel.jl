include("../NNLib/src/NNLib.jl")

using Random
using Statistics
using Plots

point_count = 250

# Dataset Prep

# Generate random X values with 20 features for each sample (100 samples in this case)
X = rand(point_count, 1)
X = (X .- mean(X)) ./ std(X)

# Calculate Y using an unknown function (adjusted to match the input data)
y_pos = X.^2 .+ 0.1 * randn(point_count, 1)
y_pos = (y_pos .- mean(y_pos)) ./ std(y_pos)

y_neg = -X.^2 .+ 0.1 * randn(point_count, 1)
y_neg = (y_neg .- mean(y_neg)) ./ std(y_neg)

# Plotting the results
scatter(X, y_neg, label="Training Data", marker=:circle, color=:blue)
xlabel!("X values")
ylabel!("Y values")
title!("Training Data")

# Save the plot as a PNG
png_path = "training_data_neg.png"
savefig(png_path)

# Plotting the results
scatter(X, y_pos, label="Training Data", marker=:circle, color=:blue)
xlabel!("X values")
ylabel!("Y values")
title!("Training Data")

# Save the plot as a PNG
png_path = "training_data_pos.png"
savefig(png_path)

layer_sizes = [1, 4096, 2048, 1024, 1]
activations = [NNLib.Activations.relu, NNLib.Activations.relu, NNLib.Activations.relu, identity]

# Initialize the network
model = NNLib.NeuralNetwork.initialize_network(layer_sizes, activations, 0.01)
optimizer = NNLib.Optimizer.SGD(0.001)

# Set the training parameters
epochs = Inf
batch_size = 2000
target_loss = 0.023

# Additional parameters for dynamic learning rate decay
min_lr = 1e-6
decay_factor = 0.99
patience = 100

# Train the model with the updated train! function
NNLib.Train.train!(model, X, y_pos, optimizer, epochs, batch_size, target_loss, min_lr, decay_factor, patience)
# NNLib.Train.train!(model, X, y_neg, optimizer, epochs, batch_size, target_loss, min_lr, decay_factor, patience)

# Save the trained model
NNLib.SaveModel.save_model(model, "trained_model.jld2")