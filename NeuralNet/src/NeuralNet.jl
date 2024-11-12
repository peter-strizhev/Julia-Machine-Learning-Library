# run_network.jl
include("../NNLib/src/NNLib.jl")

using Random
using Statistics
using Plots

# Load the model for future use (for testing or predictions)
loaded_model = NNLib.LoadModel.load_model("my_trained_model.jld2")

# Test the model on new data
X_test = rand(100, 1) * 10  # Generate new x values for the test set
X_test = (X_test .- mean(X_test)) ./ std(X_test)  # Scale them similarly to the training data

# Calculate true y values for test data
# sign = rand([-1, 1])
sign = 1
y_test = (sign .* X_test.^2) .+ 0.1 * randn(100, 1)
y_test = (y_test .- mean(y_test)) ./ std(y_test)

# Get model predictions for all test data (X_test)
predictions = NNLib.NeuralNetwork.forward_pass(loaded_model, X_test)

# Debug: Print predictions and shapes
println("Shape of final layer: ", size(predictions[end]))
println("Shape of y_test: ", size(y_test))

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
