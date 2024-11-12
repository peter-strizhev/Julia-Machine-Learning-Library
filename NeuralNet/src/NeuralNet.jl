# run_network.jl
include("../NNLib/src/NNLib.jl")

using Random
using Statistics
using Plots

# Load the model for future use (for testing or predictions)
loaded_model = NNLib.LoadModel.load_model("my_trained_model.jld2")

# Calculate true X values for test data
X_test = rand(100, 1) * 10  # Generate new x values for the test set
X_test = (X_test .- mean(X_test)) ./ std(X_test)  # Scale them similarly to the training data

# Calculate true y values for test data
sign = rand([-1, 1])
y_test = (sign .* X_test.^2) .+ 0.1 * randn(100, 1)
y_test = (y_test .- mean(y_test)) ./ std(y_test)

# Get model predictions for all test data (X_test)
predictions = NNLib.NeuralNetwork.forward_pass(loaded_model, X_test)

# Select the corresponding successful predictions and their X values
successful_predictions = predictions[end]

# Plot the data points
scatter(X_test, y_test, label="Data Points", legend=:topleft)
plot!(X_test, successful_predictions, label="Successful Prediction", marker=:circle, color=:green)

# Save the plot as a PNG
png_path = "successful_predictions_plot.png"
savefig(png_path)

println("Plot saved as $png_path")
