# run_network.jl
include("../NNLib/src/NNLib.jl")

using Random
using Statistics
using Plots

# Generate random X values with 20 features for each sample (100 samples in this case)
X = rand(100, 1)

# Calculate Y using an unknown function (adjusted to match the input data)
y = 3 * sum(X.^2, dims=2) .+ 2 * sum(X, dims=2) .+ 5 .+ 0.1 * randn(100, 1)  # Add noise to the output

X = (X .- mean(X)) ./ std(X)
y = (y .- mean(y)) ./ std(y)

# Load the model for future use (for testing or predictions)
loaded_model = NNLib.LoadModel.load_model("my_trained_model.jld2")

# Test the model on new data
X_test = rand(4, 1) * 10  # Generate new x values for the test set
X_test = (X_test .- mean(X)) ./ std(X)  # Scale them similarly to the training data

# Get model predictions for all test data (X_test)
predictions = NNLib.NeuralNetwork.forward_pass(loaded_model, X_test)

# Calculate true y values for test data (based on the known function)
y_test = 3 * X_test.^2 .+ 2 * X_test .+ 5  # True y values for the test data (using the known function)
y_test = (y_test .- mean(y)) ./ std(y)  # Normalize test y values similarly to training data

# Flatten y_test to a 1D vector with 4 elements
y_test = vec(y_test)

# Now perform the subtraction
errors = abs.(predictions[end] .- y_test)

# Flatten the errors array into a 1D array
errors_flat = vec(errors)  # Convert matrix to vector
sorted_indices = sort(sortperm(errors_flat))  # Get sorted indices based on the errors (ascending)

# Select the indices of the top predictions (but not more than available)
top_indices = sorted_indices[1]  # Dynamically selects the top predictions based on available elements
successful_predictions = predictions[top_indices]

# Plot the data points
scatter(X, y, label="Data Points", legend=:topleft)
plot!(X_test, successful_predictions, label="Successful Prediction 1", marker=:circle, color=:green)

# Save the plot as a PNG
png_path = "successful_predictions_plot.png"
savefig(png_path)

println("Plot saved as $png_path")
