# run_network.jl
include("../NNLib/src/NNLib.jl")

# Load the Iris dataset
using MLDatasets
using Random
using Statistics
using LinearAlgebra
using Plots
using DataFrames
using EvalMetrics

# Load the Iris dataset
iris_data = MLDatasets.Iris()

# Extract features and targets
X = iris_data.features  # Features (sepal length, sepal width, petal length, petal width)
y = iris_data.targets   # Targets (species names)

# Convert features (X) into a matrix and normalize (standardization)
X = Matrix(X)
X = (X .- mean(X, dims=1)) ./ std(X, dims=1)

# Convert labels (y) into numeric values (integer encoding)
label_map = Dict("Iris-setosa" => 0, "Iris-versicolor" => 1, "Iris-virginica" => 2)

y_int = [label_map[label] for label in y.class]  # Convert species names to integers

# Scatter actual data
scatter(X[:, 1], X[:, 2], X[:, 3], c=y_int)

# Add labels and title
xlabel!("Sepal Length")
ylabel!("Sepal Width")
zlabel!("Petal Length")
title!("True Labels for Iris Dataset")

# Show the plot
savefig("iris_real_data_plot.png")

model = NNLib.LoadModel.load_model("Iris_Model_063.jld2")

# Generate predictions
predictions = NNLib.NeuralNetwork.forward_pass(model, X)
predicted_classes = predictions[end]

# Find the predicted class for each test sample by taking the argmax
model_y_output = [argmax(predicted_classes[i, :]) for i in 1:size(predicted_classes, 1)]

# Scatter model output on 3d graph
scatter(X[:, 1], X[:, 2], X[:, 3], c=model_y_output)

# Add labels and title
xlabel!("Sepal Length")
ylabel!("Sepal Width")
zlabel!("Petal Length")
title!("Iris Dataset - Predicted Classes")

# Show the plot
display(plot)
savefig("iris_predictions_plot.png")