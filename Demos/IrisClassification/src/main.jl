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

# Step 1: Load the Iris dataset
iris_data = MLDatasets.Iris()

# Extract features and targets
X = iris_data.features  # Features (sepal length, sepal width, petal length, petal width)
y = iris_data.targets   # Targets (species names)

# Convert features (X) into a matrix and normalize (standardization)
X = Matrix(X)  # Convert to matrix format
X = (X .- mean(X, dims=1)) ./ std(X, dims=1)  # Z-score normalization

# Step 2: Convert labels (y) into numeric values (integer encoding)
label_map = Dict("Iris-setosa" => 0, "Iris-versicolor" => 1, "Iris-virginica" => 2)

y_int = [label_map[label] for label in y.class]  # Convert species names to integers

# Step 3: One-hot encode the labels (y) for neural network output
function one_hot_encode(labels, num_classes)
    return hcat([Float64.(labels .== i) for i in 0:num_classes-1]...)  # One-hot encoding
end

y_one_hot = one_hot_encode(y_int, 3)  # 3 classes in total

# Step 4: Shuffle the dataset (to ensure random splits)
shuffled_indices = shuffle(1:size(X, 1))
X_shuffled = X[shuffled_indices, :]
y_shuffled = y_one_hot[shuffled_indices, :]

# Step 5: Split the dataset into training and testing sets (80% training, 20% testing)
train_ratio = 0.8
split_idx = Int(round(train_ratio * size(X, 1)))

X_train, y_train = X_shuffled[1:split_idx, :], y_shuffled[1:split_idx, :]
X_test, y_test = X_shuffled[split_idx+1:end, :], y_shuffled[split_idx+1:end, :]

# Step 6: Check the shapes of the resulting datasets
println("X_train size: ", size(X_train))
println("y_train size: ", size(y_train))
println("X_test size: ", size(X_test))
println("y_test size: ", size(y_test))

# Define the neural network architecture
input_size = 4
hidden_size = 4096
output_size = 3

layer_sizes = [input_size, hidden_size, output_size]
activations = [NNLib.Activations.relu, identity]  # Make sure there is one less activation than layers

# Initialize the model
model = NNLib.NeuralNetwork.initialize_network(layer_sizes, activations, 0.001)

# Set up the optimizer and training parameters
optimizer = NNLib.Optimizer.SGD(0.001)
epochs = Inf
batch_size = 1000
target_loss = 0.08
min_lr = 1.0e-6
decay_factor = 0.99
patience = Inf

# Train the model on the Iris dataset
NNLib.Train.train!(model, X_train, y_train, optimizer, epochs, batch_size, target_loss, min_lr, decay_factor, patience)

# Generate predictions
predictions = NNLib.NeuralNetwork.forward_pass(model, X_test)
predicted_classes = [index[1] for index in argmax(predictions, dims=2)] .- 1

true_classes = [index[1] for index in argmax(y_test, dims=2)] .- 1


# Plot the true vs. predicted values
# scatter(1:length(true_classes), true_classes, label="True Labels", color=:blue, markersize=4)
scatter(1:length(predicted_classes), predicted_classes, label="Predicted Labels", color=:yellow, markersize=4)

scatter!(X[:, 1], X[:, 2], 
        color = y_int, 
        xlabel = "Sepal Length", ylabel = "Sepal Width", 
        title = "Iris Dataset - Sepal Length vs Sepal Width", 
        label = ["Iris-setosa" "Iris-versicolor" "Iris-virginica"], 
        legend = :topright)


# Add labels and title
xlabel!("Sample Index")
ylabel!("Class Label")
title!("True vs. Predicted Labels for Iris Dataset")

# Show the plot
display(plot)

# Create a confusion matrix
# cm = ConfusionMatrix(predicted_classes, true_classes)

# Plot the confusion matrix
# heatmap(cm, title="Confusion Matrix", xlabel="Predicted", ylabel="True", color=:Blues)
savefig("iris_predictions_plot.png")