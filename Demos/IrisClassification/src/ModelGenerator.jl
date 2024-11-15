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

# One-hot encode the labels (y) for neural network output
function one_hot_encode(labels, num_classes)
    return hcat([Float64.(labels .== i) for i in 0:num_classes-1]...)  # One-hot encoding
end

y_one_hot = one_hot_encode(y_int, 3)  # 3 classes in total

# Shuffle the dataset (to ensure random splits)
shuffled_indices = shuffle(1:size(X, 1))
X_shuffled = X[shuffled_indices, :]
y_shuffled = y_one_hot[shuffled_indices, :]

# Split the dataset into training and testing sets (90% training, 10% testing)
train_ratio = 0.9
split_idx = Int(round(train_ratio * size(X, 1)))

X_train, y_train = X_shuffled[1:split_idx, :], y_shuffled[1:split_idx, :]
X_test, y_test = X_shuffled[split_idx+1:end, :], y_shuffled[split_idx+1:end, :]

# Define the neural network architecture
input_size = 4
output_size = 3

layer_sizes = [input_size, 256, 128, output_size]
activations = [NNLib.Activations.relu, NNLib.Activations.relu, identity]  # Make sure there is one less activation than layers

# Initialize the model
model = NNLib.NeuralNetwork.initialize_network(layer_sizes, activations, 0.01)

# Set up the optimizer and training parameters
optimizer = NNLib.Optimizer.SGD(0.001)
epochs = Inf
batch_size = 135
target_loss = 0.074
min_lr = 1.0e-6
decay_factor = 0.99
patience = Inf

# Train the model on the Iris dataset
# Conduct 3 runs of the training, each with a more precise optimizer
NNLib.Train.train!(model, X_train, y_train, optimizer, epochs, batch_size, target_loss, min_lr, decay_factor, patience)
NNLib.SaveModel.save_model(model, "Iris_Model_0074.jld2")

optimizer = NNLib.Optimizer.SGD(0.00025)
epochs = Inf
batch_size = 135
target_loss = 0.065
min_lr = 1.0e-6
decay_factor = 0.99
patience = Inf

NNLib.Train.train!(model, X_train, y_train, optimizer, epochs, batch_size, target_loss, min_lr, decay_factor, patience)
NNLib.SaveModel.save_model(model, "Iris_Model_0065.jld2")

optimizer = NNLib.Optimizer.SGD(0.0001)
epochs = Inf
batch_size = 135
target_loss = 0.063
min_lr = 1.0e-6
decay_factor = 0.99
patience = Inf

NNLib.Train.train!(model, X_train, y_train, optimizer, epochs, batch_size, target_loss, min_lr, decay_factor, patience)

# Save the model
NNLib.SaveModel.save_model(model, "Iris_Model_063.jld2")