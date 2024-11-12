module NNLib

# Include all source files
include("activations.jl")
include("NeuralNetwork.jl")
include("optimizer.jl")
include("data_utils.jl")
include("train.jl")

# Use submodules
using .Optimizer
using .Activations
using .NeuralNetwork
using .DataUtils
using .Train

# Export functions and types
export NeuralNetworkModel, 
       initialize_network, 
       forward_pass,
       update!,
       train!, 
       relu, 
       sigmoid, 
       softmax, 
       SGD, 
       Adam, 
       normalize, 
       batch_generator, 
       dsigmoid,
       backward_pass

end  # module MySimpleNN
