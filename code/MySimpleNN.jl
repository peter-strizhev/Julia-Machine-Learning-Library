module MySimpleNN

# Include all source files
include("src/activations.jl")
include("src/NeuralNetwork.jl")
include("src/optimizer.jl")
include("src/data_utils.jl")
include("src/train.jl")

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
