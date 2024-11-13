module LoadModel

using JLD2
using ..NeuralNetwork

# Load model function
function load_model(filename::String)
    @info "Loading model from $filename"
    model = nothing
    try
        jldopen(filename, "r") do file
            layers = read(file, "layers")
            biases = read(file, "biases")
            activations = read(file, "activations")
            model = NeuralNetwork.NeuralNetworkModel(layers, biases, activations)
        end
    catch e
        @error "Failed to load model: $e"
    end
    return model
end

end  # End of LoadModel module
