module SaveModel

using JLD2
using ..NeuralNetwork
using ..Transformer

# Save model function
function save_model(model, filename::String)
    @info "Saving model to $filename"
    jldopen(filename, "w") do file
        write(file, "layers", model.layers)
        write(file, "biases", model.biases)
        write(file, "activations", model.activations)
    end
end

function save_transformer(model::Transformer.TransformerModel, path::String)
    open(path, "w") do io
        serialize(io, model)
    end
end

end  # End of SaveModel module
