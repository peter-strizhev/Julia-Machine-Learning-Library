include("../NNLib/src/NNLib.jl")

using MLDatasets
using IterTools
using Random

# Load the UD English dataset
ud_data = MLDatasets.UD_English()

# Dereference the memory reference
dereferenced_data = collect(ud_data.features.ref)

function extract_tokens_and_labels(sentence)
    tokens = [word[2] for word in sentence]  # Extract the "Form"
    tags = [word[4] for word in sentence]    # Extract the "UPOS" tag
    return tokens, tags
end

function build_vocabulary(sentences)
    token_vocab = Set()
    tag_vocab = Set()

    for sentence in sentences
        tokens, tags = extract_tokens_and_labels(sentence)
        union!(token_vocab, tokens)
        union!(tag_vocab, tags)
    end

    token_to_index = Dict(token => idx for (idx, token) in enumerate(token_vocab))
    tag_to_index = Dict(tag => idx for (idx, tag) in enumerate(tag_vocab))

    return token_to_index, tag_to_index
end

function sentences_to_indices(sentences, token_to_index, tag_to_index)
    X, Y = [], []

    for sentence in sentences
        tokens, tags = extract_tokens_and_labels(sentence)
        push!(X, [get(token_to_index, token, 0) for token in tokens])  # Use 0 for unknown words
        push!(Y, [get(tag_to_index, tag, 0) for tag in tags])          # Use 0 for unknown tags
    end

    return X, Y
end

function pad_sequences(sequences, max_length, padding_value=0)
    return [vcat(seq, fill(padding_value, max_length - length(seq))) for seq in sequences]
end

function prepare_batch(X, Y, batch_size, max_length)
    # Ensure batch size does not exceed dataset size
    batch_size = min(batch_size, length(X))
    
    # Randomly select indices for the batch
    indices = shuffle(1:length(X))[1:batch_size]
    
    # Prepare the batch with padded sequences
    X_batch = pad_sequences(X[indices], max_length)
    Y_batch = pad_sequences(Y[indices], max_length)
    return X_batch, Y_batch
end

all_sentences = dereferenced_data  # Assuming `dereferenced_data` is a list of sentence arrays
tokens, tags = extract_tokens_and_labels(all_sentences[1])

token_to_index, tag_to_index = build_vocabulary(all_sentences)
X, Y = sentences_to_indices(all_sentences, token_to_index, tag_to_index)

max_length = 30  # Adjust based on dataset
batch_size = 32
X_batch, Y_batch = prepare_batch(X, Y, batch_size, max_length)

hidden_size = 4096
input_size = length(X_batch)
output_size = length(Y_batch)
precision = 0.01

model = NNLib.NeuralNetwork.initialize_rnn(hidden_size, input_size, output_size, precision)

epochs = Inf
precision = 0.001

NNLib.Train.train_rnn!(model, X_batch, Y_batch, epochs, precision)
NNLib.SaveModel(model, "LLM_Model.jld2")