module Transformer

using ..Activations
using ..Optimizer

# Define Multi-head Attention
struct MultiHeadAttention
    num_heads::Int
    head_dim::Int
    W_q::Array{Float64}  # Query weights
    W_k::Array{Float64}  # Key weights
    W_v::Array{Float64}  # Value weights
    W_o::Array{Float64}  # Output weights
end

function MultiHeadAttention(num_heads::Int, input_dim::Int, head_dim::Int)
    W_q = randn(input_dim, num_heads * head_dim)
    W_k = randn(input_dim, num_heads * head_dim)
    W_v = randn(input_dim, num_heads * head_dim)
    W_o = randn(num_heads * head_dim, input_dim)
    return MultiHeadAttention(num_heads, head_dim, W_q, W_k, W_v, W_o)
end

function attention(Q, K, V)
    scores = Q * K' ./ sqrt(size(K, 2))
    weights = softmax(scores, dim=2)  # Using softmax from your activations
    return weights * V
end

function forward(mha::MultiHeadAttention, x)
    Q = x * mha.W_q
    K = x * mha.W_k
    V = x * mha.W_v
    heads = [attention(Q[:, (i-1)*mha.head_dim+1:i*mha.head_dim],
                       K[:, (i-1)*mha.head_dim+1:i*mha.head_dim],
                       V[:, (i-1)*mha.head_dim+1:i*mha.head_dim]) for i in 1:mha.num_heads]
    concat_heads = hcat(heads...)
    return concat_heads * mha.W_o
end

# Define Transformer Encoder Layer
struct TransformerEncoderLayer
    self_attention::MultiHeadAttention
    feedforward::Function
end

function TransformerEncoderLayer(input_dim::Int, num_heads::Int, head_dim::Int, hidden_dim::Int)
    attention = MultiHeadAttention(num_heads, input_dim, head_dim)
    feedforward = x -> relu(x * randn(input_dim, hidden_dim)) * randn(hidden_dim, input_dim)
    return TransformerEncoderLayer(attention, feedforward)
end

function forward(layer::TransformerEncoderLayer, x)
    # Self-attention
    attn_output = forward(layer.self_attention, x)
    x = x + attn_output  # Add residual connection
    # Feedforward
    ff_output = layer.feedforward(x)
    return x + ff_output  # Add residual connection
end

# Define Full Transformer
struct TransformerModel
    encoder_layers::Vector{TransformerEncoderLayer}
end

function TransformerModel(num_layers::Int, input_dim::Int, num_heads::Int, head_dim::Int, hidden_dim::Int)
    layers = [TransformerEncoderLayer(input_dim, num_heads, head_dim, hidden_dim) for _ in 1:num_layers]
    return TransformerModel(layers)
end

function forward(model::TransformerModel, x)
    for layer in model.encoder_layers
        x = forward(layer, x)
    end
    return x
end

end # module
