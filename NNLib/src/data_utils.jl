module DataUtils

# Function to normalize data (scales values between 0 and 1)
function normalize(X::AbstractArray)
    min_val = minimum(X, dims=1)
    max_val = maximum(X, dims=1)
    return (X .- min_val) ./ (max_val .- min_val)
end

# Define a struct for batch generation
struct BatchGenerator
    X::AbstractArray
    y::AbstractArray
    batch_size::Int
end

# Implement the iteration protocol for BatchGenerator
function Base.iterate(generator::BatchGenerator, state=1)
    start_idx = state
    if start_idx > size(generator.X, 1)
        return nothing  # Signals end of iteration
    end
    end_idx = min(start_idx + generator.batch_size - 1, size(generator.X, 1))
    batch_indices = start_idx:end_idx
    return (generator.X[batch_indices, :], generator.y[batch_indices, :]), end_idx + 1
end

# Convenience function to create a batch generator
function batch_generator(X::AbstractArray, y::AbstractArray, batch_size::Int)
    return BatchGenerator(X, y, batch_size)
end

end  # module DataUtils
