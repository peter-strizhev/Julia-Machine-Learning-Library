module Train

using ..DataUtils
using ..NeuralNetwork
using ..Optimizer

function train!(model, X, y, optimizer, epochs, batch_size)
    for epoch in 1:epochs
        # (X_batch, y_batch) = batch_generator(X, y, batch_size)
        # println(first(X_batch))
        # activations_list = forward_pass(model, first(X_batch))

        for (X_batch, y_batch) in DataUtils.batch_generator(X, y, batch_size)
            # Perform a forward pass and calculate the activations
            activations_list = NeuralNetwork.forward_pass(model, X_batch)

            # Compute the loss (using Mean Squared Error for example)
            loss = sum((activations_list[end] .- y_batch) .^ 2) / size(X_batch, 1)

            # Compute the gradients
            gradients_w, gradients_b = NeuralNetwork.backward_pass(model, X_batch, y_batch, activations_list)

            # Update model parameters using the optimizer
            # for i in 1:length(model.layers)
            #     println(model.layers[i])
            #     Optimizer.update!(optimizer, model.layers[i], gradients_w[i], gradients_b[i])
            # end
            for i in 1:length(model.layers)
                Optimizer.update!(optimizer, model.layers[i], gradients_w[i], model.biases[i], vec(gradients_b[i]))
            end
            # Optionally, print the loss for monitoring
            println("Epoch: $epoch, Loss: $loss")
        end
    end
end

end  # module Train
