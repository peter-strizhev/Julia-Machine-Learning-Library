module Train

using ..DataUtils
using ..NeuralNetwork
using ..Optimizer

function train!(model, X, y, optimizer, epochs, batch_size, target_loss=0.09)
    epoch = 1
    loss = Inf  # Initialize loss to a high value to enter the loop
    
    while loss > target_loss && epoch <= epochs
        for (X_batch, y_batch) in DataUtils.batch_generator(X, y, batch_size)
            # Perform a forward pass and calculate the activations
            activations_list = NeuralNetwork.forward_pass(model, X_batch)

            # Compute the loss (using Mean Squared Error for example)
            loss = sum((activations_list[end] .- y_batch) .^ 2) / size(X_batch, 1)

            # Compute the gradients
            gradients_w, gradients_b = NeuralNetwork.backward_pass(model, X_batch, y_batch, activations_list)

            # Update model parameters using the optimizer
            for i in 1:length(model.layers)
                Optimizer.update!(optimizer, model.layers[i], gradients_w[i], model.biases[i], vec(gradients_b[i]))
            end
            
            # Print the current epoch and loss on a single line with carriage return
            print("\rEpoch: $epoch, Loss: $loss")
            flush(stdout)  # Ensure output is immediately written
        end
        epoch += 1
    end

    # Final message depending on whether target loss was reached
    if loss <= target_loss
        println("\nTraining stopped early. Loss has reached the target value of $target_loss.")
    else
        println("\nTraining completed after $epochs epochs. Final loss: $loss")
    end
end

end  # module Train
