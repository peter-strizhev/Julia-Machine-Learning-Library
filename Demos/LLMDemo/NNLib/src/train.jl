module Train

using ..DataUtils
using ..NeuralNetwork
using ..Optimizer
using ..Activations

function train!(model::NeuralNetwork.NeuralNetworkModel, X, y, optimizer, epochs, batch_size, target_loss=0.05, min_lr=1e-6, decay_factor=0.99, patience=10, loss_threshold=1e-6)
    epoch = 1
    local loss::Float64  # Initialize loss to a high value to enter the loop
    loss = Inf64
    previous_loss = Inf  # Initialize previous loss
    epochs_since_improvement = 0  # Counter to track the number of epochs since the last improvement

    while loss > target_loss && epoch <= epochs
        for (X_batch, y_batch) in DataUtils.batch_generator(X, y, batch_size)
            # Perform a forward pass and calculate the activations
            activations_list = NeuralNetwork.forward_pass(model, X_batch)

            # Compute the loss (using Mean Squared Error)
            loss = sum((activations_list[end] .- y_batch) .^ 2) / size(X_batch, 1)

            # Compute the gradients
            gradients_w, gradients_b = NeuralNetwork.backward_pass(model, X_batch, y_batch, activations_list)

            # Update model parameters using the optimizer
            for i in 1:length(model.layers)
                Optimizer.update!(optimizer, model.layers[i], gradients_w[i], model.biases[i], vec(gradients_b[i]))
            end
            
            # Print the current epoch and loss on a single line with carriage return
            lr = optimizer.lr
            print("\rEpoch: $epoch, Loss: $loss, Optimizer LR: $lr")
            flush(stdout)  # Ensure output is immediately written
        end

        # Calculate the change in loss compared to the previous epoch
        loss_change = abs(previous_loss - loss)
        
        # Check if the change in loss is smaller than the threshold
        if loss_change > loss_threshold
            epochs_since_improvement = 0  # Reset counter if loss is improving significantly
        else
            epochs_since_improvement += 1  # Increment counter if loss improvement is too small
            
            # If no significant improvement for a certain number of epochs (patience), decay the learning rate
            if epochs_since_improvement >= patience && optimizer.lr > min_lr
                optimizer.lr *= decay_factor
                # println("\nLearning rate decayed to: ", optimizer.lr)
                epochs_since_improvement = 0  # Reset counter after decay
            end
        end
        
        # Store the current loss for next comparison
        previous_loss = loss
        epoch += 1
    end

    # Final message depending on whether target loss was reached
    if loss <= target_loss
        println("\nTraining stopped early. Loss has reached the target value of $target_loss.")
    else
        println("\nTraining completed after $epochs epochs. Final loss: $loss")
    end
end

function train_rnn!(model::NeuralNetwork.RecurrentNeuralNetworkModel, 
                    X_batch::Vector{Vector{Int64}}, 
                    Y_batch::Vector{Vector{Int64}}, 
                    num_epochs::Int64, 
                    learning_rate::Float64)
    Wxh, Whh, bh, Why, by = model.Wxh, model.Whh, model.bh, model.Why, model.by
    hidden_size = model.hidden_size
    input_size = size(Wxh, 2)  # Input size should match the number of columns of Wxh

    for epoch in 1:num_epochs
        total_loss = 0.0

        # Loop over each sequence in X_batch and Y_batch
        for i in 1:length(X_batch)
            # Get the inputs and targets for the current sequence
            inputs = X_batch[i]
            targets = Y_batch[i]
            
            # Convert to Float64 for model compatibility
            inputs = [Float64(x) for x in inputs]  # Convert to Float64
            targets = [Float64(y) for y in targets]  # Convert to Float64
            
            # Initialize gradients
            dWxh = zeros(size(Wxh))
            dWhh = zeros(size(Whh))
            dbh = zeros(size(bh))
            dWhy = zeros(size(Why))
            dby = zeros(size(by))

            # Forward pass
            # TODO: Call forward pass function
            # y_pred, h_states, y_pred = NeuralNetwork.rnn_forward(model, inputs, hprev)

            # total_loss = NeuralNetwork.rnn_forward(model, inputs, hidden_size)

            h_prev = zeros(Float64, hidden_size)  # Initialize h_prev as Float64 vector
            h_states = Vector{Vector{Float64}}(undef, length(inputs))
            y_pred = Vector{Vector{Float64}}(undef, length(inputs))
            loss = 0.0

            for t in 1:length(inputs)
                # Convert inputs[t] to a 1x1 column vector
                input_t = [Float64(inputs[t])]  # Convert scalar to 1x1 column vector

                # Ensure matrix-vector multiplication produces correct dimensions and apply tanh
                h_prev = Activations.tanh(Wxh * input_t + Whh * h_prev + bh)  # tanh is applied to a vector result
                h_states[t] = h_prev
                y_pred[t] = Why * h_prev + by

                # Compute loss (mean squared error)
                loss += sum((y_pred[t] .- targets[t]).^2)
            end
            total_loss += loss / length(inputs)

            # Backpropagation through time
            # TODO: Call back propagation

            dh_next = zeros(Float64, hidden_size)
            for t in reverse(1:length(inputs))
                # Output layer gradients
                dy = 2 * (y_pred[t] .- targets[t])
                dWhy += dy * h_states[t]'
                dby += dy

                # Hidden layer gradients
                dh = Why' * dy + dh_next
                dh_raw = (1 .- h_states[t].^2) .* dh  # Derivative of tanh
                input_t = [Float64(inputs[t])]  # Ensure it's treated as a column vector
                dWxh += dh_raw * input_t'
                dWhh += dh_raw * (t > 1 ? h_states[t-1] : zeros(Float64, hidden_size))'
                dbh += dh_raw
                dh_next = Whh' * dh_raw
            end

            # Update parameters using gradient descent
            model.Wxh -= learning_rate * dWxh
            model.Whh -= learning_rate * dWhh
            model.bh -= learning_rate * dbh
            model.Why -= learning_rate * dWhy
            model.by -= learning_rate * dby
        end
        print("\rEpoch: $epoch, Loss: $total_loss")
        flush(stdout)  # Ensure output is immediately written
    end
end

function train_transformer(model, data, labels, num_epochs::Int, optimizer)
    for epoch in 1:num_epochs
        for (x, y) in zip(data, labels)
            preds = forward(model, x)
            loss = mean((preds .- y).^2)
            grads = gradient(() -> loss, model)  # Backpropagation
            optimizer.update!(model, grads)
        end
        println("Epoch $epoch: Loss = $loss")
    end
end


end  # module Train