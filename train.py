import tensorflow as tf


def train_step(train_data, model, optimizer, loss_func, metric):
    """Train Step

    Parameters
    ----------
    train_data: ``BatchedDataset``
        A batched dataset of training examples with labels
    
    model: ``tf.keras.Model``
        A keras model

    optimizer: ``tf.keras.optimizers.Optimizer``
        A keras optimizer

    loss_func: ``callable``
        A loss function to optimize

    metric: ``callable``
        A metric function that takes as arguments
        `labels` and `logits`

    Returns
    -------
    Mean Loss over all the batches
    Mean accuracy over all the batches
    """
    # Lists to log the values of loss and metric for each batch
    loss_history = []
    metric_history = []

    # Train all the batches
    for i, batch in enumerate(train_data):
        # load the batch
        x_train, y_train = batch

        # Initialize the graph
        with tf.GradientTape() as tape:
            # Predict the labels
            preds = model(x_train)
            # Calculate loss
            loss = loss_func(y_train, preds)
        # Find gradients
        grads = tape.gradient(loss, model.trainable_variables)
        # Apply gradients
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Log the values of loss and metric for each batch
        loss_history.append(loss)
        metric_eval = metric(y_train, preds)
        metric_history.append(metric_eval)

        # Print the current values of loss and metric
        sys.stdout.write(f"\rStep {i+1}: \tloss={loss:.4f}\metric={metric_eval:.4f}")
    print("")

    # Find mean loss and metric accross the batches and log the values
    loss_val   = tf.reduce_mean(tf.convert_to_tensor(loss_history))
    metric_val = tf.reduce_mean(tf.convert_to_tensor(metric_history))
    print(f"Mean Loss: {loss_val:.4f}\tMean Metric: {metric_val}")

    # return the loss_value and metric_value
    return loss_val, metric_val


def test_step(test_data, model, loss_func, metric):
    """Evaluating the performance on test set

    Parameters
    ----------
    test_data: ``BatchedDataset``
        A batched dataset of testing examples with labels

    model: ``tf.keras.Model``
        A keras model

    loss_func: ``callable``
        A loss function to optimize

    metric: ``callable``
        A metric function that takes as arguments
        `labels` and `logits`

    Returns
    -------
    Mean Loss over all the batches
    Mean accuracy over all the batches
    """
    loss_over_batches     = []
    accuracy_over_batches = []

    for batch in test_data:
        x_test, y_test = batch
        preds = model(x_test)
        loss = loss_func(y_test, preds)
        loss_over_batches.append(loss)
        accuracy = metric(y_test, preds)

    loss_over_batches     = tf.convert_to_tensor(loss_over_batches)
    accuracy_over_batches = tf.convert_to_tensor(accuracy_over_batches)
    
    return tf.reduce_mean(loss_over_batches), tf.reduce_mean(accuracy_over_batches)
