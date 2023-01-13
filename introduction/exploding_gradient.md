# Exploding Gradient

Exploding gradients are a problem where large error gradients accumulate and result in very large updates to neural network model weights during training.

This has the effect of your model being unstable and unable to learn from your training data.

The explosion occurs through exponential growth by repeatedly multiplying gradients through the network layers that have values larger than 1.0.

## How do You Know if You Have Exploding Gradients?

There are some subtle signs that you may be suffering from exploding gradients during the training of your network, such as:

- The model is unable to get traction on your training data (e.g. poor loss).
- The model is unstable, resulting in large changes in loss from update to update.
- The model loss goes to NaN during training.

If you have these types of problems, you can dig deeper to see if you have a problem with exploding gradients.

There are some less subtle signs that you can use to confirm that you have exploding gradients.

- The model weights quickly become very large during training.
- The model weights go to NaN values during training.
- The error gradient values are consistently above 1.0 for each node and layer during training.

## How to Fix Exploding Gradients?

There are many approaches to addressing exploding gradients; this section lists some best practice approaches that you can use.

### Re-Design the Network Model

In deep neural networks, exploding gradients may be addressed by redesigning the network to have `fewer` layers.

There may also be some benefit in using a `smaller` batch size while training the network.

In recurrent neural networks, updating across fewer prior time steps during training, called `truncated Backpropagation through time`, may reduce the exploding gradient problem.

### Use Long Short-Term Memory Networks

In recurrent neural networks, gradient exploding can occur given the inherent instability in the training of this type of network, e.g. via Backpropagation through time that essentially transforms the recurrent network into a deep multilayer Perceptron neural network.

Exploding gradients can be reduced by using the Long Short-Term Memory (LSTM) memory units and perhaps related gated-type neuron structures.

Adopting LSTM memory units is a new best practice for recurrent neural networks for sequence prediction.

### Use Gradient Clipping

Exploding gradients can still occur in very deep Multilayer Perceptron networks with a large batch size and LSTMs with very long input sequence lengths.

If exploding gradients are still occurring, you can check for and limit the size of gradients during the training of your network.

This is called gradient clipping.

In the Keras deep learning library, you can use gradient clipping by setting the clipnorm or clipvalue arguments on your optimizer before training.

Good default values are clipnorm=1.0 and clipvalue=0.5.

### Use Weight Regularization

Another approach, if exploding gradients are still occurring, is to check the size of network weights and apply a penalty to the networks loss function for large weight values.

This is called weight regularization and often an L1 (absolute weights) or an L2 (squared weights) penalty can be used.

In the Keras deep learning library, you can use weight regularization by setting the kernel_regularizer argument on your layer and using an L1 or L2 regularizer.

## References

- [A Gentle Introduction to Exploding Gradients in Neural Networks](https://machinelearningmastery.com/exploding-gradients-in-neural-networks/)