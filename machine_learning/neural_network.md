# Artificial Neural Network

## The Neuron 

The very first step to grasping what an artificial neural network does is to understand the neuron. Neural networks in computer science mimic actual human brain neurons, hence the name “neural” network. 

Neurons have branches coming out of them from both ends, called dendrites. One neuron can’t do much, but when thousands of neurons connect and work together, they are powerful and can process complex actions and concepts. 

<p align="center">
<img src="pic/Group-26.webp">
</p>

For our digital neuron, independent values (input signals) get passed through the “neuron” in order to generate a dependent value (output signal). These independent variables in one layer are just a row of data for one single observation. For example, in the context of a neural network problem, one input layer would signify one variable – maybe the age or gender (independent variable) of a person whose identity (dependent) we are trying to figure out. 

This neural network is then applied as many times as the amount of data points we have per independent variable. So what would be the output value, since we now know what the input value signifies? Output values can be continuous, binary, or categorical variables. They just have to correspond to the one row you input as the independent variables. 

<p align="center">
<img src="pic/Group-23-1.webp">
</p>

## Weights and Activation 

The next thing you need to know are what goes in the synapses. The synapses are those lines connecting the “neuron” to the input signals. Weights are assigned to all of them. Weights are crucial to artificial neural networks because they let the networks “learn.” The weights decide which input signals are not important – which ones get passed along and which don’t.

<p align="center">
<img src="pic/Group-21.webp">
</p>

What happens in the neuron? The first step is that all the values passing through get summed. In other words, it takes the weighted sum of all the input values. It then applies an activation function. Activation functions are just functions applied to the weighted sum. Depending on the outcome of the applied function, the neuron will either pass on a signal or it won’t pass it on.

<p align="center">
<img src="pic/Group-17.webp">
</p>

Most machine learning algorithms can be done in this type of form, with an array of input signals going through an activation function (which can be anything: logistic regression, polynomial regression, etc), and an output signal at the end.

## How do Artificial Neural Networks Learn?

<p align="center">
<img src="pic/artificial-neural-network-concept.png">
</p>

In order to learn, the predicted value created by the process above gets compared to the actual value, which is given as a test variable. The model keeps checking itself against the actual value, and makes modifications with each correct or incorrect value. It does this by calculating the cost function.

## Cost Function to determine performance 

In machine learning, cost functions are used to estimate how successfully models are performing. The cost function’s purpose is to calculate the error we get from our prediction. Our goal is to minimize the cost function. The smaller the output of the cost function, the closer the predicted value is to the actual value. Once we’ve compared, we feed this information back into the neural network. As we feed the results of the cost function into the neural network, it updates the weights. 

## References

- [Artificial Neural Network: Everything you need to know](https://viso.ai/deep-learning/artificial-neural-network/)