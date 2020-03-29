# Normalization 
 <p align="center">
   <img src="visual_comparison_of_normalizations.png" width="500px" title="A visual comparison of various normalization methods">
 </p>

## Benefits of using normalization
* It makes the Optimization faster because normalization doesn’t allow weights to explode all over the place and restricts them to a certain range.

* It normalizes each feature so that they maintains the contribution of every feature, as some feature has higher numerical value than others. This way our network can be unbiased(to higher value features).

* It reduces Internal Covariate Shift. It is the change in the distribution of network activations due to the change in network parameters during training. To improve the training, we seek to reduce the internal covariate shift.

* Batch Norm makes loss surface smoother(i.e. it bounds the magnitude of the gradients much more tightly).

* An unintended benefit of Normalization is that it helps network in Regularization(only slightly, not significantly).

 From above, we can conclude that getting Normalization right can be a crucial factor in getting your model to train effectively, but this isn’t as easy as it sounds. Let me support this by certain questions.
 
* How Normalization layers behave in Distributed training ?

* Which Normalization technique should you use for your task like CNN, RNN, style transfer etc ?

* What happens when you change the batch size of dataset in your training ?

* Which norm technique would be the best trade-off for computation and accuracy for your network ?

## Batch Normalization

  The mainstream normalization technique for almost all convolutional neural networks today is <b>Batch Normalization (BN)</b>, which has been widely adopted in the development of deep learning. Proposed by Google in 2015, BN can not only accelerate a model’s converging speed, but also alleviate problems such as Gradient Dispersion in the deep neural network, making it easier to train models.
 
  Batch normalization is a method that normalizes activations in a network across the mini-batch of definite size. For each feature, batch normalization computes the mean and variance of that feature in the mini-batch. It then subtracts the mean and divides the feature by its mini-batch standard deviation.
 <p align="center">
   <img src="batch_normalization_formula.png" width="500px" title="Simple Batch Normalization">
 </p>

  But wait, what if increasing the magnitude of the weights made the network perform better?
  
  To solve this issue, we can add γ and β as scale and shift learn-able parameters respectively. This all can be summarized as:
  <p align="center">
   <img src="batch_normalization_function.png" width="400px" title="Batch Normalization">
  </p>
  
### Problems associated with Batch Normalization
* Variable Batch Size → If batch size is of 1, then variance would be 0 which doesn’t allow batch norm to work. Furthermore, if we have small mini-batch size then it becomes too noisy and training might affect. 

* BN cannot ensure the model accuracy rate when the batch size becomes smaller. As a result, researchers today are normalizing with large batches, which is very memory intensive.

* There would also be a problem in distributed training. As, if you are computing in different machines then you have to take same batch size because otherwise γ and β will be different for different systems.

* Recurrent Neural Network → In an RNN, the recurrent activations of each time-step will have a different story to tell(i.e. statistics). This means that we have to fit a separate batch norm layer for each time-step. This makes the model more complicated and space consuming because it forces us to store the statistics for each time-step during training.
 
     
 GN divides channels — also referred to as feature maps that look like 3D chunks of data — into groups and normalizes the features within each group. GN only exploits the layer dimensions, and its computation is independent of batch sizes.     

 Layer Normalization (LN), proposed in 2016 by a University of Toronto team led by Dr. Geoffrey Hinton; and Instance Normalization (IN), proposed by Russian and UK researchers, are also alternatives for normalizing batch dimensions. While LN and IN are effective for training sequential models such as RNN/LSTM or generative models such as GANs, GN appears to present a better result in visual recognition.
