# Normalization 
  * [Benefits of using normalization](#benefits-of-using-normalization)
  * [Batch Normalization](#batch-normalization)
    * [Fused Batch Normalization](#fused-batch-normalization)
  * [Layer Normalization](#layer-normalization)
  * [Instance Normalization](#instance-normalization)
  * [Group Normalization](#group-normalization)
  * [Batch-Instance Normalization](#batch-instance-normalization)
  * [Switchable Normalization](#switchable-normalization)
 
 <p align="center">
   <img src="visual_comparison_of_normalizations.png" width="800px" title="A visual comparison of various normalization methods">
 </p>

 We assume that the activations at any layer would be of the dimensions NxCxHxW (and, of course, in the real number space), where, 
 - N = Batch Size, 
 - C = Number of Channels (filters) in that layer, 
 - H = Height of each activation map, 
 - W = Width of each activation map.

 Generally, normalization of activations require `shifting` and `scaling` the activations by `mean` and `standard deviation` respectively. Batch Normalization, Instance Normalization and Layer Normalization differ in the manner these statistics are calculated.

 <p align="center">
  <img src="1_3ieGJOruPtmgTYlb7ZTtnw.webp">
 </p>

## Benefits of using normalization
* It makes the Optimization faster because normalization doesn’t allow weights to explode all over the place and restricts them to a certain range.
 <p align="center">
  <img src="gradient_update_curvature.png" width="600px" title="Same gradient can actually cause the loss to increase depending on the curvature and setup-size">
 </p>

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

<p align="center">
<img src="1_a7tkJTGmDLD4ovMmGCgziA.webp">
<img src="1_JqbhYjs4yYieoAG1tjzkkA.webp">
</p>

In “Batch Normalization”, mean and variance are calculated for each individual channel across all samples and both spatial dimensions.

  The mainstream normalization technique for almost all convolutional neural networks today is <b>Batch Normalization (BN)</b>, which has been widely adopted in the development of deep learning. Proposed by Google in 2015, BN can not only accelerate a model’s converging speed, but also alleviate problems such as Gradient Dispersion in the deep neural network, making it easier to train models.
 
  Batch normalization is a method that <b>normalizes activations</b> in a network across the mini-batch of definite size. For each feature, batch normalization computes the mean and variance of that feature in the mini-batch. It then subtracts the mean and divides the feature by its mini-batch standard deviation.
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

### Fused Batch Normalization

Fused batch norm combines the multiple operations needed to do batch normalization into a single kernel. Batch norm is an expensive process that for some models makes up a large percentage of the operation time. Using fused batch norm can result in a 12%-30% speedup.

### Folding in Convolution Layer

One of the advantages of Batch Normalization is that it can be folded in a convolution layer. This means that we can replace the Convolution followed by Batch Normalization operation by just one convolution with different weights.

To prove this, we only need a few equations. We keep the same notations as algorithm 1 above. Below, in (1) we explicit the batch norm output as a function of its input. (2) Locally, we can define the input of BatchNorm as a product between the convolution weights and the previous activations, with an added bias. We can thus express in (3) the BatchNorm output as a function of the convolution input which we can factor as equation (4) with new weights W’ and b’ described in (5) and (6).

<p align="center">
   <img src="folding_formulas_no_b-700x453.png" width="500px" title="Folding Batch Normalization">
</p>

To fold batch normalization there is basically three steps:

- Given a TensorFlow graph, filter the variables that need folding,
- Fold the variables,
- Create a new graph with the folded variables.

We need to filter the variables that require folding. When using batch normalization, it creates variables with names containing moving_mean and moving_variance. You can use this to extract fairly easily the variables from layers that used batch norm.

Now that you know which layers used batch norm, for every such layer, you can extract its weights `W`, bias `b`, batch norm variance `v`, mean `m`, `gamma` and `beta` parameters. You need to create a new variable to store the folded weights and biases as follow:

```
W_new = gamma * W / var
b_new = gamma * (b - mean) / var + beta
```

The last step consists in creating a new graph in which we deactivate batch norm and add bias variables if necessary –which should be the case for every foldable layer since using bias with batch norm is pointless.

The whole code should look something like below. Depending on the parameters used for the batch norm, your graph may not have `gamma` or `beta`.

```
# ****** (1) Get variables ******
variables = {v.name: session.run(v) for v in tf.global_variables()}

# ****** (2) Fold variables ******
folded_variables = {}
for v in variables.keys():
    if not v.endswith('moving_variance:0'):
        continue

    n = get_layer_name(v) # 'model/conv1/moving_variance:0' --> 'model/conv1'

    W = variable[n + '/weights:0'] # or "/kernel:0", etc.
    b = variable[n + '/bias:0'] # if a bias existed before
    gamma = variable[n + '/gamma:0']
    beta = variable[n + '/beta:0']
    m = variable[n + '/moving_mean:0']
    var = variable[n + '/moving_variance:0']

    # folding batch norm
    W_new = gamma * W / var
    b_new = gamma * (b - mean) / var + beta # remove `b` if no bias
    folded_variables[n + '/weights:0'] = W_new        
    folded_variables[n + '/bias:0'] = b_new   

    # ****** (3) Create new graph ******
    new_graph = tf.Graph()
    new_session = tf.Session(graph=new_graph) 
    network = ... # instance batch-norm free graph with bias added.
                  # Careful, the names should match the original model

    for v in tf.global_variables():
        try:
            new_session.run(v.assign(folded_variables[v.name]))
        except:
            new_session.run(v.assign(variables[v.name]))
```            

## Layer Normalization

<p align="center">
<img src="1_Fe3rXQBU15z4CoiG3zInKg.webp">
</p>

In “Layer Normalization”, mean and variance are calculated for each individual sample across all channels (and maybe both spatial dimensions).

 Layer normalization normalizes input across the features instead of normalizing input features across the batch dimension in batch normalization.
 
 A mini-batch consists of multiple examples with the same number of features. Mini-batches are matrices(or tensors) where one axis corresponds to the batch and the other axis(or axes) correspond to the feature dimensions.
 <p align="center">
  <img src="layer_normalization_formula.png" width="200px" title="Layer Normalization">
 </p>
     
 Layer normalization performs better than batch norm in case of <b>RNNs</b>.
 <p align="center">
  <img src="batch_normalization_vs_layer_normalization_example.png" width="600px" title="Difference between batch normalization and layer normalization">
 </p> 
 
## Instance Normalization

<p align="center">
<img src="1_wa1PwStln3dWKkqEKrlinA.webp">
<img src="1_H8WrL_Xqxdle8qWgMr82tA.webp">
</p>

In “Instance Normalization”, mean and variance are calculated for each individual channel for each individual sample across both spatial dimensions.

 Layer normalization and instance normalization is very similar to each other but the difference between them is that instance normalization normalizes across each channel in each training example instead of normalizing across input features in an training example. Unlike batch normalization, the instance normalization layer is applied at test time as well(due to non-dependency of mini-batch).

 <p align="center">
  <img src="instance_normalization_formula.png" width="600px" title="Instance Normalization">
 </p>
 
  Here, x∈ ℝ T ×C×W×H be an input tensor containing a batch of T images. Let xₜᵢⱼₖ denote its tijk-th element, where k and j span spatial dimensions(Height and Width of the image), i is the feature channel (color channel if the input is an RGB image), and t is the index of the image in the batch.
  
 This technique is originally devised for <b>style transfer</b>, the problem instance normalization tries to address is that the network should be agnostic to the contrast of the original image.
 
## Group Normalization

 Group Normalization normalizes over group of channels for each training examples. We can say that, Group Norm is in between Instance Norm and Layer Norm.
 
 When we put all the channels into a single group, group normalization becomes Layer normalization. And, when we put each channel into different groups it becomes Instance normalization.
 
 GN divides channels — also referred to as feature maps that look like 3D chunks of data — into groups and normalizes the features within each group. GN only exploits the layer dimensions, and its computation is independent of batch sizes.     
 <p align="center">
  <img src="group_normalization_formula.png" width="400px" title="Group Normalization">
  <img src="group_normalization_formula_2.png" width="400px" title="Group Normalization 2">
  <img src="group_normalization_formula_3.png" width="150px" title="Group Normalization 3">
  <img src="group_normalization_formula_4.png" width="150px" title="Group Normalization 4">
 </p>

  Here, x is the feature computed by a layer, and i is an index. In the case of 2D images, i = (iN , iC , iH, iW ) is a 4D vector indexing the features in (N, C, H, W) order, where N is the batch axis, C is the channel axis, and H and W are the spatial height and width axes. G is the number of groups, which is a pre-defined hyper-parameter. C/G is the number of channels per group. ⌊.⌋ is the floor operation, and “⌊kC/(C/G)⌋= ⌊iC/(C/G)⌋” means that the indexes i and k are in the same group of channels, assuming each group of channels are stored in a sequential order along the C axis. GN computes µ and σ along the (H, W) axes and along a group of C/G channels.

## Batch-Instance Normalization
 Batch-Instance Normalization is just an interpolation between batch norm and instance norm.
 <p align="center">
  <img src="batch_instance_normalization_formula.png" width="400px" title="Batch-Instance Normalization"
 </p>
 
 The interesting aspect of batch-instance normalization is that the balancing parameter ρ is learned through gradient descent.
 
 <b>From batch-instance normalization, we can conclude that models could learn to adaptively use different normalization methods using gradient descent.</b>
 
## Switchable Normalization
 Switchable normalization uses a weighted average of different mean and variance statistics from batch normalization, instance normalization, and layer normalization.
 
 switch normalization could potentially outperform batch normalization on tasks such as image classification and object detection.
 
 The instance normalization were used more often in earlier layers, batch normalization was preferred in the middle and layer normalization being used in the last more often. Smaller batch sizes lead to a preference towards layer normalization and instance normalization.

## References
* [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf%27) Sergey Ioffe, Christian Szegedy.
* [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf) Jimmy Lei Ba, Jamie Ryan Kiros and Geoffrey E. Hinton.
* [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/pdf/1607.08022.pdf) Dmitry Ulyanov, Andrea Vedaldi.
* [Group Normalization](https://arxiv.org/pdf/1803.08494.pdf) Yuxin Wu, Kaiming He.
* [Batch-Instance Normalization for Adaptively Style-Invariant Neural Networks](https://arxiv.org/pdf/1805.07925.pdf) Hyeonseob Nam, Kyo-Eun Kim
* [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/pdf/1602.07868.pdf) Tim Salimans, Diederik P. Kingma.
* [Batch Normalization, Instance Normalization, Layer Normalization: Structural Nuances](https://becominghuman.ai/all-about-normalization-6ea79e70894b)