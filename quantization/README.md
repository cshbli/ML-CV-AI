# Model Quantization
  * Quantization for deep learning is the process of approximating a neural network that uses floating-point numbers by a neural network of low bit width numbers.
  * [Quantization Algorithms](https://nervanasystems.github.io/distiller/algo_quantization.html)
  * [Quantizing deep convolutional networks for efficient inference: A white paper](https://arxiv.org/pdf/1806.08342.pdf)
    * Overview of schemes for model quantization: One can quantize weights post training (left) or quantize weights and activations post training (middle). It is also possible to perform quantization aware training for improved accuracy.
    
    <p align="center">
       <img src="./figs/tensorflow_overview_of_schemes_for_model_quantization.png" width="600px" title="Overview of schemes for model quantization">
    </p>
    
    * Weight only quantization: per-channel quantization provides good accuracy, with asymmetric quantization providing close to floating point accuracy.
    
    <p align="center">
       <img src="./figs/weight_only_quantization.png" width="600px" title="weight only quantization">
    </p>
    
    *  Post training quantization of weights and activations: per-channel quantization of weights and per-layer quantization of activations works well for all the networks considered, with asymmetric quantization providing slightly better accuracies.
    
    <p align="center">
       <img src="./figs/post_training_quantization_of_weights_and_activation.png" width="600px" title="post training quantization of weights and activations">
    </p>
    
  * Continuous-discrete learning
    * During training there are effectively two networks : float-precision and binary-precision. The binary-precision is updated in the forward pass using the float-precision, and the float-precision is updated in the backward pass using the binary-precision. In this sense, the training is a type of alternating optimization.
    
    <p align="center">
       <img src="./figs/quantization_of_parameteres during training.png" width="400px" title="Quantization of parameters during training">
    </p>
    
     * Ternary parameter networks are an alternative to binary parameters that allow for higher accuracy, at cost of larger model size.
    <p align="center">
       <img src="./figs/ternary_quantization_for_gaussain_distributed_parameters.png" width="600px" title="Ternary quantization for Gaussian-distributed parameters">
    </p>
    
  * Quantized activations
 
    <p align="center">
       <img src="./figs/quantization_of_activation.png" width="400px" title="Quantization of activations, allowing binary calculations, with integer accumulations. The binary calculation can be convolutional layer or fully-connected layer">
    </p>
    
  * Multiple bit width activations
 
    <p align="center">
       <img src="./figs/multiple_bit_width_activations.png" width="400px" title="DoReFa-Net style 3-bit activation quantizer function">
    </p>
   
## References

### Documentation

### Sites

### Conferences

### Online Courses
