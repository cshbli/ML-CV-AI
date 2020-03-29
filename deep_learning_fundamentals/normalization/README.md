# Normalization 
 <p align="center">
   <img src="visual_comparison_of_normalizations.png" width="500px" title="A visual comparison of various normalization methods">
 </p>

## Benefits of using normalization
 The mainstream normalization technique for almost all convolutional neural networks today is <b>Batch Normalization (BN)</b>, which has been widely adopted in the development of deep learning. Proposed by Google in 2015, BN can not only accelerate a model’s converging speed, but also alleviate problems such as Gradient Dispersion in the deep neural network, making it easier to train models.

 <b>BN cannot ensure the model accuracy rate when the batch size becomes smaller. As a result, researchers today are normalizing with large batches, which is very memory intensive.</b>
     
 GN divides channels — also referred to as feature maps that look like 3D chunks of data — into groups and normalizes the features within each group. GN only exploits the layer dimensions, and its computation is independent of batch sizes.     

 Layer Normalization (LN), proposed in 2016 by a University of Toronto team led by Dr. Geoffrey Hinton; and Instance Normalization (IN), proposed by Russian and UK researchers, are also alternatives for normalizing batch dimensions. While LN and IN are effective for training sequential models such as RNN/LSTM or generative models such as GANs, GN appears to present a better result in visual recognition.
