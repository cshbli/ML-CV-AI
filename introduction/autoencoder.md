# Autoencoder

Autoencoder is a neural network designed to learn an identity function in an unsupervised way to reconstruct the original input while compressing the data in the process so as to discover a more efficient and compressed representation. 

It consists of two networks:

* <b>Encoder network</b>: It translates the original high-dimension input into the latent low-dimensional code. The input size is larger than the output size.

* <b>Decoder network</b>: The decoder network recovers the data from the code, likely with larger and larger output layers.

<img src="pic/autoencoder-architecture.png">

The encoder network essentially accomplishes the dimensionality reduction, just like how we would use Principal Component Analysis (PCA) or Matrix Factorization (MF) for. In addition, the autoencoder is explicitly optimized for the data reconstruction from the code. A good intermediate representation not only can capture latent variables, but also benefits a full decompression process.

## Denoising Autoencoder

Since the autoencoder learns the identity function, we are facing the risk of “overfitting” when there are more network parameters than the number of data points.

To avoid overfitting and improve the robustness, Denoising Autoencoder (Vincent et al. 2008) proposed a modification to the basic autoencoder. The input is partially corrupted by adding noises to or masking some values of the input vector in a stochastic manner, then the model is trained to recover the original input (note: not the corrupt one).

<img src="pic/denoising-autoencoder-architecture.png">

## Variational Autoencoder (VAE)

<img src="pic/vae-gaussian.png">

## References

* [From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)