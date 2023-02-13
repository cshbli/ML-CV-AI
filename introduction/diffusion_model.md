# Diffusion Model

The rise of the Diffusion Model can be regarded as the main factor for the recent breakthrough in the AI generative artworks field.

Diffusion models define a Markov chain of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise. Unlike VAE or flow models, diffusion models are learned with a fixed procedure and the latent variable has high dimensionality (same as the original data).

<img src="pic/generative-overview.png">

## Overview

The training of the Diffusion Model can be divided into two parts:

* Forward Diffusion Process → add noise to the image.

* Reverse Diffusion Process → remove noise from the image.

<img src="pic/1_xc2Y6jwIUhfEGxJLytU1RA.webp">

## Forward Diffusion Process

<img src="pic/1_zghUHsVaD9c1ebV_ixrkxg.webp">

## Reverse Diffusion Process

<img src="pic/1_CfarzOOYM-_ytQQC0vpOHg.webp">

## The U-Net Model

In each epoch:

* A random time step t will be selected for each training sample (image).

* Apply the Gaussian noise (corresponding to t) to each image.

* Convert the time steps to embeddings (vectors).

<img src="pic/1_QyBy1sdErujooBHqt5XAOg.webp">

<img src="pic/1_AQH76e-TTTUAZ09uZ7tzgA.webp">

<img src="pic/1_S8XcA8cQIjHLxUsOUPk39Q.webp">

## Summary

* The Diffusion model is divided into two parts: forward diffusion and reverse diffusion.

* The forward diffusion can be done using the closed-form formula.

* The backward diffusion can be done using a trained neural network.

* To approximate the desired denoising step q, we just need to approximate the noise εₜ using a neural network εθ.

* Training on the simplified loss function yields better sample quality.

## Latent Diffusion Model

<img src="pic/latent-diffusion-arch.png">

## References

* [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
* [Diffusion Model Clearly Explained!](https://medium.com/@steinsfu/diffusion-model-clearly-explained-cd331bd41166)
