# Model Optimization
  * [Introduction](#introduction)
  * [Quantization](./quantization/README.md) 
  * Pruning
  * Compression
     * [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/pdf/1510.00149.pdf)
  * [Deep Learning Compiler](./compiler/README.md)

## Introduction

Edge devices often have limited memory or computational power. Various optimizations can be applied to models so that they can be run within these constraints. In addition, some optimizations allow the use of specialized hardware for accelerated inference.

There are several main ways model optimization can help with application development.

### Size reduction

Some forms of optimization can be used to reduce the size of a model. Smaller models have the following benefits:

  * Smaller storage size: Smaller models occupy less storage space on your users' devices. For example, an Android app using a smaller model will take up less storage space on a user's mobile device.
  
  * Smaller download size: Smaller models require less time and bandwidth to download to users' devices.
  
  * Less memory usage: Smaller models use less RAM when they are run, which frees up memory for other parts of your application to use, and can translate to better performance and stability.
  
Quantization can reduce the size of a model in all of these cases, potentially at the expense of some accuracy. Pruning can reduce the size of a model for download by making it more easily compressible.

### Latency reduction

Latency is the amount of time it takes to run a single inference with a given model. Some forms of optimization can reduce the amount of computation required to run inference using a model, resulting in lower latency. Latency can also have an impact on power consumption.

Currently, quantization can be used to reduce latency by simplifying the calculations that occur during inference, potentially at the expense of some accuracy.

### Accelerator compatibility

Some hardware accelerators, such as the Edge TPU, can run inference extremely fast with models that have been correctly optimized.

Generally, these types of devices require models to be quantized in a specific way. See each hardware accelerators documentation to learn more about their requirements.

### Trade-offs

Optimizations can potentially result in changes in model accuracy, which must be considered during the application development process.

The accuracy changes depend on the individual model being optimized, and are difficult to predict ahead of time. Generally, models that are optimized for size or latency will lose a small amount of accuracy. Depending on your application, this may or may not impact your users' experience. In rare cases, certain models may gain some accuracy as a result of the optimization process.
