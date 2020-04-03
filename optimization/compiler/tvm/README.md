# TVM
   * [Introduction](#introduction)
   * [Installation](https://tvm.apache.org/docs/install/from_source.html)
   * [Quantization](#quantization)
   * [Tutorials](#tutorials)
   
## Introduction
Apache(incubating) TVM is an open deep learning compiler stack for CPUs, GPUs, and specialized accelerators. It aims to close the gap between the productivity-focused deep learning frameworks, and the performance- or efficiency-oriented hardware backends. TVM provides the following main features:

  * Compilation of deep learning models in Keras, MXNet, PyTorch, Tensorflow, CoreML, DarkNet into minimum deployable modules on diverse hardware backends.
  
  * Infrastructure to automatic generate and optimize tensor operators on more backend with better performance.
  
TVM began as a research project at the SAMPL group of Paul G. Allen School of Computer Science & Engineering, University of Washington. The project is now an effort undergoing incubation at The Apache Software Foundation (ASF), driven by an open source community involving multiple industry and academic institutions under the Apache way.

TVM provides two level optimizations show in the following figure. Computational graph optimization to perform tasks such as high-level operator fusion, layout transformation, and memory management. Then a tensor operator optimization and code generation layer that optimizes tensor operators. More details can be found at the [techreport](https://arxiv.org/pdf/1802.04799.pdf).

![TVM Stack](tvm-stack.png)

## Quantization

![TVM Quantization](tvm_quantization.png)

### Frameworks to Relay

As shown in the above figure, there are two different parallel efforts ongoing

  * <b>Automatic Integer Quantization:</b> It takes a FP32 framework graph and automatically converts it to Int8 within Relay.
  
  * <b>Accepting Pre-quantized Integer models:</b> This approach accepts a pre-quantized model, introduces a Relay dialect called QNN and generates an Int8 Relay graph.

### Relay Optimizations

  * <b>Target-independent Relay passes</b>: TVM community is continuously adding these passes. Examples are fuse constant, common subexpression elimination etc.
  
  * <b>Target-dependent Relay passes</b>: These passes transform the Relay graph to optimize it for the target. An example is Legalize or AlterOpLayout transform, where depending on the target, we change the layouts of convolution/dense layer. TVM community is working on improving on both infrastructure to enable such transformation, and adding target-specific layout transformations. 

### Relay to Hardware

Once we have an optimized Relay graph, we need to write optimized schedules. Like FP32, we have to focus our efforts only on expensive ops like conv2d, dense etc. There are scattered efforts and TVM community is working on unifying them. Some of the developers that have worked on different backends (not necessarily Int8)

## Tutorials
  
  * Quick Start Tutorial for Compiling Deep Learning Models
      * [Jupyter notebook](relay_quick_start.ipynb)
      * [Python source code](relay_quick_start.py)

## References

* [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://arxiv.org/pdf/1802.04799.pdf) by Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Meghan Cowan, Haichen Shen, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy.
