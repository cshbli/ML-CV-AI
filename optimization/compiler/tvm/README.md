# TVM
   * [Introduction](#introduction)
   * [Installation](https://tvm.apache.org/docs/install/from_source.html)
   * [Tutorials](#tutorials)
   
## Introduction
Apache(incubating) TVM is an open deep learning compiler stack for CPUs, GPUs, and specialized accelerators. It aims to close the gap between the productivity-focused deep learning frameworks, and the performance- or efficiency-oriented hardware backends. TVM provides the following main features:

  * Compilation of deep learning models in Keras, MXNet, PyTorch, Tensorflow, CoreML, DarkNet into minimum deployable modules on diverse hardware backends.
  
  * Infrastructure to automatic generate and optimize tensor operators on more backend with better performance.
  
TVM began as a research project at the SAMPL group of Paul G. Allen School of Computer Science & Engineering, University of Washington. The project is now an effort undergoing incubation at The Apache Software Foundation (ASF), driven by an open source community involving multiple industry and academic institutions under the Apache way.

TVM provides two level optimizations show in the following figure. Computational graph optimization to perform tasks such as high-level operator fusion, layout transformation, and memory management. Then a tensor operator optimization and code generation layer that optimizes tensor operators. More details can be found at the [techreport](https://arxiv.org/pdf/1802.04799.pdf).

![TVM Stack](tvm-stack.png)

## Tutorials
  
  * [Quick Start Tutorial for Compiling Deep Learning Models](relay_quick_start.ipynb)([full python source code](relay_quick_start.py))

## References

* [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://arxiv.org/pdf/1802.04799.pdf) by Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Meghan Cowan, Haichen Shen, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy.
