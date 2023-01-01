
# Deep Learning

  * [Activation Functions](./activation_function.md)
  * [Covolution](./convolution/README.md)
  * [Normalization](./normalization/README.md)
  * [Residual Block and Inverted Residual Block](./residual_block/README.md)
  * Deep Learning Framework
    * PyTorch
        * [PyTorch Installation](./framework/pytorch/install.md)
        * [PyTorch Quickstart](./framework/pytorch/quickstart_tutorial.ipynb)
    * Tensorflow
        * [logits, softmax and softmax_cross_entropy_with_logits](./framework/logits_softmax.ipynb)
        * [Protobuf and Flat Buffers](./framework/protobuf.md)
        * [tf.placeholder vs tf.Variable](./framework/placeholder_variable.ipynb)
        * [Graph vs GraphDef](./framework/Graph_and_GraphDef.md)
        * [Save and Restore Tensorflow Models](./framework/save_and_restore_tensorflow_models.ipynb)
        * [TFRecord to Store and Extract Data](./framework/TFRecord.ipynb)
        * [Convert a Global Average Pooling layer to Conv2D](./framework/gap_to_conv2d.ipynb) 
    * ONNX
      * [Create a toy model with LayerNormalization](./framework/onnx/onnx_layernorm_transformer.py)
      * [NNEF and ONNX: Similarities and Differences](https://www.khronos.org/blog/nnef-and-onnx-similarities-and-differences)
  * Quantization
    * [Quantization Arithmetic](./quantization/quantization_arithmetic.md)
    * [PyTorch](https://pytorch.org/docs/stable/quantization.html)
      * [Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
      * [Practical Quantization in PyTorch](https://pytorch.org/blog/quantization-in-practice/)        
      * [PyTorch Numeric Suite Tutorial](https://pytorch.org/tutorials/prototype/numeric_suite_tutorial.html)
      * [Torch Quantization Design Proposal](https://github.com/pytorch/pytorch/wiki/torch_quantization_design_proposal)
      * Eager Mode Quantization
        * [MobileNetV2 QAT on CIFAR-10](./quantization/PyTorch/mobilenetv2_cifar10.ipynb)        
        * [Resnet18 QAT on CIFAR-10](./quantization/PyTorch/qat_resnet18_cifar10.ipynb)
        * [Static quantization with Eager Mode in PyTorch](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)          
      * FX Graph Graph Mode Quantization
    * [ONNX](https://onnxruntime.ai/docs/performance/quantization.html)
      * [ONNX Runtime Qunatization Example MobilenetV2 with QDQ Debugging](./quantization/ONNX/quantization_example.md)
      * [Mobilenet v2 Quantization with ONNX Runtime on CPU](./quantization/ONNX/mobilenet.ipynb)
  * Transformer
    * [Vision Transformer](https://github.com/google-research/vision_transformer)
  * [Compiler](./compiler/README.md)
     * [GLOW](https://github.com/pytorch/glow)
