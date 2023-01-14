# Machine Learning, Computer Vision and Data Science
* Introductions
  * [Artificial Intelligence, Machine Learning (ML) and Deep Learning (DL)](./introduction/machine_learning_and_deep_learning.md)
  * [Artificial Neural Network](./introduction/neural_network.md)
  * [Deep Neural Network](./introduction/deep_neural_network.md)
  * [Generative Adversarial Networks-GANs](./introduction/gan.md)
  * [Machine Learning Performance Evaluation Metrics](./introduction/metrics.md)
    * [Mean Average Precision (mAP) for Image Recognition](./object_detection/concepts/mAP.md)
  * [Overfitting and Underfitting](./introduction/overfitting.md)
    * [Bias and Variance](./introduction/overfitting.md#bias-and-variance)
    * [Train Test Split](./introduction/overfitting.md#train-test-split)
    * [Cross Validation](./introduction/overfitting.md#cross-validation)
    * [Feature Selection (ML)](./machine_learning/feature_selection.ipynb)
    * [Dimensionality Reduction (ML)](./machine_learning/dimensionality_reduction.ipynb)
    * [Normalization (DL)](./deep_learning/normalization/README.md)  
    * [Regularization (DL)](./introduction/overfitting.md#regularization)    
    * [Dropout for Neural Networks (DL)](./introduction/overfitting.md#dropout-for-neural-networks)
    * [Data Augmentation (DL)](./introduction/overfitting.md#data-augmentation)
  * [Gradient Descent](./introduction/gradient_descent.md)
    * [Cost Function and Loss Function](./introduction/gradient_descent.md#cost-function-and-loss-function)
      * [Information, Entropy, Cross Entropy, Binary Cross Entropy, KL Divergence](./machine_learning/concepts/entropy.md)
    * [Learning Rate](./introduction/gradient_descent.md#learning-rate)
    * [Batch Gradient Descent (BGD)](./introduction/gradient_descent.md#batch-gradient-descent-bgd)
    * [Stochastic Gradient Descent (SGD)](./introduction/gradient_descent.md#stochastic-gradient-descent-sgd)
    * [MiniBatch Gradient Descent](./introduction/gradient_descent.md#minibatch-gradient-descent)
    * [Convex Function](./introduction/gradient_descent.md#convex-function)
    * [Local Minima and Saddle Point](./introduction/gradient_descent.md#local-minima-and-saddle-point)
    * [Vanishing and Exploding Gradient](./introduction/gradient_descent.md#vanishing-and-exploding-gradient)
    * [Exploding Gradient](./introduction/exploding_gradient.md)

* Machine Learning  
  * [IRIS classification with Scikit-learn quickstart](./machine_learning/iris_tutorial.ipynb)
    * [IRIS with Different Machine Learning Algorithms](./machine_learning/iris.ipynb)
  * [Data Loading](./machine_learning/data_loading.ipynb)
  * [Understanding Data with Statistics: Summary, Distributions, Correlations and Skewness](./machine_learning/data_statistics.ipynb)
  * [Understanding Data with Visualizaiton: Histogram, Density, Box and Correlation Matrix Plots](./machine_learning/data_visualization.ipynb)
  * [Data Preprocessing: Scaling and Standardization](./machine_learning/data_preprocessing.ipynb)
    * [Data Preprecessing for Mixed Feature Data Types](./machine_learning/column_transformer.ipynb)
    * [Data Cleaning: Outlier Detection and Removal](./machine_learning/outlier.ipynb)
  * Machine Learning Models
    * Linear Regression
    * [Logistic Regression](./machine_learning/logistic_regression.ipynb)
    * [Perceptron](./machine_learning/perceptron.ipynb)
    * Support Vector Machines
    * [Decision Tree](./machine_learning/decision_tree.ipynb)
    * [Random Forest](./machine_learning/random_forest/Random_Forest_Tutorial.ipynb)
    * XGBoost
    * K-Nearest Neighbors (KNN)
    * Naive Bayes Classification
    * Clustering
      * K-means
      * Mean Shift
      * Hierarchical
    * Ensemble
      * Bagging
        * [Random Forest](./machine_learning/iris_tutorial.ipynb)
      * Boosting
        * XGBoost
      * Voting
  * [Concepts](./glossary/README.md)
    * [Information, Entropy, Cross Entropy, Binary Cross Entropy, KL Divergence](./machine_learning/concepts/entropy.md)    
    * [Synthetic Minority Oversampling Technique (SMOTE)](./machine_learning/concepts/smote.md)

* Deep Learning
  * [Activation Functions](./deep_learning/activation_function.md)
  * [Covolution](./deep_learning/convolution/README.md)
  * [Normalization](./deep_learning/normalization/README.md)
  * [Residual Block and Inverted Residual Block](./deep_learning/residual_block/README.md)
  * Deep Learning Framework
    * PyTorch
        * [PyTorch Installation](./deep_learning/framework/pytorch/install.md)
        * [PyTorch Quickstart](./deep_learning/framework/pytorch/quickstart_tutorial.ipynb)
    * Tensorflow
        * [logits, softmax and softmax_cross_entropy_with_logits](./deep_learning/framework/logits_softmax.ipynb)
        * [Protobuf and Flat Buffers](./deep_learning/framework/protobuf.md)
        * [tf.placeholder vs tf.Variable](./deep_learning/framework/placeholder_variable.ipynb)
        * [Graph vs GraphDef](./deep_learning/framework/Graph_and_GraphDef.md)
        * [Save and Restore Tensorflow Models](./deep_learning/framework/save_and_restore_tensorflow_models.ipynb)
        * [TFRecord to Store and Extract Data](./deep_learning/framework/TFRecord.ipynb)
        * [Convert a Global Average Pooling layer to Conv2D](./deep_learning/framework/gap_to_conv2d.ipynb) 
    * ONNX
      * [Create a toy model with LayerNormalization](./deep_learning/framework/onnx/onnx_layernorm_transformer.py)
      * [NNEF and ONNX: Similarities and Differences](https://www.khronos.org/blog/nnef-and-onnx-similarities-and-differences)
  * Quantization
    * [Quantization Arithmetic](./deep_learning/quantization/quantization_arithmetic.md)
    * [PyTorch](https://pytorch.org/docs/stable/quantization.html)
      * [Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
      * [Practical Quantization in PyTorch](https://pytorch.org/blog/quantization-in-practice/)        
      * [PyTorch Numeric Suite Tutorial](https://pytorch.org/tutorials/prototype/numeric_suite_tutorial.html)
      * [Torch Quantization Design Proposal](https://github.com/pytorch/pytorch/wiki/torch_quantization_design_proposal)
      * Eager Mode Quantization
        * [MobileNetV2 QAT on CIFAR-10](./deep_learning/quantization/PyTorch/mobilenetv2_cifar10.ipynb)        
        * [Resnet18 QAT on CIFAR-10](./deep_learning/quantization/PyTorch/qat_resnet18_cifar10.ipynb)
        * [Static quantization with Eager Mode in PyTorch](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)          
      * FX Graph Graph Mode Quantization
    * [ONNX](https://onnxruntime.ai/docs/performance/quantization.html)
      * [ONNX Runtime Qunatization Example MobilenetV2 with QDQ Debugging](./deep_learning/quantization/ONNX/quantization_example.md)
      * [Mobilenet v2 Quantization with ONNX Runtime on CPU](./deep_learning/quantization/ONNX/mobilenet.ipynb)
    * [TensorRT](https://github.com/pytorch/TensorRT/tree/main/notebooks)
      * [Docker Environment Setup](./deep_learning/quantization/TensorRT/docker.md)
      * [VGG16 QAT on CIFAR-10](./deep_learning/quantization/TensorRT/vgg-qat.ipynb)
      * [YOLOv5 Quant Example](https://github.com/maggiez0138/yolov5_quant_sample)
  * Transformer
    * [The Attention Mechanism from Scratch](./deep_learning/transformer/attention.ipynb)
    * [The illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
    * [Vision Transformer](https://github.com/google-research/vision_transformer)
  * [RepVGG: Making VGG-style ConvNets Great Again](./deep_learning/RepVgg/RepVgg.ipynb)
  * [Compiler](./deep_learning/compiler/README.md)
     * [GLOW](https://github.com/pytorch/glow)

* Computer Vision  
  * [Overview](./computer_vision/README.md)
  * [Object Detection](./object_detection/README.md)
    * Concepts
      * [Anchor Boxes](./object_detection/concepts/README.md#anchor-boxes)
      * [Feature Pyramid Network (FPN)](./object_detection/concepts/README.md#feature-pyramid-network-fpn)   
      * [Focal Loss](./object_detection/concepts/README.md#focal-loss)   
      * [Intersection over Union (IoU)](./object_detection/concepts/README.md#intersection-over-union-iou)
        * [IoU sample notebook](./object_detection/concepts/IoU.ipynb)
      * [Mean Average Precision (mAP)](./object_detection/concepts/mAP.md)
      * [Non Maximum Suppression (NMS)](./object_detection/concepts/README.md#non-maximum-suppression-nms)
        * [NMS in PyTorch](./object_detection/concepts/nms_pytorch.ipynb)
      * [Region Proposal Network (RPN)](./object_detection/concepts/README.md#region-proposal-network-rpn)
    * [YOLOv7](./object_detection/YOLOv7/README.md)
      * [YOLOv7 QAT](./object_detection/YOLOv7/qat.md)
    * [YOLOv5](./object_detection/YOLOv5/README.md)
      * [Object Detection using YOLOv5 and OpenCV DNN](./object_detection/YOLOv5/object_detection_using_yolov5.ipynb)
      * [YOLOv5 Post Training Quantization with ONNX](https://github.com/cshbli/yolov5_qat/blob/ptq_onnx/ptq_onnx.ipynb)
      * [YOLOv5 QAT](https://github.com/cshbli/yolov5_qat)
    * [SSD - Single Shot Detector](./object_detection/SSD/README.md)
    * [RetinaNet](./object_detection/RetinaNet/README.md)
      * [Face mask detection with RetinaNet example](./object_detection/RetinaNet/face_mask_detector/FaceMaskDetector.ipynb)
     
  * Classification  
    * [Transfer Learning with PyTorch tutorial](./classification/transfer_learning_tutorial.ipynb) 
  * Object Segmentation
    * [Deep Adversarial Training for Multi-Organ Nuclei Segmentation in Histopathology Images](./object_segmentation/nuclei_segmentation.md)
  * Image Translation
    * [Pix2Pix - Image-to-Image Translation with Conditional Adversarial Networks](./image_translation/pix2pix/README.md)
    * [CycleGAN - Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](./image_translation/CycleGAN/README.md)
  * [Deepface](https://github.com/serengil/deepface)

  * Data Annotation
    * [Computer Vision Annotation Tool (CVAT)](https://github.com/opencv/cvat)
    * [LabelImg](https://github.com/heartexlabs/labelImg)
    * Autonomous driving  
      * 2D Bounding Box    
      * Lane Line    
      * Semantic Segmentation    
      * Video Tracking Annotation
      * 3D Point
      * 3D Object Recognition (3D Cube)
      * 3D Segmentation
      * Sensor Fusion: Cuboids/Segmentation/Tracking   
* Text
  * [Word2Vec](./text/Word2Vec.md)
  * [Doc2Vec](./text/Doc2Vec.md)  
     
## Miscellaneous
* Python
  * [Advanced Python Functions](./tools/advanced_python_functions.ipynb)
  * [Replacing Loops with Vectorization](./tools/python_vectorization.ipynb)
* [Docker](./tools/docker.md)
  * [Visual Studio Code remote connect to a docker container](./tools/docker.md#visual-studio-code-remote-connect-to-a-docker-container)
* [Git](./tools/git.md)
* [Markdown and LaTeX introduction](https://ashki23.github.io/markdown-latex.html)
  * [PDF print out](./tools/Markdown%20and%20LaTeX%20introduction.pdf)
* [Visualize Tensorflow graph with Tensorboard](./tools/tensorboard.md)
* [PyQt5 and designer Installation](./tools/PyQt5.md)
* [Notes](./tools/notes.md)

## Resources
* [Netron](https://netron.app/)
* [Hugging Face](https://huggingface.co/)
* [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)
* [Learn OpenCV : C++ and Python Examples](https://github.com/spmallick/learnopencv)

## Dataset
* [Kaggle Datasets](https://www.kaggle.com/datasets)
* [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
* [Registry of Open Data on AWS](https://registry.opendata.aws/)
* [Google's Dataset Search Engine](https://datasetsearch.research.google.com/)
* [Microsoft Research Open Data](https://msropendata.com/)
* [Awesome Public Datasets](https://github.com/awesomedata/awesome-public-datasets)
* [scikit-learn dataset](https://scikit-learn.org/stable/datasets.html)
* Computer Vision
  * [VisualData Discovery](https://visualdata.io/discovery)
  * [COCO Common Objects in Context](https://cocodataset.org/#home)
  * [ImageNet](https://www.image-net.org/)
  * [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
  * [YouTube-Objects](https://data.vision.ee.ethz.ch/cvl/youtube-objects/)
  * Face
    * [IMDB-WIKI – 500k+ face images with age and gender labels](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
    * [WIDER FACE: A Face Detection Benchmark](http://shuoyang1213.me/WIDERFACE/)
    * [FDDB: Face Detection Data Set and Benchmark](http://vis-www.cs.umass.edu/fddb/)
  * Super-Resolution  
    * [DIV2K dataset: DIVerse 2K resolution high quality images for super-resolution](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
  * Object Segmentation
    * [Cityscapes](https://www.cityscapes-dataset.com/)
    * [KITTI](https://www.cvlibs.net/datasets/kitti/)
    * [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
    * [A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology](https://monuseg.grand-challenge.org/Data/)
  * [Roboflow public datasets](https://public.roboflow.com/)

## Online books
* [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) by Michael Nielsen
* [Deep Learning](http://www.deeplearningbook.org/) by Ian Goodfellow, Yoshua Bengio and Aron Courville
* [Computer Vision: Algorithms and Applications, 1st ed.](http://szeliski.org/Book/) by Richard Szeliski
* [Digital Image Processing](https://sisu.ut.ee/imageprocessing/documents) by University of Tartu

## Online Courses
* [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/) by Google
* [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/2020/index.html) by Stanford  
  * [Programming Assignments and Lectures for Stanford's CS 231: Convolutional Neural Networks for Visual Recognition](https://github.com/khanhnamle1994/computer-vision) by Standford
* [Deep Learning Nanodegree Foundation Program](https://github.com/udacity/deep-learning) by Udacity
* [Deep Learning course: lecture slides and lab notebooks](https://github.com/m2dsupsdlclass/lectures-labs) by University Paris Saclay
