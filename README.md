﻿# Machine Learning, Computer Vision and Artificial Intelligence

* Introductions
  * [Artificial Intelligence (AI), Machine Learning (ML) and Deep Learning (DL)](./introduction/machine_learning_and_deep_learning.md)
  * [Artificial Neural Network](./introduction/neural_network.md)
  * [Deep Neural Network](./introduction/deep_neural_network.md)
  * Generative Model
    * [Generative Adversarial Network-GAN](./introduction/gan.md)
    * [Encoder-Decoder, Autoencoder and U-Net](./introduction/autoencoder.md)
    * [Diffusion Models](./introduction/diffusion_model.md)
    * [Stable Diffusion](./introduction/stable_diffusion_model.md)
    * [ChatGPT](./introduction/chatgpt.md)
  * Machine Learning Performance Evaluation Metrics
    * [Evaluation for Classification](./introduction/metrics_classification.ipynb)
    * [Evaluation for Regression](./introduction/metrics_regression.md)
    * [Mean Average Precision (mAP) for Image Recognition](./object_detection/concepts/mAP.md)
  * [Machine Learning Distance Metrics](./machine_learning/distance_metrics/distance_metrics.ipynb)    
  * [Overfitting and Underfitting](./introduction/overfitting.md)
    * [Bias and Variance](./introduction/overfitting.md#bias-and-variance)
    * [Train Test Split](./introduction/overfitting.md#train-test-split)
    * [Cross Validation](./introduction/overfitting.md#cross-validation)
    * [Feature Selection and Dimensionality Reduction (ML)](./machine_learning/data_preprocessing/dimensionality_reduction.md)
    * [Normalization (DL)](./deep_learning/normalization/README.md)  
    * [Regularization](./introduction/overfitting.md#regularization)
      * [Ridge Regression and Lasso Regression](./machine_learning/regression.ipynb)
    * [Dropout for Neural Networks (DL)](./introduction/overfitting.md#dropout-for-neural-networks)
    * [Data Augmentation (DL)](./introduction/overfitting.md#data-augmentation)
  * [Gradient Descent](./introduction/gradient_descent.md)
    * [Backpropagation](./introduction/back_propagation.md)
    * [Cost Function and Loss Function](./introduction/gradient_descent.md#cost-function-and-loss-function)
      * [Information, Entropy, Cross Entropy, Categorical CE, Binary CE](./machine_learning/concepts/entropy.md)
      * [KL Divergence and JS Divergence](./machine_learning/concepts/kl_divergence.md)
      * [Focal Loss](./machine_learning/concepts/focal_loss.md)
    * [Learning Rate](./introduction/gradient_descent.md#learning-rate)
    * [Batch Gradient Descent (BGD)](./introduction/gradient_descent.md#batch-gradient-descent-bgd)
    * [Stochastic Gradient Descent (SGD)](./introduction/gradient_descent.md#stochastic-gradient-descent-sgd)
    * [MiniBatch Gradient Descent](./introduction/gradient_descent.md#minibatch-gradient-descent)
    * [Convex Function](./introduction/gradient_descent.md#convex-function)
    * [Local Minima and Saddle Point](./introduction/gradient_descent.md#local-minima-and-saddle-point)
    * [Vanishing and Exploding Gradient](./introduction/gradient_descent.md#vanishing-and-exploding-gradient)
    * [Exploding Gradient](./introduction/exploding_gradient.md)
    * [Optimization Algorithms SGD, Adam, RMSProp](./introduction/optimization_algorithms.ipynb)

* LLM
  * Introductions and Tutorials
    * [Attention](./LLM/attention.md)
      * [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
      * [The Attention Mechanism from Scratch](./deep_learning/transformer/attention.ipynb)      
    * [Token Embedding](./LLM/token_embedding.md)
      * [Word Embedding](./text/word_embedding.md)
        * [Word2Vec](./text/word_embedding.md#word2vec)
        * [What Are Word Embeddings for Text?](https://machinelearningmastery.com/what-are-word-embeddings/)
        * [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/)    
    * Transformer
      * [Vision Transformer](https://github.com/google-research/vision_transformer)
  * [HuggingFace](./LLM/HuggingFace.md)
  * LLM Inference Optimizations
    * C/C++ based
      * [llama.cpp](./LLM/llama.cpp.md)
      * [llamafile](https://github.com/Mozilla-Ocho/llamafile)
      * [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer)
    * Python based
      * [vLLM](https://github.com/vllm-project/vllm)
      * [SGLang](https://github.com/sgl-project/sglang)
      * [KTransformers](https://github.com/kvcache-ai/ktransformers)    
    * Kernel Optimization
      * [Marlin: a Mixed Auto-Regressive Linear kernel, an extremely optimized FP16xINT4 matmul kernel](https://github.com/IST-DASLab/marlin)

* VLM
  * [Vision Language Models Explained](https://huggingface.co/blog/vlms)
  * [CLIP: Contrastive Language-Image Pre-Training](https://github.com/openai/CLIP)
    * [SigLIP: Sigmoid Loss for Language Image Pre-Training](https://huggingface.co/docs/transformers/v4.48.0/model_doc/siglip)
    * [BLIP and BLIP-2: Boostrapping Lanugage-Image Pre-training](https://huggingface.co/blog/blip-2)
  * [BAAI: Aquila-VL](https://huggingface.co/BAAI/Aquila-VL-2B-llava-qwen)
  * [DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models](https://tsinghua-mars-lab.github.io/DriveVLM/)
  * [Senna: Bridging Large Vision-Language Models and End-to-End Autonomous Driving](https://github.com/hustvl/Senna)
  * [LLaVA: Large Language and Vision Assistant](https://llava-vl.github.io/)
  * [LLaVA-OneVision: Easy Visual Task Transfer](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/)

* Machine Learning
  * [IRIS classification with Scikit-learn quickstart](./machine_learning/iris_tutorial.ipynb)
    * [IRIS with Different Machine Learning Algorithms](./machine_learning/iris.ipynb)
  * [Data Loading](./machine_learning/data_loading.ipynb)
  * Data Exploration
    * [Understanding Data with Statistics: Summary, Distributions, Correlations and Skewness](./machine_learning/data_statistics.ipynb)
    * [Understanding Data with Visualizaiton: Histogram, Density, Box and Correlation Matrix Plots](./machine_learning/data_visualization.ipynb)
  * [Data Preprocessing](./machine_learning/data_preprocessing/data_preprocessing.md)
    * Data Cleaning
      * [Data Cleaning: Outlier Detection and Removal](./machine_learning/data_preprocessing/outlier.ipynb)
    * Data Transformation
      * [Scaling in Data Preprocessing](./machine_learning/data_preprocessing/scaling.ipynb)
      * [Discretization in Data Preprocessing](./machine_learning/data_preprocessing/discretization.ipynb)
      * [Categorical Data Encoding](./machine_learning/data_preprocessing/categorical_data_encoding.ipynb)
        * [ColumnTransformer and Pipeline for Mixed Data Types](./machine_learning/data_preprocessing/column_transformer.ipynb)      
    * [Data Reduction](./machine_learning/data_preprocessing/dimensionality_reduction.md)
      * Principal Component Analysis (PCA)
        * [PCA with Scikit-learn on IRIS dataset](./machine_learning/PCA/pca_scikit_learn.ipynb)
        * [PCA: feature transformation intuitive guide](./machine_learning/PCA/pca_feature_transformation.md)
        * [Statistical and Mathematical Concepts behind PCA](./machine_learning/PCA/pca_math.ipynb)
        * [Image Compression Using PCA](./machine_learning/PCA/pca_mnist_image_compression.ipynb)
      * Linear Discriminant Analysis (LDA)
        * [LDA with scikit-learn on Wine dataset](./machine_learning/LDA/LDA.ipynb)
        * [Statistical and Mathematical concepts behind LDA](./machine_learning/LDA/LDA_math.ipynb)
  * [Hyper-Parameters Tuning](./machine_learning/hyper_parameter/hyper_parameters.md)
    * [Bayesian Optimization of Hyperparameter Tuning With scikit-opitmize](./machine_learning/hyper_parameter/bayesian_optimization_scikit_optimize.ipynb)
    * [Bayesian Optimization of Hyperparameter Tuning With Hyperopt](./machine_learning/hyper_parameter/bayesian_optimization_hyperopt.ipynb)
  * Machine Learning Models
    * [Linear Regression](./machine_learning/regression.ipynb)
    * [Logistic Regression](./machine_learning/logistic_regression.ipynb)
    * [Perceptron](./machine_learning/perceptron.ipynb)
    * [Support Vector Machines](./machine_learning/svm/svm.ipynb)
    * [Decision Tree](./machine_learning/decision_tree/decision_tree.ipynb)
      * [Decision Tree with IRIS dataset](./machine_learning/decision_tree/decision_tree_iris.ipynb)
    * [Random Forest](./machine_learning/random_forest/Random_Forest_Tutorial.ipynb)
    * XGBoost
    * [K-Nearest Neighbors (KNN)](./machine_learning/knn.ipynb)
    * [Naive Bayes Classification](./machine_learning/naive_bayes/naive_bayes.ipynb)
    * [Clustering](./machine_learning/clustering/clustering.md)
      * [Centroid Model: K-means](./machine_learning/clustering/k_means.ipynb)
      * Mean Shift
      * [Connectivity Model: Hierarchical Clustering](./machine_learning/clustering/hierarchical_clustering.ipynb)
      * [Density Model: DBSCAN](./machine_learning/clustering/DBSCAN.ipynb)
    * [Association Rules](./data_mining/association_rule.ipynb)      
    * Ensemble
      * Bagging
        * [Random Forest](./machine_learning/iris_tutorial.ipynb)
      * Boosting
        * XGBoost
      * Voting
  * Machine Learning Model Explainability
    * [Machine Learning Pipeline with Explainability](./machine_learning/pic/model_explainability.jpg)
    * [SHAP: Explain Any Machine Learning Model](./machine_learning/shap.ipynb)
  * [Concepts](./glossary/README.md)
    * [Information, Entropy, Cross Entropy, Categorical CE, Binary CE](./machine_learning/concepts/entropy.md)
    * [KL Divergence and JS Divergence](./machine_learning/concepts/kl_divergence.md)
    * [Vector Norm](./machine_learning/concepts/vector_norm.ipynb)
    * [Correlation](./machine_learning/concepts/correlation.ipynb)
    * [Focal Loss](./machine_learning/concepts/focal_loss.md)
    * [Synthetic Minority Oversampling Technique (SMOTE)](./machine_learning/concepts/smote.md)
    * [Zero-Shot Learning](./machine_learning/concepts/zero_shot_learning.md)

* Deep Learning
  * [A Visual and Interactive Guide to the Basics of Neural Networks](https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/)
  * [A Visual And Interactive Look at Basic Neural Network Math](https://jalammar.github.io/feedforward-neural-networks-visual-interactive/)
  * [Batch Size, Training Steps and Epochs](./deep_learning/batch_epoch.md)
  * [Hyperparameters](./deep_learning/hyper-parameters.md)
    * [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)
  * [Activation Functions](./deep_learning/activation_function.md)
  * [Convolution](./deep_learning/convolution/README.md)
    * [1D Convolution](./deep_learning/convolution/conv_1d.md)
    * [Convolution Filters](./deep_learning/convolution/convolution_filters.ipynb)
  * [Pooling](./deep_learning/pooling.md)
  * [Normalization](./deep_learning/normalization/README.md)
  * [Residual Block and Inverted Residual Block](./deep_learning/residual_block/README.md)
  * Deep Learning Framework
    * [PyTorch vs Tensorflow](./deep_learning/framework/tensorflow_vs_pytorch.md)
    * PyTorch
      * [PyTorch Installation](./deep_learning/framework/pytorch/install.md)
      * [PyTorch Quickstart](./deep_learning/framework/pytorch/quickstart_tutorial.ipynb)
      * [PyTorch Computational Graph](./deep_learning/framework/pytorch/computational_graph.ipynb)
        * [Update the parameters of a neural netowrk](./deep_learning/framework/pytorch/update_parameters.md)
        * [Leaf Tensor in-place update](./deep_learning/framework/pytorch/leaf_tensor_in_place_update.ipynb)
      * [PyTorch Access A Layer by the Module Name](./deep_learning/framework/pytorch/layer_access.ipynb)
        * [modules() vs children()](./deep_learning/framework/pytorch/modules_vs_children.ipynb)
        * [nn.Parameter](./deep_learning/framework/pytorch/parameter.ipynb)
      * [PyTorch Extract an Intermediate Layer](./deep_learning/framework/pytorch/extract_intermediate_layer.ipynb)      
      * [nn.Module vs nn.Functional](./deep_learning/framework/pytorch/module_vs_functional.ipynb)
      * [ModuleList and ParameterList](./deep_learning/framework/pytorch/ModuleList_and_ParameterList.ipynb)      
    * Tensorflow
      * [Tensorflow model optimization for inference](./deep_learning/framework/Tensorflow/optimize_for_inference.md)
      * [logits, softmax and softmax_cross_entropy_with_logits](./deep_learning/framework/logits_softmax.ipynb)
      * [Protobuf and Flat Buffers](./deep_learning/framework/protobuf.md)
      * [tf.placeholder vs tf.Variable](./deep_learning/framework/placeholder_variable.ipynb)
      * [Graph vs GraphDef](./deep_learning/framework/Graph_and_GraphDef.md)
      * [Save and Restore Tensorflow Models](./deep_learning/framework/save_and_restore_tensorflow_models.ipynb)
      * [TFRecord to Store and Extract Data](./deep_learning/framework/TFRecord.ipynb)
      * [Convert a Global Average Pooling layer to Conv2D](./deep_learning/framework/gap_to_conv2d.ipynb) 
    * ONNX
      * [Create a toy model with LayerNormalization](./deep_learning/framework/onnx/onnx_layernorm_transformer.py)
      * [Get intermediary layer output of an ONNX model](./deep_learning/framework/onnx/onnx_output_intermediate_layer.md)
      * [NNEF and ONNX: Similarities and Differences](https://www.khronos.org/blog/nnef-and-onnx-similarities-and-differences)
  * Quantization
    * [Quantization Arithmetic](./deep_learning/quantization/quantization_arithmetic.md)
    * [PyTorch](https://pytorch.org/docs/stable/quantization.html)
      * [Selective Quantization](./deep_learning/quantization/PyTorch/selective_quantization.ipynb)
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
  * Stable Diffusion
    * [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/)
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
  * Segmentation
    * Color Segmentation
      * [Color Image Segmentation in HSV Color Space](./object_segmentation/color/hsv_color_segmentation.ipynb)
    * [Deep Adversarial Training for Multi-Organ Nuclei Segmentation in Histopathology Images](./object_segmentation/nuclei_segmentation.md)
  * Image Translation
    * [Pix2Pix - Image-to-Image Translation with Conditional Adversarial Networks](./image_translation/pix2pix/README.md)
    * [CycleGAN - Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](./image_translation/CycleGAN/README.md)

  * 3D
    * [3D Technology Map](./3D/3D_deep_learning.png)
    * Point Cloud
      * [Introduction to Point Cloud](./3D/point_cloud_processing.ipynb)
      * [Estimate Point Clouds From Depth Images](./3D/point_cloud_rgbd.ipynb)
    * [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://github.com/cshbli/PointNet)
    * PointPillars
      * [PointPillars Introduction](./3D/point_pillars.md)
    * [Depth Estimation from Stereo Images](./3D/depth_estimation/stero.md)
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

* Image Processing
  * [Color Temperature Kelvin to RGB](./image_processing/color_science/kelvin_to_rgb.ipynb)
  * [HSV Color Space](./image_processing/color_science/hsv.md)
  * Color Correction and Calibration
    * [Histogram Matching](./image_processing/color_correction/histogram_matching.ipynb)
    * [Color Correction with Color Card](./image_processing/color_correction/color_correction_card.ipynb)
  * Focus
    * [Image Focus Checking](./image_processing/focus/image_blurry.ipynb)
  * Utilities
    * [Split images](./image_processing/utils/image_split.ipynb)
  * [Color Science with Python](https://github.com/colour-science/colour)

* Text and NLP
  * [TF-IDF: Term Frequency, Inverse Document Frequency](./text/tf_idf.ipynb)  
  * [Doc2Vec](./text/Doc2Vec.md)
  * [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](https://jalammar.github.io/illustrated-bert/)
  * [How GPT-3 Works - Visualizations and Animations](https://jalammar.github.io/how-gpt3-works-visualizations-animations/)
  * [The Illustrated GPT-2 (Visualizing Transformer Language Models)](https://jalammar.github.io/illustrated-gpt2/)
     
* Data Structure and Algorithms
  * [Heap and Heap Sort in Python](./data_structure/heap_sort.ipynb)

* Linear Algebra
  * [Matrix Factorization and SVD](./linear_algebra/matrix_factorization.ipynb)

* Data Mining
  * [Association Rules](./data_mining/association_rule.ipynb)

* Kalman Filter
  * [Kalman Filter Tutorial](https://www.kalmanfilter.net/default.aspx)
  * [Kalman Filter Implementation in Python](./kalman_filter/kalman_filter_30.ipynb)

## [Miscellaneous](miscellaneous.md)

## Resources
* [Netron](https://netron.app/)
* [Hugging Face](https://huggingface.co/)
* [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)
* [Learn OpenCV : C++ and Python Examples](https://github.com/spmallick/learnopencv)
* [Machine Learning Tutorial](https://www.javatpoint.com/machine-learning)

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
* [Dive into Deep Learning](https://d2l.ai/index.html)
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
