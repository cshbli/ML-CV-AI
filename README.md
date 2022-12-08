# Machine Learning, Computer Vision and Data Science
* Introductions
  * [Artificial Intelligence, Machine Learning and Deep Learning](./introduction/machine_learning_and_deep_learning.md)
  * [Artificial Neural Network](./introduction/neural_network.md)
  * [Deep Neural Network](./introduction/deep_neural_network.md)
  * [Generative Adversarial Networks-GANs](./introduction/gan.md)
  * [Machine Learning Performance Evaluation](./introduction/metrics.md)
  * [Overfitting](./introduction/overfitting.md)
* Machine Learning
  * [Machine Learning](./glossary/README.md)
  * [Deep Learning](./glossary/README.md)  
* Deep Learning
  * [Covolution](./deep_learning/convolution/README.md)
  * [Normalization](./deep_learning/normalization/README.md)
  * [Residual Block and Inverted Residual Block](./deep_learning/residual_block/README.md)
<details>  
  <summary>Computer Vision</summary>

  * [Overview](./computer_vision/README.md)
  * [Object Detection](./object_detection/README.md)
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
    * [YOLOv5](./object_detection/YOLOv5/README.md)
      * [Object Detection using YOLOv5](./object_detection/YOLOv5/object_detection_using_yolov5.ipynb)
    * [SSD - Single Shot Detector](./object_detection/SSD/README.md)
    * [RetinaNet](./object_detection/RetinaNet/README.md)
      * [Face mask detection with RetinaNet example](./object_detection/RetinaNet/face_mask_detector/FaceMaskDetector.ipynb)
     
  * Classification  
    * [MNIST classification with Tensorflow quickstart](./classification/MNIST_classification_with_tensorflow_quickstart.ipynb) (from [Tensorflow Tutorial](https://www.tensorflow.org/tutorials/quickstart/beginner) )
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
      
</details>


* Deep Learning Optimization
  * Efficient Training
  * Efficient Neural Network
  * [Model Optimization](./optimization/README.md)
    * [Quantization](./optimization/quantization/README.md) 
    * Pruning
    * Compression
      * [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/pdf/1510.00149.pdf)
   * Matrix Operation Optimization
   * [Deep Learning Compiler](./optimization/compiler/README.md)
* Text
  * [Word2Vec](./text/Word2Vec.md)
  * [Doc2Vec](./text/Doc2Vec.md)
* Generative Adversarial Networks
  * Deep Convolutional Generative Adversarial Networks (DCGANs)    
* Neural Network Exchange
  * [NNEF and ONNX: Similarities and Differences](https://www.khronos.org/blog/nnef-and-onnx-similarities-and-differences)
* Deep Learning Framework
  * Tensorflow
    * [logits, softmax and softmax_cross_entropy_with_logits](./framework/logits_softmax.ipynb)
    * [Protobuf and Flat Buffers](./framework/protobuf.md)
    * [tf.placeholder vs tf.Variable](./framework/placeholder_variable.ipynb)
    * [Graph vs GraphDef](./framework/Graph_and_GraphDef.md)
    * [Save and Restore Tensorflow Models](./framework/save_and_restore_tensorflow_models.ipynb)
    * [TFRecord to Store and Extract Data](./framework/TFRecord.ipynb)
    * [Convert a Global Average Pooling layer to Conv2D](./framework/gap_to_conv2d.ipynb)
  * PyTorch
    * [PyTorch Installation](./framework/pytorch/install.md)
    * [PyTorch Quickstart](./framework/pytorch/quickstart_tutorial.ipynb)
     
## Small Techniques and Utilities
* [Visualize Tensorflow graph with Tensorboard](./tools/tensorboard.md)
* [Notes](./tools/notes.md)

## Dataset
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
* [Deep Learning course: lecture slides and lab notebooks](https://github.com/m2dsupsdlclass/lectures-labs) by University Paris Saclay
