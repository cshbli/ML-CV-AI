# Machine Learning, Computer Vision and Data Science Introductions 
* Classification
  * Images  
    * [MNIST classification with Tensorflow quickstart](./classification/MNIST_classification_with_tensorflow_quickstart.ipynb) (from [Tensorflow Tutorial](https://www.tensorflow.org/tutorials/quickstart/beginner) )
* Object Detection
  * [SSD - Single Shot Detector](./object_detection/SSD/README.md)
  * Focal Loss
    Methods like SSD or YOLO suffer from an extreme class imbalance: The detectors evaluate roughly between ten to hundred thousand candidate locations and of course most of these boxes do not contain an object. Even if the detector easily classifies these large number of boxes as negatives/background, there is still a problem.
    
    * Cross entropy loss function      
      This is the cross entropy loss function 
      
      where i is the index of the class, y_i the label (1 if the object belongs to class i, 0 otherwise), and p_i is the predicted probability that the object belongs to class i.
    
* [Quantization](./quantization/README.md) 
* Neural Network Exchange
  * [NNEF and ONNX: Similarities and Differences](https://www.khronos.org/blog/nnef-and-onnx-similarities-and-differences)
     
## References

### Documentation

### Sites

### Conferences

### Online Courses
