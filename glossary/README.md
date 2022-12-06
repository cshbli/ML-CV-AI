# Concept
 * Machine Learning
   * [bag of words](./README.md#bag-of-words)
   * [clustering](./README.md#clustering)
   * [Confusion Matrix](./README.md#confusion-matrix)
   * [convex function](./README.md#convex-function)
   * [convex set](./README.md#convex-set)   
   * [decision tree](./README.md#decision-tree)
   * [hashing](./README.md#hashing)
 * Deep Learning
   * [convolutional layer](./README.md#convolutional-layer)
   * [convolutional operation](./README.md#convolutional-operation)
 * Computer Vision
   * [Anchor Boxes](./README.md#anchor-boxes)
   * [Feature Pyramid Network (FPN)](./README.md#feature-pyramid-network-fpn)   
   * [Focal Loss](./README.md#focal-loss)   
   * [Intersection over Union (IoU)](./README.md#intersection-over-union-iou)
     * [IoU sample notebook](./IoU.ipynb)
   * [Non Maximum Suppression (NMS)](./README.md#non-maximum-suppression-nms)
     * [NMS in PyTorch](./nms_pytorch.ipynb)
   * [Region Proposal Network (RPN)](./README.md#region-proposal-network-rpn)   
 
## Anchor Boxes
  
  Anchor boxes were first introduced in Faster RCNN paper and later became a common element in all the following papers like yolov2, ssd and RetinaNet. Previously selective search and edge boxes used to generate region proposals of various sizes and shapes depending on the objects in the image, with standard convolutions it is highly impossible to generate region proposals of varied shapes, so anchor boxes comes to our rescue.
     <p align="center">
        <img src="pic/anchor_boxes.png" width="800px" title="Anchor boxes mapping to Images from feature map">
     </p>
     

## bag of words
A representation of the words in a phrase or passage, irrespective of order. For example, bag of words represents the following three phrases identically:

  * the dog jumps
  * jumps the dog
  * dog jumps the
  
Each word is mapped to an index in a sparse vector, where the vector has an index for every word in the vocabulary. For example, the phrase the dog jumps is mapped into a feature vector with non-zero values at the three indices corresponding to the words the, dog, and jumps. The non-zero value can be any of the following:

  * A 1 to indicate the presence of a word.
  * A count of the number of times a word appears in the bag. For example, if the phrase were <i>the maroon dog is a dog with maroon fur</i>, then both maroon and dog would be represented as 2, while the other words would be represented as 1.
  * Some other value, such as the logarithm of the count of the number of times a word appears in the bag. 
 
## clustering
Grouping related examples, particularly during unsupervised learning. Once all the examples are grouped, a human can optionally supply meaning to each cluster.

Many clustering algorithms exist. For example, the k-means algorithm clusters examples based on their proximity to a centroid, as in the following diagram:

<img src="kmeans_example_1.svg" width="400px">

A human researcher could then review the clusters and, for example, label cluster 1 as "dwarf trees" and cluster 2 as "full-size trees."

As another example, consider a clustering algorithm based on an example's distance from a center point, illustrated as follows:

<img src="RingCluster_example.svg" width="400px">

## Confusion Matrix
An NxN table that summarizes how successful a classification model's predictions were; that is, the correlation between the label and the model's classification. One axis of a confusion matrix is the label that the model predicted, and the other axis is the actual label. N represents the number of classes. In a binary classification problem, N=2. For example, here is a sample confusion matrix for a binary classification problem:

|  | Tumor(predicted) | Non-Tumor(predicted|
| --- | ---| --- | 
| Tumor (actual) |	18 |	1 |
|Non-Tumor (actual)	| 6	| 452 |

The preceding confusion matrix shows that of the 19 samples that actually had tumors, the model correctly classified 18 as having tumors (18 <b>true positives</b>), and incorrectly classified 1 as not having a tumor (1 <b>false negative</b>). Similarly, of 458 samples that actually did not have tumors, 452 were correctly classified (452 <b>true negatives</b>) and 6 were incorrectly classified (6 <b>false positives</b>).

The confusion matrix for a multi-class classification problem can help you determine mistake patterns. For example, a confusion matrix could reveal that a model trained to recognize handwritten digits tends to mistakenly predict 9 instead of 4, or 1 instead of 7.

Confusion matrices contain sufficient information to calculate a variety of performance metrics, including <b>precision</b> and <b>recall</b>.

## convex function
A function in which the region above the graph of the function is a convex set. The prototypical convex function is shaped something like the letter U. For example, the following are all convex functions:

<img src="convex_functions.png" title="Convex Functions">

A typical convex function is shaped like the letter 'U'.

By contrast, the following function is not convex. Notice how the region above the graph is not a convex set:

<img src="nonconvex_function.svg" title="Non-Convex Function">

A strictly convex function has exactly one local minimum point, which is also the global minimum point. The classic U-shaped functions are strictly convex functions. However, some convex functions (for example, straight lines) are not U-shaped.

A lot of the common loss functions, including the following, are convex functions:

 * L2 loss
 * Log Loss
 * L1 regularization
 * L2 regularization

Many variations of gradient descent are guaranteed to find a point close to the minimum of a strictly convex function. Similarly, many variations of stochastic gradient descent have a high probability (though, not a guarantee) of finding a point close to the minimum of a strictly convex function.

The sum of two convex functions (for example, L2 loss + L1 regularization) is a convex function.

Deep models are never convex functions. Remarkably, algorithms designed for convex optimization tend to find reasonably good solutions on deep networks anyway, even though those solutions are not guaranteed to be a global minimum.

## convex set
A subset of Euclidean space such that a line drawn between any two points in the subset remains completely within the subset. For instance, the following two shapes are convex sets:

<img src="convex_set.png" title="A rectangle and a semi-ellipse are both convex sets.">

By contrast, the following two shapes are not convex sets:

<img src="nonconvex_set.png" title="A pie-chart with a missing slice and a firework are both nonconvex sets.">

## convolutional layer
A layer of a deep neural network in which a convolutional filter passes along an input matrix. For example, consider the following 3x3 convolutional filter:

<img src="ConvolutionalFilter_example.svg" title="3x3 convolutional filter">

The following animation shows a convolutional layer consisting of 9 convolutional operations involving the 5x5 input matrix. Notice that each convolutional operation works on a different 3x3 slice of the input matrix. The resulting 3x3 matrix (on the right) consists of the results of the 9 convolutional operations:

<img src="AnimatedConvolution.gif">

## convolutional operation
The following two-step mathematical operation:

  * Element-wise multiplication of the convolutional filter and a slice of an input matrix. (The slice of the input matrix has the same rank and size as the convolutional filter.)
  * Summation of all the values in the resulting product matrix.
  
For example, consider the following 5x5 input matrix:

<img src="ConvolutionalLayerInputMatrix.svg">

Now imagine the following 2x2 convolutional filter:

<img src="ConvolutionalLayerFilter.svg">

Each convolutional operation involves a single 2x2 slice of the input matrix. For instance, suppose we use the 2x2 slice at the top-left of the input matrix. So, the convolution operation on this slice looks as follows:

<img src="ConvolutionalLayerOperation.svg">

A convolutional layer consists of a series of convolutional operations, each acting on a different slice of the input matrix.

## decision tree
A model represented as a sequence of branching statements. For example, the following over-simplified decision tree branches a few times to predict the price of a house (in thousands of USD). According to this decision tree, a house larger than 160 square meters, having more than three bedrooms, and built less than 10 years ago would have a predicted price of 510 thousand USD.

<img src="DecisionTree.svg" title="A tree three-levels deep whose branches predict house prices.">

Machine learning can generate deep decision trees.

## Feature Pyramid Network (FPN)
    
In RPN, we have built anchor boxes only using the top high level feature map. Though convnets are robust to variance in scale, all the top entries in ImageNet or COCO have used multi-scale testing on featurized image pyramids. Imagine taking a 800 * 800 image and detecting bounding boxes on it. Now if your are using image pyramids, we have to take images at different sizes say 256*256, 300*300, 500*500 and 800*800 etc, calculate feature maps for each of this image and then apply non-maxima supression over all these detected positive anchor boxes. This is a very costly operation and inference times gets high.
      <p align="center">
        <img src="pic/deep_learning_using_image_pyramids.png" width="800px" title="Deep Learning Using Image Pyramids">
      </p>
       
Deep convnet computes a feature hierarchy layer by layer, and with subsampling layers the feature hierarchy has an inherent multi-scale, pyramidal shape. For example, take a Resnet architecture and instead of just using the final feature map as shown in RPN network, take feature maps before every pooling (subsampling) layer. Perform the same operations as for RPN on each of these feature maps and finally combine them using non-maxima supression. This is the crude way of building the feature pyramid networks. But there is one of the problem with this approach, there are large semantic gaps caused by different depths. The high resolution maps (earlier layers) have low-level features that harm their representational capacity for object detection.
     <p align="center">
       <img src="pic/feature_pyramid_network_rough.png" width="800px" title="Feature Pyramid Network Rough">
     </p>
       
The goal of the authors is to naturally leverage the pyramidal shape of a Convnet feature hierarchy while creating a feature pyramid that has strong semantics at all scales. To achieve this goal, the authors relayed on a architecture that combines low-resolution, semantically strong features with high-resolution, semantically strong features via top-down pathway and lateral connection as shown in the diagram below.
     <p align="center">
       <img src="pic/feature_pyramid_network.png" width="800px" title="Feature Pyramid Network">
     </p>
     <p align="center">
       <img src="pic/feature_pyramid_network_output.png" width="600px" title="Feature Pyramid Network Output">
     </p>
       
The predictions are made on each level independently.
       
Important points while designing anchor boxes:
       
* Since the pyramids are of different scales, no need to have multi-scale anchors on a specific level. We define the anchors to have size of [32, 54, 128, 256, 512] on P3, P4, P5, P6, P7 respectively. We use anchors of multiple aspect ratio [1:1, 1:2, 2:1]. so in-total there will be 15 anchors over the pyramid at each location.

* All the anchor boxes outside image dimensions were ignored.

* positive if the given anchor box has highest IoU with the ground truth box or if the IoU is greater than 0.7. negative if the IoU is less than 0.3.

* The scales of the ground truth boxes are not used to assign them to levels of the pyramid. Instead, ground-truth boxes are associated with anchors, which have been assigned to pyramid levels. This above statement is very important to understand. I had two confusions here, weather we need to assign ground truth boxes to each level separately or compute all the anchor boxes and then assign label to the anchor box with which it has max IoU or IoU greater than 0.7. Finally I have chosen the second option to assign labels.       

## Focal Loss  
  
Methods like SSD or YOLO suffer from an extreme class imbalance: The detectors evaluate roughly between ten to hundred thousand candidate locations and of course most of these boxes do not contain an object. Even if the detector easily classifies these large number of boxes as negatives/background, there is still a problem.
    
### Cross entropy loss function      

<p align="center">
          <img src="pic/cross_entropy_loss_function.png" width="300pm" title="Cross Entropy Loss Function">
</p>
      
<p align="center">
          <img src="pic/cross_entropy_loss_function_plot.png" width="600pm" title="Cross Entropy Loss Function Plot">
</p>
      
where i is the index of the class, y_i the label (1 if the object belongs to class i, 0 otherwise), and p_i is the predicted probability that the object belongs to class i.
      
Let’s say a box contains background and the network is 80% sure that it actually is only background. In this case y(background)=1, all other y_i are 0 and p(background)=0.8.
      
You can see that at 80% certainty that the box contains only background, the loss is still ~0.22. The large number of easily classified examples absolutely dominates the loss and thus the gradients and therefore overwhelms the few interesting cases the network still has difficulties with and should learn from.
      
### Focal Loss Function
    
Lin et al. (2017) [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) had the beautiful idea to scale the cross entropy loss so that all the easy examples the network is already very sure about contribute less to the loss so that the learning can focus on the few interesting cases. The authors called their loss function <i>Focal loss </i>and their architecture <b>RetinaNet</b> (note that RetinaNet also includes <b>Feature Pyramid Networks (FPN)</b> which is basically a new name for U-Net).

<p align="center">
         <img src="pic/focal_loss_function.png" width="300px" title="Focal Loss Function">
</p>

<p align="center">
         <img src="pic/focal_loss_function_plot.png" width="600px" title="Focal Loss Function_plot">
</p>
        
With this rescaling, the large number of easily classified examples (mostly background) does not dominate the loss anymore and learning can concentrate on the few interesting cases.


## hashing
In machine learning, a mechanism for bucketing categorical data, particularly when the number of categories is large, but the number of categories actually appearing in the dataset is comparatively small.

For example, Earth is home to about 60,000 tree species. You could represent each of the 60,000 tree species in 60,000 separate categorical buckets. Alternatively, if only 200 of those tree species actually appear in a dataset, you could use hashing to divide tree species into perhaps 500 buckets.

A single bucket could contain multiple tree species. For example, hashing could place baobab and red maple—two genetically dissimilar species—into the same bucket. Regardless, hashing is still a good way to map large categorical sets into the desired number of buckets. Hashing turns a categorical feature having a large number of possible values into a much smaller number of values by grouping values in a deterministic way.

## Intersection over Union (IoU)

__Intersection Over Union (IoU)__ is a number that quantifies the degree of overlap between two boxes. In the case of object detection and segmentation, __IoU__ evaluates the overlap of the __Ground Truth__ and __Prediction__ region.

For example, in the image below:

  * The predicted bounding box (the coordinates delimiting where the model predicts the night table in the painting is located) is outlined in purple.
  
  * The ground-truth bounding box (the coordinates delimiting where the night table in the painting is actually located) is outlined in green.

<img src="pic/iou_van_gogh_bounding_boxes.jpg">

Here, the intersection of the bounding boxes for prediction and ground truth (below left) is 1, and the union of the bounding boxes for prediction and ground truth (below right) is 7, so the IoU is 1/7.

<img src="pic/iou_van_gogh_intersection.jpg">

<img src="pic/iou_van_gogh_union.jpg">

<img src="pic/Understanding-Intersection-Over-Union-in-Object-Detection-and-Segmentation.jpg">

Let’s go through the following example to understand how IoU is calculated. Let there be three models- A, B, and C, trained to predict birds. We pass an image through the models where we already know the __Ground Truth (marked in red)__. The image below shows __predictions__ of the models __(marked in cyan)__.

<img src="pic/1-bird-detection-by-different-models-1024x387.jpg">

IoU is the ratio of the __overlap area__ to the __combined area of prediction__ and __ground truth__.

IoU values range from 0 to 1. Where 0 means no overlap and 1 means perfect overlap.
<p align="center">
<img src="pic/2-iou-illustration-768x300.jpg">
</p>

Looking closely, we are adding the area of the intersection __twice__ in the denominator. So actually we calculate IoU as shown in the illustration below.

<p align="center">
<img src="pic/3-iou-illustration-300x119.jpg">
</p>

### Qualitative Analysis of Predictions

With the help of the IoU threshold, we can decide whether the prediction is __True Positive(TP)__, __False Positive(FP)__, or __False Negative(FN)__. The example below shows predictions with the IoU threshold __ɑ__ set at __0.5__.

<img src="pic/4-birds-prediction-types-1.jpg">

The decision of making a detection as __True Positive__ or __False Positive__, completely depends on the requirement.

- The first prediction is __True Positive__ as the IoU threshold is 0.5.
- If we set the threshold at 0.97, then it becomes a __False Positive__.
- Similarly, the second prediction shown above is __False Positive__ due to the threshold but can be __True Positive__ if we set the threshold at 0.20.
- Theoretically, the third prediction can also be __True Positive__, given that we lower the threshold all the way to 0.

### Intersection over Union in Image Segmentation

__IoU in object detection is a helper metric__. However, in image segmentation; IoU is the primary metric to evaluate model accuracy.

In the case of Image Segmentation, the area is not necessarily rectangular. It can have any regular or irregular shape. That means the predictions are segmentation masks and not bounding boxes. Therefore, pixel-by-pixel analysis is done here. Moreover, the definition of TP, FP, and FN is slightly different as it is not based on a predefined threshold.

(a) __True Positive__: The area of intersection between Ground Truth(__GT__) and segmentation mask(__S__). Mathematically, this is __logical AND__ operation of GT and S i.e., 

$$TP = GT.S$$

(b) __False Positive__: The predicted area outside the Ground Truth. This is the __logical OR__ of GT and segmentation minus GT. 

$$FP = (GT + S) - GT$$

(c) __False Negative__: Number of pixels in the Ground Truth area that the model failed to predict. This is the __logical OR__ of GT and segmentation minus S.

$$FN = (GT + S) - S$$


We know from Object Detection that IoU is the ratio of the __intersected area__ to the __combined area__ of __prediction__ and __ground truth__. Since the values of TP, FP, and FN are nothing but areas or number of pixels; we can write IoU as follows.

$$IoU = \dfrac{TP} {TP + FP + FN} $$


<img src="pic/5-segmentation-iou-1024x296.jpg">

## Non Maximum Suppression (NMS)

Non Maximum Suppression (NMS) is a technique used in numerous computer vision tasks. It is a class of algorithms to select one entity (e.g., bounding boxes) out of many overlapping entities. We can choose the selection criteria to arrive at the desired results. The criteria are most commonly some form of probability number and some form of overlap measure (e.g. Intersection over Union).

<p align="center">
  <img src="pic/nms-intro.jpg">
</p>

### Why we need it?

Most object detection algorithms use NMS to whittle down many detected bounding boxes to only a few. At the most basic level, most object detectors do some form of __windowing__. Thousands of windows (__anchors__) of various __sizes and shapes__ are generated.

These windows supposedly contain only one object, and a classifier is used to obtain a probability/score for each class. Once the detector outputs a large number of bounding boxes, it is necessary to filter out the best ones. NMS is the most commonly used algorithm for this task.

### The NMS Algorithm

Let us get to the nitty-gritty of this post, the actual algorithm. I will divide this into three parts, what we need as input, what we get after applying the algorithm and the actual algorithm itself.

#### Input
We get a list `P` of prediction BBoxes of the form `(x1,y1,x2,y2,c)`, where `(x1,y1)` and `(x2,y2)` are the ends of the BBox and `c` is the predicted confidence score of the model. We also get overlap threshold IoU `thresh_iou`.

#### Output
We return a list `keep` of filtered prediction BBoxes.

#### Algorithm

- Step 1 : Select the prediction `S` with highest confidence score and remove it from `P` and add it to the final prediction list `keep`. (`keep` is empty initially).

- Step 2 : Now compare this prediction `S` with all the predictions present in `P`. Calculate the IoU of this prediction `S` with every other predictions in `P`. If the IoU is greater than the threshold `thresh_iou` for any prediction `T` present in `P`, remove prediction `T` from `P`.

- Step 3 : If there are still predictions left in `P`, then go to __Step 1__ again, else return the list `keep` containing the filtered predictions.

In layman terms, we select the predictions with the `maximum confidence` and `suppress all the other predictions` having overlap with the selected predictions greater than a threshold. In other words, we `take the maximum and suppress the non-maximum ones`, hence the name non-maximum suppression.

If you observe the algorithm above, the whole filtering process depends on a single threshold value thresh_iou. So selection of threshold value is vital for the performance of the model. Usually, we take its value as 0.5, but it depends on the experiment you are doing.As discussed in the NMS algorithm above, we extract the BBox of highest confidence score and remove it from P.

## Region Proposal Network (RPN)
     
* <b>Regression head</b>: The output of the Faster RPN network as discussed and shown in the image above is a 50x50 feature map. A conv layer [kernal 3x3] strides through this image, At each location it predicts the 4 [x1, y1, h1, w1] values for each anchor boxes (9). In total, the output layer has 50x50x9x4 output probability scores. Usually this is represented in numpy as np.array(2500, 36).
     
* <b>Classification head</b>: Similar to the Regression head, this will predict the probability of an object present or not at each location for each anchor bos. This is represented in numpy array as np.array(2500, 9).
     <p align="center">
        <img src="pic/RPN_network_head.png" width="800px" title="RPN network head">
     </p>
     
* Problems with RPN
      
  * The Feature map created after a lot of subsampling losses a lot of semantic information at low level, thus unable to detect small objects in the image. <b>[Feature Pyramid networks solves this]</b>
          
  * The loss functions uses negative hard-mining by taking 128 +ve samples, 128 -ve samples because using all the labels hampers training as it is highly imbalanced and there will be many easily classified examples. <b>[Focal loss solves this]</b>