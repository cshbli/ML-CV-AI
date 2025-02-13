﻿# SSD-Single Shot Detector

By using SSD, we only need to <b>take one single shot to detect multiple objects within the image</b>, while regional proposal network (RPN) based approaches such as R-CNN series that need two shots, one for generating region proposals, one for detecting the object of each proposal. Thus, SSD is much faster compared with two-shot RPN-based approaches.
  * <b>MultiBox Detector</b>
    <p align="center">
       <img src="ssd_multiple_bounding_boxes_for_localization_and_confidence.png" width="600px" title="SSD: Multiple Bounding Boxes for Localization (loc) and Confidence (conf)">
    </p>
                                                                                                  
    * After going through a certain of convolutions for feature extraction, we obtain <b>a feature layer of size m×n with p channels</b>, such as 8×8 or 4×4 above. And a 3×3 conv is applied on this m×n×p feature layer.
    
    * <b>For each location, we got k bounding boxes</b>. These k bounding boxes have different sizes and aspect ratios. The concept is, maybe a vertical rectangle is more fit for human, and a horizontal rectangle is more fit for car.
    
    * <b>For each of the bounding box, we will compute c class scores and 4 offsets relative to the original default bounding box shape</b>.
    
    * Thus, we got <b>(c+4)kmn outputs</b>.
    
  * <b>SSD Network Architecture</b>
     <p align="center">
        <img src="SSD_vs_YOLO.png" width="800px" title="SSD (Top) vs YOLO (Bottom)">
     </p>
    
    SSD uses multiple layers <b>(multi-scale feature maps)</b> to detect objects independently. As CNN reduces the spatial dimension gradually, the resolution of the feature maps also decrease. SSD uses lower resolution layers to detect larger scale objects.
    
    To have more accurate detection, different layers of feature maps are also going through a small 3×3 convolution for object detection as shown above.
    * Say for example, at Conv4_3, it is of size 38×38×512. 3×3 conv is applied. And there are 4 bounding boxes and each bounding box will have (classes + 4) outputs. Thus, at Conv4_3, the output is 38×38×4×(c+4). Suppose there are 20 object classes plus one background class, the output is 38×38×4×(21+4) = 144,400. In terms of number of bounding boxes, there are 38×38×4 = 5776 bounding boxes.
    
    Similarly for other conv layers:
   
    * Conv7: 19×19×6 = 2166 boxes (6 boxes for each location)
    
    * Conv8_2: 10×10×6 = 600 boxes (6 boxes for each location)
    
    * Conv9_2: 5×5×6 = 150 boxes (6 boxes for each location)
    
    * Conv10_2: 3×3×4 = 36 boxes (4 boxes for each location)
    
    * Conv11_2: 1×1×4 = 4 boxes (4 boxes for each location)
    
    If we sum them up, we got 5776 + 2166 + 600 + 150 + 36 +4 = 8732 boxes in total. If we remember YOLO, there are 7×7 locations at the end with 2 bounding boxes for each location. YOLO only got 7×7×2 = 98 boxes. Hence, SSD has 8732 bounding boxes which is more than that of YOLO.

  * <b>Loss Function</b>
    * The loss function consists of two terms: Lconf and Lloc
    
    * Lloc is the localization loss which is the smooth L1 loss between the predicted box (l) and the ground-truth box (g) parameters.
    
    * Lconf is the confidence loss which is the softmax loss over multiple classes confidences (c).
  
  * <b>Some Details of Training</b>
    * <b>Hard Negative Mining</b>
      Instead of using all the negative examples, we sort them using the highest confidence loss for each default box and pick the top ones so that the ratio between the negatives and positives is at most 3:1.
      
      This can lead to faster optimization and a more stable training.
      
    * <b>Scale and Aspect Ratios</b>
      Default boundary boxes are chosen manually. SSD defines a scale value for each feature map layer. Starting from the left, Conv4_3 detects objects at the smallest scale 0.2 (or 0.1 sometimes) and then increases linearly to the rightmost layer at a scale of 0.9. Combining the scale value with the target aspect ratios, we compute the width and the height of the default boxes. For layers making 6 predictions, SSD starts with 5 target aspect ratios: 1, 2, 3, 1/2 and 1/3. Then the width and the height of the default boxes are calculated as:
      <p align="center">
         <img src="ssd_scale_and_aspect_ratio.png" width="600px" title="SSD Scale and Ratios">
      </p>
      
      and aspect_ratio = 1.
    * <b>Atrous Convolution (Hole Algorithm / Dilated Convolution)</b>
    
      The base network is VGG16 and pre-trained using ILSVRC classification dataset. <b>FC6 and FC7 are changed to convolution layers as Conv6 and Conv7 </b>which is shown in the figure above.
     
      Furthermore, <b>FC6 and FC7 use Atrous convolution </b>(a.k.a Hole algorithm or dilated convolution) instead of conventional convolution.       
      <p align="center">
          <img src="atrous_convolution_hole_algorithm_dilated_convolution.gif" width="400px" title="Atrous convolution / Hole algorithm / Dilated convolution">
      </p>

      As we can see, the feature maps are large at Conv6 and Conv7, using Atrous convolution as shown above can <b>increase the receptive field while keeping number of parameters relatively fewer </b>compared with conventional convolution. 
  
     * <b>Data Augmentation</b>
       The authors of SSD stated that data augmentation, like in many other deep learning applications, has been crucial to teach the network to become more robust to various object sizes in the input. To this end, they generated additional training examples with patches of the original image at different IoU ratios (e.g. 0.1, 0.3, 0.5, etc.) and random patches as well. Moreover, each image is also randomly horizontally flipped with a probability of 0.5, thereby making sure potential objects appear on left and right with similar likelihood.
       
      * <b>Non-Maximum Suppression (NMS)</b>
        Given the large number of boxes generated during a forward pass of SSD at inference time , it is essential to prune most of the bounding box by applying a technique known as non-maximum suppression: boxes with a confidence loss threshold less than ct (e.g. 0.01) and IoU less than lt (e.g. 0.45) are discarded, and only the top N predictions are kept. This ensures only the most likely predictions are retained by the network, while the more noisier ones are removed.
        
