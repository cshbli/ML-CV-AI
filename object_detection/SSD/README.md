# SSD-Single Shot Detector
By using SSD, we only need to <b>take one single shot to detect multiple objects within the image</b>, while regional proposal network (RPN) based approaches such as R-CNN series that need two shots, one for generating region proposals, one for detecting the object of each proposal. Thus, SSD is much faster compared with two-shot RPN-based approaches.
  * MultiBox Detector
    <p align="center">
       <img src="ssd_multiple_bounding_boxes_for_localization_and_confidence.png" width="600px" title="SSD: Multiple Bounding Boxes for Localization (loc) and Confidence (conf)">
    </p>
                                                                                                  
    * After going through a certain of convolutions for feature extraction, we obtain <b>a feature layer of size m×n with p channels</b>, such as 8×8 or 4×4 above. And a 3×3 conv is applied on this m×n×p feature layer.
    
    * <b>For each location, we got k bounding boxes</b>. These k bounding boxes have different sizes and aspect ratios. The concept is, maybe a vertical rectangle is more fit for human, and a horizontal rectangle is more fit for car.
    
    * <b>For each of the bounding box, we will compute c class scores and 4 offsets relative to the original default bounding box shape</b>.
    
    * Thus, we got <b>(c+4)kmn outputs</b>.
    
  * SSD Network Architecture
     <p align="center">
        <img src="SSD_vs_YOLO.png" width="800px" title="SSD (Top) vs YOLO (Bottom)">
     </p>
    
    To have more accurate detection, different layers of feature maps are also going through a small 3×3 convolution for object detection as shown above.
    * Say for example, at Conv4_3, it is of size 38×38×512. 3×3 conv is applied. And there are 4 bounding boxes and each bounding box will have (classes + 4) outputs. Thus, at Conv4_3, the output is 38×38×4×(c+4). Suppose there are 20 object classes plus one background class, the output is 38×38×4×(21+4) = 144,400. In terms of number of bounding boxes, there are 38×38×4 = 5776 bounding boxes.
    
    Similarly for other conv layers:
   
    * Conv7: 19×19×6 = 2166 boxes (6 boxes for each location)
    
    * Conv8_2: 10×10×6 = 600 boxes (6 boxes for each location)
    
    * Conv9_2: 5×5×6 = 150 boxes (6 boxes for each location)
    
    * Conv10_2: 3×3×4 = 36 boxes (4 boxes for each location)
    
    * Conv11_2: 1×1×4 = 4 boxes (4 boxes for each location)
    
    If we sum them up, we got 5776 + 2166 + 600 + 150 + 36 +4 = 8732 boxes in total. If we remember YOLO, there are 7×7 locations at the end with 2 bounding boxes for each location. YOLO only got 7×7×2 = 98 boxes. Hence, SSD has 8732 bounding boxes which is more than that of YOLO.
