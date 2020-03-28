# SSD-Single Shot Detector
By using SSD, we only need to <b>take one single shot to detect multiple objects within the image</b>, while regional proposal network (RPN) based approaches such as R-CNN series that need two shots, one for generating region proposals, one for detecting the object of each proposal. Thus, SSD is much faster compared with two-shot RPN-based approaches.
  * MultiBox Detector
  
   <p align="center">
       <img src="ssd_multiple_bounding_boxes_for_localization_and_confidence.png" width="600px" title="SSD: Multiple Bounding Boxes for Localization (loc) and Confidence (conf)>
    </p>
                                                                                                  
    * After going through a certain of convolutions for feature extraction, we obtain <b>a feature layer of size m×n with p channels</b>, such as 8×8 or 4×4 above. And a 3×3 conv is applied on this m×n×p feature layer.
    * <b>For each location, we got k bounding boxes</b>. These k bounding boxes have different sizes and aspect ratios. The concept is, maybe a vertical rectangle is more fit for human, and a horizontal rectangle is more fit for car.
    * <b>For each of the bounding box, we will compute c class scores and 4 offsets relative to the original default bounding box shape</b>.
    * Thus, we got <b>(c+4)kmn outputs</b>.
