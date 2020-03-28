# Pix2Pix: Image-to-Image Translation with Conditional Adversarial Netwoks

[Pix2Pix](https://arxiv.org/pdf/1611.07004.pdf) network is basically a Conditional GANs (cGAN) that learn the mapping from an input image to output an image. 

Image-To-Image Translation is a process for translating one representation of an image into another representation.

* The Generator Network

  Generator network uses a <b>U-Net</b>-based architecture. U-Net’s architecture is similar to an <b>Auto-Encoder</b> network except for one difference. Both U-Net and Auto-Encoder network has two networks The <b>Encoder</b> and the <b>Decoder</b>.
  
  * U-Net Architecture Diagram
    <p align="center">
      <img src="unet_architecture_diagram" width="400px" title="U-Net Architecture">
    </p>
    
    * U-Net’s network has skip connections between Encoder layers and Decoder layers.
    
    * As shown in the picture the output of the first layer of Encoder is directly passed to the last layer of the Decoder and output of the second layer of Encoder is pass to the second last layer of the Decoder and so on.
    
    * Let’s consider if there are total N layers in U-Net’s(including middle layer), Then there will be a skip connection from the kth layer in the Encoder network to the (N-k+1)th layer in the Decoder network. where 1 ≤ k ≤ N/2.
