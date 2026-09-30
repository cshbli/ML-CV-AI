# Segment Anything (SAM)

**Segment Anything (SAM)** and **SAM 2** are foundation models for **promptable segmentation**: you guide masks with points, boxes, or masks (SAM 2 also handles video). They are strong at open-world / class-agnostic masks, interactive annotation, and zero-shot transfer — not primarily closed-set “label every pixel as road/car/person” heads out of the box.

Official code: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) · [facebookresearch/sam2](https://github.com/facebookresearch/sam2)

See also: [Computer Vision overview](../README.md) · [CNN vs ViT](../../classification/cnn_vs_vit.md)

## Is SAM the current SOTA?

**Short answer:** SAM is state of the art for **promptable / interactive / zero-shot** segmentation — not automatically the best model for every segmentation job. What “SOTA” means depends on the task.

### What SAM is good at

* Open-world / class-agnostic masks
* Interactive annotation (clicks, boxes)
* Zero-shot transfer to new domains
* “Segment anything” without training a head per class

SAM is **not** primarily a closed-set **semantic** segmenter that outputs fixed class names alone (you need prompting, grounding, or another model on top).

### By task

| Task | Is SAM “the” SOTA? | Strong alternatives |
|------|--------------------|---------------------|
| **Promptable / interactive / zero-shot masks** | **Yes — SAM / SAM 2** are the reference | HQ-SAM, MobileSAM, FastSAM, EfficientSAM (speed/quality tradeoffs) |
| **Video object segmentation** | **SAM 2** is a top generalist | Cutie, XMem, specialist VOS models on some benchmarks |
| **Closed-set semantic seg** (ADE20K, Cityscapes, medical labels) | **Usually no** | Mask2Former, OneFormer, SegFormer, InternImage-UPerNet, domain-specific U-Nets |
| **Instance / panoptic** (COCO-style) | Competitive via combos, not always pure SAM | Mask2Former, Mask DINO, OneFormer |
| **Text-driven “segment the cat”** | SAM alone needs help | **Grounded-SAM**, SEEM, Semantic-SAM, CLIPSeg, recent VLMs — and open-vocab **detectors** first: [open-vocabulary detection](../object_detection/open_vocabulary_detection.md) |

### Practical takeaway

* Need **flexible masks with clicks/boxes**, labeling tools, or open-world cropping → start with **SAM 2**.
* Need **fixed class labels** on your dataset (like fine-tuning ResNet for classification) → fine-tune a **semantic/instance** model (Mask2Former / SegFormer / U-Net family); use SAM to **help annotate**, not as the only production head.
* Need **language → mask** → Grounded-SAM / SEEM-style stacks, not vanilla SAM alone. For text → **boxes** first, see [open-vocabulary / zero-shot detection](../object_detection/open_vocabulary_detection.md).

So: SAM (especially **SAM 2**) is the current go-to **general segmentation foundation model**, but for classical per-pixel classification on custom classes, specialized semantic/instance models are still often better — and “SOTA” on those benchmarks is a moving race among Mask2Former-class architectures, not SAM by default.

## SAM vs classical segmentation (quick map)

| | **SAM / SAM 2** | **Semantic / instance models** (e.g. Mask2Former) |
|--|-----------------|-----------------------------------------------------|
| Output | Class-agnostic masks (prompted) | Per-pixel / per-instance **class labels** |
| Supervision at use | Prompt (point/box/mask) | Trained on labeled classes |
| Custom dataset | Great for annotation + zero-shot masks | Fine-tune when you need your ontology |
| Video | SAM 2 | Task-specific VOS / tracking stacks |

## References

* A. Kirillov et al., [Segment Anything](https://arxiv.org/abs/2304.02643) (2023).
* N. Ravi et al., [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) (2024).
* [segment-anything](https://github.com/facebookresearch/segment-anything) · [sam2](https://github.com/facebookresearch/sam2)
