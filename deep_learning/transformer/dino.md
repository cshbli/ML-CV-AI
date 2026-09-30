# DINO, DINOv2, and DINOv3

The **DINO** family (Meta FAIR) is a line of **self-supervised vision backbones**. They learn **general visual features** from images **without labels** (no class tags, no captions). They are **feature extractors / foundation encoders**, not detectors or segmenters by themselves.

**Name:** DINO = **Di**stillation with **no** labels (self-distillation).  
Do not confuse with **Grounding DINO** (open-vocabulary **detector**) — different model; see [open-vocabulary detection](../../computer_vision/object_detection/open_vocabulary_detection.md).

See also: [Vision Transformer](vision_transformer.md) · [CNN vs ViT](../../classification/cnn_vs_vit.md) · [Segment Anything](../../computer_vision/Segmentation/segment_anything.md)

## What DINO-style models output

* A global **CLS** embedding (whole image) — classification, retrieval, k-NN
* **Patch** embeddings (dense / local features) — segmentation cues, depth, matching, detection heads

Typical use today: **freeze the backbone** → linear probe or small head, or light fine-tune.

---

## DINO (2021) — the original method

**Paper:** *Emerging Properties in Self-Supervised Vision Transformers* (Caron et al., ICCV 2021).

DINO showed that **self-supervised ViTs** learn qualitatively different features than supervised ViTs or many convnets — especially **semantic layout / object boundaries** visible in attention maps, and strong **k-NN** classification from frozen features.

### Core idea (self-distillation, no labels)

```text
same image → different crops / augmentations
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   Student network         Teacher network
   (trained by CE)         (EMA of student)
        │                       │
        └──────── match ────────┘
         student predicts teacher
```

* **Student** and **teacher** share the same architecture (often a ViT).
* Teacher weights are an **EMA (momentum)** copy of the student — not trained by backprop.
* **Multi-crop:** teacher sees **global** crops; student also sees **local** crops → learn local-to-global consistency.
* Loss is basically **cross-entropy**: student matches the teacher’s softmax output.
* **Centering + sharpening** of the teacher distribution avoids **collapse** (everything mapping to one trivial feature), without needing a heavy contrastive negative bank.

So DINO is “knowledge distillation” where the teacher is built from the student itself — **no human labels**.

### Why it mattered

* Simple SSL recipe that works especially well with **ViTs**
* Emergent **semantic segmentation-like** structure in features / attention
* Strong frozen features for **k-NN** and linear eval on ImageNet
* Foundation for later **DINOv2** / **DINOv3** scaling

Repo: [facebookresearch/dino](https://github.com/facebookresearch/dino) · Paper: [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)

---

## DINOv2 (2023)

**DINOv2** scales and hardens the DINO idea into a **production-style foundation model**: trained on a large curated set (~142M images). Features are strong **out of the box** for many tasks with little or no fine-tuning of the backbone.

**What DINOv2 can do well**

* **Image classification** / fine-grained recognition (linear head or light fine-tune) — especially useful on **small custom datasets**
* **Semantic segmentation** / object-part understanding (dense patch features)
* **Monocular depth** (surprisingly strong with a simple head)
* **Image / instance retrieval** (similarity search)
* **Feature matching** across images
* Transfer features for video understanding

**What it is not**

* Not [SAM](../../computer_vision/Segmentation/segment_anything.md) — no promptable masks out of the box
* Not YOLO / [Grounding DINO](../../computer_vision/object_detection/open_vocabulary_detection.md) — no boxes from a forward pass alone
* Not CLIP by design — **vision-only** SSL (no native text tower)

Repo: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)

---

## DINOv3 (2025)

**DINOv3** (Meta, Aug 2025) is the scaled successor: same role (universal SSL vision features), trained much larger.

Reported highlights vs DINOv2:

* Much larger scale (on the order of **~1.7B images**, models up to **~7B** parameters)
* **Gram anchoring** to keep **dense** feature maps sharp during long training (dense quality used to degrade when scaling)
* Strong results with a **frozen** backbone on many **dense** tasks (segmentation, detection heads, etc.), including broader domains (e.g. satellite variants)
* A **family** of distilled smaller **ViT** and **ConvNeXt** models for deployment; optional post-hoc flexibility (resolution, text alignment)

**What DINOv3 can do** — same jobs as v2, generally stronger dense features and more size options. Prefer DINOv3 when you want the latest backbone; keep DINOv2 if your stack already depends on it and is good enough.

Repo / paper: [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) · [Meta DINOv3 blog](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/)

---

## Family comparison

| | **DINO (2021)** | **DINOv2 (2023)** | **DINOv3 (2025)** |
|--|-----------------|-------------------|-------------------|
| Role | SSL method + early ViT features | Scaled foundation backbone | Further scaled foundation family |
| Focus | Self-distillation recipe; emergent properties | Curated data + robust universal features | Scale + dense-feature quality (Gram anchoring) |
| What you use today | Historical / research baseline | Still excellent default SSL backbone | Latest / strongest dense features |
| vs YOLO / SAM | Feature encoder only | Same | Same |

## Practical takeaway

* Need a **pretrained visual encoder** for your dataset (classify, retrieve, feed a det/seg/depth head) → **DINOv2 or DINOv3**.
* Need **boxes or masks as the product** → [YOLO](../../computer_vision/object_detection/yolo_is_sota.md) / [open-vocab detection](../../computer_vision/object_detection/open_vocabulary_detection.md) / [SAM](../../computer_vision/Segmentation/segment_anything.md); DINO can still be the backbone behind a head.
* Fine-tuning tips for ViT-style models: [Fine-tuning ViT on custom data](vision_transformer.md#fine-tuning-vit-on-custom-data).

## References

* M. Caron et al., [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294) (DINO, ICCV 2021).
* M. Oquab et al., [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) (2023).
* [DINOv3 technical report](https://ai.meta.com/research/publications/dinov3/) (2025).
* [dino](https://github.com/facebookresearch/dino) · [dinov2](https://github.com/facebookresearch/dinov2) · [dinov3](https://github.com/facebookresearch/dinov3)
