# Vision Transformer (ViT)

Vision Transformer applies the **Transformer** encoder to images by splitting an image into patches, embedding them as tokens, and running self-attention — the same family of ideas as NLP Transformers, adapted for vision.

For **when to use ViT vs a CNN (e.g. ResNet50) for image classification**, see [CNN vs ViT](../../classification/cnn_vs_vit.md).

Related: [Attention from scratch](attention.ipynb) · [LLM Transformer overview](../../LLM/transformer.md) · [Residual / CNN blocks](../residual_block/README.md)

## Core idea

1. Split the image into fixed-size patches (e.g. 16×16).
2. Flatten each patch and project it to a embedding vector (**patch embedding**).
3. Add **positional embeddings** (patches have no inherent order like conv grids).
4. Prepend a learnable **[CLS]** token (optional but common for classification).
5. Feed the token sequence through a standard **Transformer encoder** (MHSA + MLP blocks).
6. Classify from the CLS token (or pooled patch tokens).

Unlike ResNet, there is no convolution inductive bias for locality — the model must learn spatial structure from data or from pretraining.

## Variants you will meet

| Family | Notes |
|--------|--------|
| **ViT** | Original patch + encoder; strong with large pretrain |
| **DeiT** | Data-efficient training / distillation recipes for ViT |
| **Swin** | Hierarchical windows; popular backbone for detection/seg too |
| **ConvNeXt** | Modern CNN inspired by Transformer training recipes (not a ViT, but a common alternative) |

## Fine-tuning ViT on custom data

Same idea as fine-tuning ResNet: load pretrained weights, replace the head, train with a smaller backbone LR. Prefer **pretrained** ViTs; scratch ViT on a small custom set usually loses to ResNet.

**Classification**

1. Load ImageNet-pretrained ViT (or DINOv2 / MAE / CLIP-ViT when it fits).
2. Set `num_classes` to your label count (new linear head).
3. Optional: freeze backbone briefly, then unfreeze with backbone LR ≈ 0.1× head LR.
4. Match checkpoint **image size** (224 / 384 / …) and normalization.
5. Optimizer: **AdamW** + cosine / OneCycle; weight decay matters more than in many small CNN fine-tunes.

**Regression:** same backbone; head `Linear(hidden, 1)` (or `K` targets); loss MSE / SmoothL1 / Huber.

**Starter checkpoints**

| Model | When |
|-------|------|
| ViT-B/16 AugReg (ImageNet-21k→1k) | Default classification fine-tune |
| DeiT-B / DeiT III | Data-efficient ImageNet-style ViT |
| DINOv2 / DINOv3 ViT | Small custom sets; strong linear probe / light fine-tune — see [DINO](dino.md) |
| Swin-T/S/B | Want multi-scale Transformer (closer to CNN habits) |
| CLIP / SigLIP ViT | Noisy labels, embeddings, later VLM work |

**Minimal timm example**

```python
import timm
model = timm.create_model(
    "vit_base_patch16_224.augreg_in21k_ft_in1k",
    pretrained=True,
    num_classes=10,  # classification; use 1 for single-target regression
)
```

torchvision (`vit_b_16` + replace `heads.head`) and Hugging Face `ViTForImageClassification` / `Trainer` are equivalent paths.

**Tips:** for &lt;5k images prefer freeze / [DINOv2](dino.md) probe; use Mixup–CutMix / drop-path when training harder; watch VRAM (ViT-B &gt; ResNet50 at the same batch).

**Libraries / repos:** [timm](https://github.com/huggingface/pytorch-image-models) · [torchvision ViT](https://pytorch.org/vision/stable/models/vision_transformer.html) · [transformers image classification](https://huggingface.co/docs/transformers/tasks/image_classification) · [DINOv2](https://github.com/facebookresearch/dinov2) · [DINOv3](https://github.com/facebookresearch/dinov3) · [MAE](https://github.com/facebookresearch/mae) · [Swin](https://github.com/microsoft/Swin-Transformer) · [google-research/vision_transformer](https://github.com/google-research/vision_transformer)

Task comparison with CNNs: [CNN vs ViT](../../classification/cnn_vs_vit.md).

## Repo map

| Topic | Where |
|-------|--------|
| CNN vs ViT for **classification** | [classification/cnn_vs_vit.md](../../classification/cnn_vs_vit.md) |
| **DINO / DINOv2 / DINOv3** (SSL backbones) | [dino.md](dino.md) |
| Attention mechanics | [attention.ipynb](attention.ipynb) |
| Text Transformer | [LLM/transformer.md](../../LLM/transformer.md) |
| Official ViT code | [google-research/vision_transformer](https://github.com/google-research/vision_transformer) |

## References

* A. Dosovitskiy et al., [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) (2020).
* H. Touvron et al., [Training data-efficient image transformers (DeiT)](https://arxiv.org/abs/2012.12877).
* Z. Liu et al., [Swin Transformer](https://arxiv.org/abs/2103.14030).
