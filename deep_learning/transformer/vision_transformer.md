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

## Repo map

| Topic | Where |
|-------|--------|
| CNN vs ViT for **classification** | [classification/cnn_vs_vit.md](../../classification/cnn_vs_vit.md) |
| Attention mechanics | [attention.ipynb](attention.ipynb) |
| Text Transformer | [LLM/transformer.md](../../LLM/transformer.md) |
| Official ViT code | [google-research/vision_transformer](https://github.com/google-research/vision_transformer) |

## References

* A. Dosovitskiy et al., [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) (2020).
* H. Touvron et al., [Training data-efficient image transformers (DeiT)](https://arxiv.org/abs/2012.12877).
* Z. Liu et al., [Swin Transformer](https://arxiv.org/abs/2103.14030).
