# CNN (e.g. ResNet50) vs Vision Transformer for image classification

Neither is universally better — it depends on **data size, compute, and deployment**.

See also: [Vision Transformer (architecture)](../deep_learning/transformer/vision_transformer.md) · [Residual blocks / ResNet building blocks](../deep_learning/residual_block/README.md) · [Transfer learning (PyTorch)](transfer_learning_tutorial.ipynb)

## Short rule of thumb

* **Small / medium datasets, limited GPU, need something solid fast** → **ResNet50 (CNN)** is often better.
* **Large data (or strong ImageNet-scale pretraining), enough compute, chase peak accuracy** → **ViT** (or hybrids like Swin) often wins.

## Comparison

| | **ResNet50 (CNN)** | **Vision Transformer (ViT)** |
|--|--------------------|------------------------------|
| **Inductive bias** | Strong (locality, translation equivariance via conv) | Weak (learns relations from data) |
| **Data hunger** | Works well with less data | Needs more data **or** good pretraining |
| **Compute / VRAM** | Usually cheaper at fixed resolution | Attention is heavier, especially high-res |
| **Accuracy (big data)** | Strong baseline | Often higher with large pretrain + fine-tune |
| **Robustness / maturity** | Mature, well understood | Strong with scale; different failure modes |
| **Ecosystem** | Everywhere (detection backbones, mobile, deployment) | Great in modern stacks; heavier on edge |

CNNs bake in “nearby pixels matter.” ViTs treat image patches more like tokens and need scale (or distillation / hierarchical designs) to match that bias.

## Practical picks

1. **Default baseline:** ResNet50 (or ConvNeXt if you want a modern CNN).
2. **Best accuracy with pretrained weights and a decent GPU:** ViT-B/L, DeiT, or **Swin** (hierarchical ViT — often a sweet spot for classification and dense CV).
3. **Tiny data (hundreds–few thousand images):** CNN or heavily pretrained ViT + careful fine-tune; scratch ViT is usually worse.
4. **Mobile / latency-critical:** CNN or efficient hybrids (MobileNet, EfficientNet, TinyViT) — plain ViT-B is rarely ideal.

## Bottom line

ResNet50 is the safer classical choice; ViT is often better when you have pretrained weights and enough data/compute. For many real projects, a **pretrained ViT/Swin fine-tune** beats ResNet50, but a **pretrained ResNet50** can beat a **ViT trained from scratch** on a small set.

## References

* K. He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (ResNet, 2015).
* A. Dosovitskiy et al., [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (ViT, 2020).
* [google-research/vision_transformer](https://github.com/google-research/vision_transformer)
* [Vision Transformer notes](../deep_learning/transformer/vision_transformer.md)
