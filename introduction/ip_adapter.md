# IP-Adapter / style reference

**IP-Adapter** (Image Prompt Adapter) is a small add-on for Stable Diffusion that conditions generation with a **reference image**, not only text. **Style reference** usually means using IP-Adapter (or similar tools) mainly to copy **look** — palette, brushwork, aesthetic — from that image.

See also: [Stable Diffusion](stable_diffusion_model.md) · [LoRA](lora.md) · [ControlNet](controlnet.md)

## Idea

* **Text encoder** → *what to do* (e.g. “a cat in watercolor”)
* **IP-Adapter** → *look like this* (feed a photo/painting; embed that image and inject it into the U-Net, usually via extra cross-attention)
* The base SD weights stay frozen; you load IP-Adapter weights on top (lightweight and swappable, similar in spirit to [LoRA](lora.md))

So the reference acts as an **image prompt**.

## Style reference vs identity / content

| Use | What you want from the reference | Typical setup |
|-----|----------------------------------|---------------|
| **Style reference** | Palette, medium, aesthetic | Moderate IP-Adapter weight; text (and maybe ControlNet) set subject/layout |
| **Stronger image prompt** | Likeness of a person/object/scene | Higher image-prompt weight; depends on IP-Adapter variant and settings |

“Style reference” in UIs (A1111, ComfyUI, etc.) is often IP-Adapter, InstantStyle, Style-Aligned, or a similar image-conditioning path — same problem family: **condition on an image without training a new full model**.

## Vs LoRA and ControlNet

| | **IP-Adapter / style ref** | **[LoRA](lora.md)** | **[ControlNet](controlnet.md)** |
|--|---------------------------|----------|----------------|
| Extra input | Reference **image** | No (trained weights) | Structure **map** |
| Training at use time | Usually **none** | Train on a style/subject set | Use a pretrained ControlNet (or train one) |
| Best for | One-shot style / likeness from examples | Reusable style/character adapter | Pose, edges, depth, layout |
| CycleGAN-like unpaired style? | **Yes**, often (img2img + reference) | **Yes** (train LoRA + img2img) | **No** (structure, not domain) |
| Pix2Pix-like structure? | Weak alone | No | **Yes** |

**Short version:** IP-Adapter = “prompt with an image.” Style reference = use that mainly to copy **look**, without paired data or training a LoRA first.

## When to pick which

| Goal | Prefer |
|------|--------|
| One reference image → match its style now | **IP-Adapter / style reference** |
| Reusable style/character you will use often | **[LoRA](lora.md)** (+ img2img if translating a photo) |
| Lock pose / edges / depth | **[ControlNet](controlnet.md)** |
| Unpaired domain/style shift (CycleGAN-like) | LoRA + img2img, **or** IP-Adapter + img2img |
| Paired structure → image (Pix2Pix-like) | ControlNet |

Combining is common: IP-Adapter or LoRA for look, ControlNet for geometry, text for subject.

## Practical tips

* Start with moderate image strength; too high copies composition/identity and fights the prompt.
* For style-only transfer, prefer style-oriented IP-Adapter variants / InstantStyle-style setups when available.
* Match IP-Adapter weights to the **same base** (SD 1.5 vs SDXL).
* For a source photo you want to restyle, use **img2img** (or light denoise) so content stays closer to the input.

## References

* Hu Ye et al., [IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models](https://arxiv.org/abs/2308.06721) (2023).
* [Stable Diffusion overview](stable_diffusion_model.md)
* [LoRA](lora.md) (incl. [CycleGAN comparison](lora.md#compared-to-cyclegan))
* [ControlNet](controlnet.md) (incl. [Pix2Pix comparison](controlnet.md#compared-to-pix2pix-gan))
