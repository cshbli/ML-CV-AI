# YOLO — is it the current SOTA for object detection?

**Short answer:** YOLO is the practical SOTA for **real-time / deployed** object detection — not always the absolute top of the **COCO accuracy** leaderboard.

This repo already covers concrete YOLO stacks under [`object_detection/`](../../object_detection/README.md) (e.g. [YOLOv5](../../object_detection/YOLOv5/README.md), [YOLOv7](../../object_detection/YOLOv7/README.md)). This note is about **where YOLO sits vs “SOTA”**.

See also: [Object detection overview](../../object_detection/README.md) · [mAP](../../object_detection/concepts/mAP.md) · [Computer Vision overview](../README.md)

## Nuance: “SOTA” depends on the goal

| Goal | Is YOLO “the” SOTA? | What leads instead |
|------|---------------------|--------------------|
| **Speed + accuracy tradeoff** (edge, video, products) | **Yes — YOLO family** is the default | Ultralytics YOLO (v8/v11…), YOLOv9/v10, YOLO-World (open-vocab) |
| **Highest COCO box AP** (offline, big models) | **Often no** | Heavy Transformer / hybrid detectors (e.g. Co-DETR-class, large DINO/DETA-style, big backbones) |
| **Open-vocabulary** (“detect an arbitrary phrase”) | YOLO-World / similar help; not classic YOLO alone | Grounding DINO, OWL-ViT, GLIP, YOLO-World |
| **Easy train on custom data** | **Yes — YOLO** | Same ecosystem (Ultralytics, etc.) |

Classic YOLO-style models are **one-stage, real-time** detectors. They win on latency and engineering. Pure accuracy races often go to slower, heavier models you would not ship on a phone or at 30–60 FPS. Version numbers (v3→v11) are **not** one continuous product — see [below](#yolov5-v7-v8--v11--what-differs-what-is-outdated).

## Practical takeaway

* Building a detector for a **custom dataset** and care about FPS → start with **current Ultralytics YOLO** (or YOLOv9/v10/v11 variants).
* Chasing **max AP on a benchmark** with no speed limit → look beyond YOLO at large DETR-style / hybrid SOTA papers.
* Need **text-driven** detection → open-vocab models (Grounding DINO, YOLO-World), not vanilla closed-set YOLO alone.

So: YOLO is the current **go-to SOTA for real-time detection**; it is not uniquely “the” SOTA for every detection leaderboard.

## YOLOv5, v7, v8, … v11 — what differs? What is outdated?

For a **new** project, treat **v3–v4 as outdated**, **v5/v7 as legacy-but-fine if you already use them**, and start with **Ultralytics YOLO11 (or v8)** unless you have a specific reason for v9/v10/YOLO-World.

### Not one linear “v3→v11” product

Version numbers come from **different groups**:

| Line | Who | Notes |
|------|-----|--------|
| v3 | Redmon | Darknet classic |
| v4 | Bochkovskiy et al. | CSP, “bag of freebies” |
| **v5, v8, 11** | **Ultralytics** | Same product lineage; best DX (train/export/deploy) |
| v6 | Meituan | Industrial real-time |
| v7 | WongKinYiu et al. | Strong 2022 real-time paper model |
| v9, v10 | Academic releases | New training/arch ideas; separate from Ultralytics numbering |
| YOLO-World | Open-vocab | Text prompts; different job than closed-set YOLO |

So “v8 vs v7” is partly **ecosystem** (Ultralytics vs other repos), not only accuracy.

### What changed (practical differences)

| Version | Era | What mattered | Status for new work |
|---------|-----|---------------|---------------------|
| **v3** | 2018 | Multi-scale heads, Darknet | **Outdated** |
| **v4** | 2020 | CSP, stronger training tricks | **Outdated** for greenfield |
| **v5** | 2020+ | PyTorch, easy train/export, huge adoption | **Legacy**; still everywhere in production |
| **v6** | 2022 | Meituan industrial stack | Niche unless you already use it |
| **v7** | 2022 | Very strong real-time accuracy then | **Legacy**; solid, fewer new features than Ultralytics |
| **v8** | 2023 | Anchor-free, detect/seg/pose/OBB/cls one API | **Still excellent default** |
| **v9** | 2024 | GELAN + PGI (trainable info bottleneck story) | Worth trying; smaller ecosystem than Ultralytics |
| **v10** | 2024 | **NMS-free** end-to-end | Interesting latency path; evaluate carefully |
| **11** | 2024 | Ultralytics successor to v8 | **Current Ultralytics default** to start with |

Theme over time: better backbones/necks, better training recipes, fewer anchors → anchor-free, then tasks beyond boxes (seg/pose), then open-vocab and NMS-free variants.

### What is basically outdated?

* **For new training:** **YOLOv3, YOLOv4** — yes, outdated.
* **YOLOv5:** not “wrong,” but **Ultralytics considers it superseded** by v8/11; use only if you inherit a v5 stack.
* **YOLOv7:** still respectable; for a **new** repo, v8/11 is usually easier long-term (docs, export, maintenance).
* **YOLOv8:** not outdated; still a fine choice (many tutorials/deploy guides).
* **v9 / v10 / 11:** current generation; pick by need (ecosystem vs paper features vs NMS-free).

### What you should use

1. **Closed-set custom detection, care about shipping** → **YOLO11** or **YOLOv8** (Ultralytics).
2. **Already on v5/v7 and it works** → no need to rewrite; upgrade when you need new tasks/export paths.
3. **Want open-vocab (“find a red screwdriver”)** → **YOLO-World** / Grounding DINO — not plain v5–v8.
4. **Benchmark tourism** → compare latest Ultralytics + v9/v10 papers on *your* data/latency, not COCO blog charts alone.

**Bottom line:** v3–v4 are historical. v5/v7 are older production workhorses. **v8 and 11 are the practical current line**; v9/v10 are newer alternatives with specific ideas, not mandatory upgrades for everyone.

## YOLO vs slower high-AP detectors

| | **YOLO family** | **Heavy DETR / hybrid SOTA** |
|--|-----------------|------------------------------|
| Strength | Latency, deployment, custom-data DX | Peak box AP on large benchmarks |
| Typical use | Products, video, edge | Research leaderboards, offline batch |
| Training on your data | Usually straightforward | Heavier compute / more fragile setups |

## References

* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
* Repo notes: [YOLOv5](../../object_detection/YOLOv5/README.md) · [YOLOv7](../../object_detection/YOLOv7/README.md) · [object detection README](../../object_detection/README.md)
