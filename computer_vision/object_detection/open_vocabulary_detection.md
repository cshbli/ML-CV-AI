# Open-vocabulary / zero-shot object detection (“detect anything”)

Models similar in spirit to [Segment Anything](../Segmentation/segment_anything.md), but for **boxes**: detect objects with **zero-shot / open-vocabulary** prompts — usually **text** (“red fire hydrant”), sometimes image exemplars — without training on that exact closed class set.

See also: [YOLO — is it SOTA?](yolo_is_sota.md) · [Segment Anything](../Segmentation/segment_anything.md) · [Object detection overview](../../object_detection/README.md)

## SAM vs zero-shot detectors

| | **SAM / SAM 2** | **Open-vocab detectors** |
|--|-----------------|---------------------------|
| Output | Masks | Usually **boxes** (masks if chained with SAM) |
| Typical prompt | Point / box / mask | **Text** (“zebra”, “scalpel on the tray”) |
| “Zero-shot” means | Segment novel things when prompted spatially | Detect novel **categories** from language |
| Needs class name? | No | Yes (in the prompt), unless using image-exemplar variants |

SAM ≈ “segment this region.” Grounding DINO / YOLO-World ≈ “find anything that matches this phrase.”

There is no single “Detect Anything” identical to SAM, but **Grounding DINO, YOLO-World, GLIP, OWL-ViT** are the usual zero-shot detection counterparts.

## Main models

| Model | What you prompt with | Role vs SAM |
|-------|----------------------|-------------|
| **Grounding DINO** | Text (and phrases) | Strong open-vocab **boxes**; often paired with SAM → Grounded-SAM |
| **YOLO-World** | Text | Real-time open-vocab YOLO-style detector |
| **GLIP / GLIPv2** | Text | Phrase grounding + detection |
| **OWL-ViT / OWLv2** | Text (or image exemplars) | Open-vocab detection from Vision Transformers |
| **Detic** | CLIP classifiers / large vocab | Many classes without classic box labels per class |
| **Florence-2, other VLMs** | Text / instructions | Broader vision-language; can do grounding/detection-style tasks |

## Practical stack (very common)

1. **Grounding DINO** or **YOLO-World** → boxes from text  
2. **SAM / SAM 2** → masks from those boxes  

That is the usual “detect anything + segment anything” pipeline (e.g. Grounded-SAM).

## What to pick

* **Best quality open-vocab boxes:** start with **Grounding DINO**  
* **Need speed:** **YOLO-World**  
* **Boxes + masks together:** Grounded-SAM-style pipeline  
* **Closed-set custom classes you will train:** normal YOLO still wins ([version guide](yolo_is_sota.md#yolov5-v7-v8--v11--what-differs-what-is-outdated)); open-vocab is for unknown/rare classes and text queries  

## References

* [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
* [YOLO-World](https://github.com/AILab-CVC/YOLO-World)
* [OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit) · [OWLv2](https://huggingface.co/docs/transformers/model_doc/owlv2)
* [Segment Anything](../Segmentation/segment_anything.md)
