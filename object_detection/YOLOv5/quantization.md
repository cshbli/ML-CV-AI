# YOLOv5 Quantization

## 1. Export the PyTorch model to ONNX using Ultralytics `export.py` script

```
python export.py --weights yolov5s.pt --include onnx --opset 13
```

```
python export.py --weights yolov5s.pt --include onnx --opset 13
```

```
export: data=data/coco128.yaml, weights=['yolov5m.pt'], imgsz=[640, 640], batch_size=1, device=cpu, half=False, inplace=False, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=13, verbose=False, workspace=4, nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45, conf_thres=0.25, include=['onnx']
YOLOv5 🚀 v7.0-24-gf8539a68 Python-3.8.10 torch-1.13.0+cu117 CPU

Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt to yolov5m.pt...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40.8M/40.8M [00:02<00:00, 19.8MB/s]

Fusing layers... 
YOLOv5m summary: 290 layers, 21172173 parameters, 0 gradients

PyTorch: starting from yolov5m.pt with output shape (1, 25200, 85) (40.8 MB)

ONNX: starting export with onnx 1.13.0...
ONNX: export success ✅ 0.9s, saved as yolov5m.onnx (81.2 MB)

Export complete (4.6s)
Results saved to /home/hongbing/Projects/yolov5
Detect:          python detect.py --weights yolov5m.onnx 
Validate:        python val.py --weights yolov5m.onnx 
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5m.onnx')  
Visualize:       https://netron.app
```

## 2. Qunatize the FP32 model using quantize_static

```
python quantize_yolov5m.py
```

```

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append("..")
from onnxruntime.quantization import quantize_static, QuantType, CalibrationMethod, CalibrationDataReader
import torch
from utils.dataloaders import LoadImages
from utils.general import check_dataset
import numpy as np

def representative_dataset_gen(dataset, ncalib=100):
    # Representative dataset generator for use with converter.representative_dataset, returns a generator of np arrays
    def data_gen():
        for n, (path, img, im0s, vid_cap, string) in enumerate(dataset):
            input = np.transpose(img, [0, 1, 2])
            input = np.expand_dims(input, axis=0).astype(np.float32)
            input /= 255
            yield [input]
    return data_gen

class CalibrationDataGenYOLO(CalibrationDataReader):
    def __init__(self,
        calib_data_gen,
        input_name
    ):
        x_train = calib_data_gen
        self.calib_data = iter([{input_name: np.array(data[0])} for data in x_train()])

    def get_next(self):
        return next(self.calib_data, None)


dataset = LoadImages(check_dataset('./data/coco128.yaml')['train'], img_size=[640, 640], auto=False)
data_generator = representative_dataset_gen(dataset)

data_reader = CalibrationDataGenYOLO(
    calib_data_gen=data_generator,
    input_name='images'
)


model_path = 'yolov5s'
# Quantize the exported model
quantize_static(
    f'{model_path}.onnx',
    f'{model_path}_ort_quant.onnx',
    calibration_data_reader=data_reader,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=True,
    calibrate_method=CalibrationMethod.MinMax
        )
```

## 3. Evaluate the mAP using Ultralytics implementation

```
python val.py --weights yolov5s_ort_quant.onnx --data coco.yaml
```

Result: 
```
val: Scanning /home/hongbing/Projects/datasets/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 [00:00<?, ?it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 5000/5000 [04:07<00:00, 20.21it/s]
                   all       5000      36335          0          0          0          0
Speed: 0.3ms pre-process, 41.4ms inference, 0.2ms NMS per image at shape (1, 3, 640, 640)
```

```
python val.py --weights yolov5m_ort_quant.onnx --data coco.yaml
```

```
val: Scanning /home/hongbing/Projects/datasets/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 [00:00<?, ?it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 5000/5000 [11:09<00:00,  7.47it/s]
                   all       5000      36335          0          0          0          0
Speed: 0.5ms pre-process, 116.6ms inference, 0.3ms NMS per image at shape (1, 3, 640, 640)
```

### 3.1 Excluding to quantize nodes which taking in those large tensors

```
nodes_to_exclude=["/model.24/Mul_1", "/model.24/Mul_3", "/model.24/Mul_5", "/model.24/Mul_7", "/model.24/Mul_9", "/model.24/Mul_11", "/model.24/Concat", "/model.24/Concat_1", "/model.24/Concat_2", "/model.24/Concat_3"],
```

```
quantize_static(
    f'{model_path}.onnx',
    f'{model_path}_ort_quant.onnx',
    calibration_data_reader=data_reader,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    nodes_to_exclude=["/model.24/Mul_1", "/model.24/Mul_3", "/model.24/Mul_5", "/model.24/Mul_7", "/model.24/Mul_9", "/model.24/Mul_11", "/model.24/Concat", "/model.24/Concat_1", "/model.24/Concat_2", "/model.24/Concat_3"],
    per_channel=True,
    reduce_range=True,
    calibrate_method=CalibrationMethod.MinMax
        )
```

Result: 

```
val: Scanning /home/hongbing/Projects/datasets/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 [00:00<?, ?it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 5000/5000 [08:37<00:00,  9.67it/s]
                   all       5000      36335      0.682       0.56      0.605      0.393
Speed: 0.4ms pre-process, 87.8ms inference, 1.4ms NMS per image at shape (1, 3, 640, 640)

Accumulating evaluation results...
DONE (t=10.03s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.397
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.611
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.427
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.202
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.452
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.539
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.321
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.530
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.584
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.397
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.641
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.752
 
```