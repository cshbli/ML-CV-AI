# YOLOv5 QAT

## Retrain the YOLOv5

### 1.1 Retrain with default settings and with 1 GPU
```
python train.py --data coco.yaml --epochs 1 --weights yolov5m.pt --cfg yolov5m.yaml --batch-size 1
```

The mAP result is not good. 

```
Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 2500/2500 [01:19<00:00,
                   all       5000      36335      0.672      0.548      0.589      0.409
                person       5000      10777      0.866      0.651      0.781      0.536
               bicycle       5000        314      0.703      0.484      0.566      0.324
                   car       5000       1918       0.76      0.583      0.666      0.432
            motorcycle       5000        367      0.768      0.632      0.724      0.453
              airplane       5000        143       0.82      0.811      0.859      0.622
                   bus       5000        283       0.78      0.746      0.801      0.661
                 train       5000        190      0.863      0.798      0.863      0.633
                 truck       5000        414       0.65      0.448      0.531      0.361
                  boat       5000        424      0.681       0.38      0.495      0.254
         traffic light       5000        634      0.683      0.517      0.553       0.28
          fire hydrant       5000        101      0.835      0.743      0.831      0.661
             stop sign       5000         75      0.666      0.707      0.724      0.642
         parking meter       5000         60      0.681        0.6      0.633      0.487
                 bench       5000        411      0.585      0.367      0.385      0.244
                  bird       5000        427      0.748      0.431       0.51      0.333
                   cat       5000        202      0.782      0.851      0.837      0.599
                   dog       5000        218      0.783       0.67      0.762      0.605
                 horse       5000        272      0.817      0.765      0.833      0.631
                 sheep       5000        354      0.719      0.695      0.753      0.534
                   cow       5000        372      0.808      0.672      0.777      0.545
              elephant       5000        252      0.781       0.82      0.833      0.631
                  bear       5000         71      0.772      0.887      0.877      0.728
                 zebra       5000        266      0.846      0.831      0.907      0.681
               giraffe       5000        232      0.852      0.816      0.887       0.68
              backpack       5000        371       0.37       0.27      0.233      0.124
              umbrella       5000        407      0.691      0.582      0.614      0.393
               handbag       5000        540      0.521      0.248      0.283      0.161
                   tie       5000        252       0.67      0.467      0.534       0.33
              suitcase       5000        299      0.605      0.562      0.601      0.411
               frisbee       5000        115      0.638      0.826      0.815      0.635
                  skis       5000        241      0.669      0.387      0.466      0.236
             snowboard       5000         69      0.514      0.449       0.45      0.325
           sports ball       5000        260      0.729       0.57      0.618      0.436
                  kite       5000        327       0.62      0.578      0.624       0.43
          baseball bat       5000        145       0.76      0.607      0.633      0.355
        baseball glove       5000        148      0.712      0.554      0.625      0.364
            skateboard       5000        179      0.829      0.743      0.753       0.53
             surfboard       5000        267      0.665      0.524       0.59      0.362
         tennis racket       5000        225      0.735      0.753      0.792      0.511
                bottle       5000       1013      0.675      0.489      0.565      0.378
            wine glass       5000        341      0.699      0.534      0.595      0.382
                   cup       5000        895      0.666      0.539      0.591      0.425
                  fork       5000        215      0.569      0.442      0.491      0.328
                 knife       5000        325      0.562       0.28      0.313      0.186
                 spoon       5000        253      0.509      0.281      0.309      0.195
                  bowl       5000        623      0.673       0.46      0.539      0.388
                banana       5000        370      0.607      0.262      0.355      0.209
                 apple       5000        236      0.449      0.254      0.265      0.174
              sandwich       5000        177      0.627      0.441      0.491       0.35
                orange       5000        285      0.453      0.344      0.343      0.251
              broccoli       5000        312      0.547      0.353      0.364      0.205
                carrot       5000        365      0.481      0.252      0.287      0.176
               hot dog       5000        125      0.523      0.456      0.472      0.346
                 pizza       5000        284      0.721      0.676      0.703        0.5
                 donut       5000        328      0.661      0.552      0.611       0.47
                  cake       5000        310      0.661      0.513      0.568      0.361
                 chair       5000       1771      0.684      0.421      0.509      0.321
                 couch       5000        261      0.679      0.521      0.606      0.433
          potted plant       5000        342      0.588      0.488      0.496      0.285
                   bed       5000        163      0.692      0.552      0.578      0.382
          dining table       5000        695       0.54      0.426      0.409       0.26
                toilet       5000        179      0.735      0.788      0.805      0.609
                    tv       5000        288      0.792      0.719      0.783      0.586
                laptop       5000        231        0.7      0.706      0.752      0.601
                 mouse       5000        106      0.741      0.708      0.735      0.559
                remote       5000        283      0.596      0.459      0.483      0.286
              keyboard       5000        153      0.574      0.621      0.632      0.442
            cell phone       5000        262      0.694      0.537      0.568      0.363
             microwave       5000         55      0.539      0.782      0.752      0.587
                  oven       5000        143      0.554      0.524      0.512      0.318
               toaster       5000          9      0.691      0.444      0.495       0.36
                  sink       5000        225      0.683      0.526      0.573      0.375
          refrigerator       5000        126       0.66      0.722      0.704      0.541
                  book       5000       1129      0.471      0.158      0.222      0.107
                 clock       5000        267       0.73      0.704      0.728      0.501
                  vase       5000        274      0.631      0.573      0.566       0.39
              scissors       5000         36      0.588      0.333      0.371      0.281
            teddy bear       5000        190      0.726      0.616      0.662      0.451
            hair drier       5000         11          1          0    0.00575    0.00409
            toothbrush       5000         57      0.418      0.333      0.315      0.206

Evaluating pycocotools mAP... saving runs/train/exp3/_predictions.json...
loading annotations into memory...
Done (t=0.68s)
creating index...
index created!
Loading and preparing results...
DONE (t=4.60s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=50.10s).
Accumulating evaluation results...
DONE (t=14.23s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.411
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.593
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.449
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.251
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.476
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.519
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.337
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.562
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.617
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.431
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.680
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.754
Results saved to runs/train/exp3
```
### 1.2 Retrain with multiple GPUs (on desktop server)

```
python -m torch.distributed.run --nproc_per_node 2 train.py --batch 64 --data coco.yaml --epochs 1 --weights yolov5m --cfg yolov5m.yaml --device 0,1,2,3
```

Failed after 6%

```
python -m torch.distributed.run --nproc_per_node 2 train.py --batch 64 --data coco.yaml --epochs 1 --weights yolov5m --cfg yolov5m.yaml --device 0,1
```

Failed after 8%

### 1.3 Retrain with COCO128 

```
python train.py --data coco128.yaml --epochs 1 --weights yolov5m.pt --cfg yolov5m.yaml --batch-size -1
```

Result will be available very quick.

```
Starting training for 1 epochs...

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        0/0      3.47G    0.04094    0.06227    0.02574         30        640: 100%|██████████| 6/6 [00:15<00:00,  2.62s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 3/3 [00:02<00:00,  1.25it/s]
                   all        128        929      0.754      0.662      0.772      0.555

1 epochs completed in 0.005 hours.
Optimizer stripped from runs/train/exp5/weights/last.pt, 42.8MB
Optimizer stripped from runs/train/exp5/weights/best.pt, 42.8MB

Validating runs/train/exp5/weights/best.pt...
Fusing layers... 
YOLOv5m summary: 212 layers, 21172173 parameters, 0 gradients, 48.9 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 3/3 [00:07<00:00,  2.34s/it]
                   all        128        929      0.755      0.661      0.772      0.555
                person        128        254      0.876      0.728      0.835      0.593
               bicycle        128          6      0.605      0.333      0.523       0.38
                   car        128         46      0.823      0.413      0.575      0.285
            motorcycle        128          5      0.816      0.894      0.928      0.764
              airplane        128          6          1      0.948      0.995      0.857
                   bus        128          7      0.711      0.706      0.731      0.624
                 train        128          3      0.698          1      0.995      0.703
                 truck        128         12      0.649      0.417      0.533      0.329
                  boat        128          6      0.961        0.5      0.689      0.403
         traffic light        128         14      0.935      0.286      0.495      0.246
             stop sign        128          2      0.831          1      0.995      0.821
                 bench        128          9      0.742      0.556       0.76       0.32
                  bird        128         16      0.835          1      0.991      0.683
                   cat        128          4      0.897          1      0.995      0.929
                   dog        128          9      0.849      0.889      0.916      0.674
                 horse        128          2       0.75          1      0.995      0.623
              elephant        128         17      0.967      0.882      0.944      0.727
                  bear        128          1      0.673          1      0.995      0.895
                 zebra        128          4      0.882          1      0.995      0.959
               giraffe        128          9      0.731      0.909      0.955      0.758
              backpack        128          6      0.858        0.5      0.614      0.355
              umbrella        128         18      0.833       0.83      0.887      0.561
               handbag        128         19      0.753      0.263      0.425      0.258
                   tie        128          7      0.998      0.857      0.857       0.67
              suitcase        128          4      0.797          1      0.995      0.659
               frisbee        128          5       0.73        0.8        0.8       0.76
                  skis        128          1      0.541          1      0.995      0.697
             snowboard        128          7      0.791      0.857      0.917      0.716
           sports ball        128          6      0.711      0.667      0.773      0.425
                  kite        128         10      0.662      0.394      0.616      0.201
          baseball bat        128          4      0.553       0.75      0.759      0.286
        baseball glove        128          7      0.568      0.429      0.577        0.3
            skateboard        128          5          1       0.79      0.822      0.512
         tennis racket        128          7      0.734      0.571      0.648      0.305
                bottle        128         18      0.565      0.363      0.545      0.362
            wine glass        128         16      0.897      0.547      0.701      0.468
                   cup        128         36      0.777      0.583      0.768      0.528
                  fork        128          6      0.815      0.333      0.581      0.267
                 knife        128         16      0.776       0.65      0.758       0.52
                 spoon        128         22      0.748      0.682      0.744      0.477
                  bowl        128         28      0.828      0.686      0.737      0.578
                banana        128          1      0.479          1      0.995      0.796
              sandwich        128          2          0          0      0.398      0.246
                orange        128          4      0.977        0.5      0.849      0.634
              broccoli        128         11      0.309      0.273      0.277      0.242
                carrot        128         24      0.656      0.458      0.664      0.422
               hot dog        128          2      0.614          1      0.663      0.663
                 pizza        128          5      0.897          1      0.995      0.812
                 donut        128         14      0.714          1      0.915       0.82
                  cake        128          4      0.783          1      0.995      0.847
                 chair        128         35      0.645      0.629      0.642      0.344
                 couch        128          6          1      0.821      0.931      0.599
          potted plant        128         14      0.864      0.571      0.759      0.506
                   bed        128          3          1      0.336      0.995      0.699
          dining table        128         13       0.81      0.332      0.579      0.361
                toilet        128          2      0.913          1      0.995      0.995
                    tv        128          2      0.563          1      0.995      0.896
                laptop        128          3          1          0      0.608      0.417
                 mouse        128          2          1          0      0.828      0.432
                remote        128          8      0.712      0.625       0.68      0.553
            cell phone        128          8      0.604      0.375       0.45      0.284
             microwave        128          3      0.515          1      0.995      0.848
                  oven        128          5      0.277        0.4      0.423      0.336
                  sink        128          6      0.331      0.333      0.406      0.281
          refrigerator        128          5      0.834        0.8      0.822      0.527
                  book        128         29      0.804       0.31      0.445      0.215
                 clock        128          9      0.846          1      0.984      0.772
                  vase        128          2      0.491          1      0.995      0.995
              scissors        128          1          1          0      0.497      0.151
            teddy bear        128         21        0.8      0.571      0.738       0.49
            toothbrush        128          5          1      0.597      0.962      0.724
Results saved to runs/train/exp5
```

## 2. Retrain with torch1.9.1

```
pip install torchvision==0.10.1
pip install -r requirements.txt
```

### 1.3 Retrain with COCO128 

```
python train.py --data coco128.yaml --epochs 1 --weights yolov5m.pt --cfg yolov5m.yaml --batch-size -1
```

- change `lr0` from 0.01 to 0.0001 in `data/hyps/hyp.scratch-low.yaml`
- change `warmup_epochs` from 3.0 to 0.1

```
DetectionModel(
  (model): Sequential(
    (0): Conv(
      (conv): Conv2d(3, 48, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2), bias=False)
      (bn): BatchNorm2d(48, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (1): Conv(
      (conv): Conv2d(48, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (2): C3(
      (cv1): Conv(
        (conv): Conv2d(96, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(48, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(96, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(48, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(48, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(48, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(48, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(48, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (3): Conv(
      (conv): Conv2d(96, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (4): C3(
      (cv1): Conv(
        (conv): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
        (2): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
        (3): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (5): Conv(
      (conv): Conv2d(192, 384, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (6): C3(
      (cv1): Conv(
        (conv): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
        (2): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
        (3): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
        (4): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
        (5): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (7): Conv(
      (conv): Conv2d(384, 768, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(768, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (8): C3(
      (cv1): Conv(
        (conv): Conv2d(768, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(768, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(768, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(768, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (9): SPPF(
      (cv1): Conv(
        (conv): Conv2d(768, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(1536, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(768, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
    )
    (10): Conv(
      (conv): Conv2d(768, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (11): Upsample(scale_factor=2.0, mode=nearest)
    (12): Concat()
    (13): C3(
      (cv1): Conv(
        (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (14): Conv(
      (conv): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (15): Upsample(scale_factor=2.0, mode=nearest)
    (16): Concat()
    (17): C3(
      (cv1): Conv(
        (conv): Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (18): Conv(
      (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (19): Concat()
    (20): C3(
      (cv1): Conv(
        (conv): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (21): Conv(
      (conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (22): Concat()
    (23): C3(
      (cv1): Conv(
        (conv): Conv2d(768, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(768, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(768, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(768, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (24): Detect(
      (m): ModuleList(
        (0): Conv2d(192, 255, kernel_size=(1, 1), stride=(1, 1))
        (1): Conv2d(384, 255, kernel_size=(1, 1), stride=(1, 1))
        (2): Conv2d(768, 255, kernel_size=(1, 1), stride=(1, 1))
      )
    )
  )
)
```

### Change default activation to ReLU

```
class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    # default_act = nn.SiLU()  # default activation
    default_act = nn.ReLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
```

- Once we changed the default_act to ReLU, we can't use auto batch size anymore. It will have errors such as:

```
'ReLU' object has no attribute 'total_ops'
'ReLU' object has no attribute 'total_ops'
'ReLU' object has no attribute 'total_ops'
'ReLU' object has no attribute 'total_ops'
'ReLU' object has no attribute 'total_ops'
Traceback (most recent call last):
  File "train.py", line 633, in <module>
    main(opt)
  File "train.py", line 527, in main
    train(opt.hyp, opt, device, callbacks)
  File "train.py", line 149, in train
    batch_size = check_train_batch_size(model, imgsz, amp)
  File "/home/hongbing/Projects/yolov5/utils/autobatch.py", line 18, in check_train_batch_size
    return autobatch(deepcopy(model).train(), imgsz)  # compute optimal batch size
  File "/home/hongbing/Projects/yolov5/utils/autobatch.py", line 60, in autobatch
    p = np.polyfit(batch_sizes[:len(y)], y, deg=1)  # first degree polynomial fit
  File "<__array_function__ internals>", line 180, in polyfit
  File "/home/hongbing/venv/torch1.9.1/lib/python3.8/site-packages/numpy/lib/polynomial.py", line 638, in polyfit
    raise TypeError("expected non-empty vector for x")
TypeError: expected non-empty vector for x
```
- So we specifiy the `batch-size`

```
python train.py --data coco128.yaml --epochs 1 --weights yolov5m.pt --cfg yolov5m.yaml --batch-size 2
```

- Export the trained model to ONNX
```
python export.py --weights ./runs/train/exp5/weights/best.pt --include torchscript onnx --opset 13
```

#### Retrain on the whole COCO on the desktop server

```
python train.py --data coco.yaml --epochs 100 --weights yolov5m.pt --cfg yolov5m.yaml --batch-size 32
```

```
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       0/99      3.37G    0.04123    0.06264     0.0184        150        640: 100%|██████████| 3697/3697 [1:05:50<00:00,  1.0
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:25<00:00,  1.08s/it]
                   all       5000      36335      0.643      0.522      0.556      0.369
```

```
python train.py --data coco.yaml --epochs 100 --weights yolov5m.pt --cfg yolov5m.yaml --batch-size 64 --resume
```

```
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       2/99      3.14G     0.0493    0.07015    0.02784        150        640: 100%|██████████| 3697/3697 [1:04:56<00:00,  1.05s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:30<00:00,  1.14s/it]
                   all       5000      36335      0.591      0.461      0.488       0.31

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       3/99      4.56G    0.05232    0.07215    0.03135        172        640: 100%|██████████| 3697/3697 [1:04:44<00:00,  1.05s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:29<00:00,  1.13s/it]
                   all       5000      36335      0.613      0.469        0.5      0.322

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       4/99      4.56G    0.05092    0.07092    0.02933        243        640: 100%|██████████| 3697/3697 [1:04:43<00:00,  1.05s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:28<00:00,  1.12s/it]
                   all       5000      36335      0.619      0.494      0.524      0.341

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       5/99      4.56G    0.04971    0.07025    0.02766        185        640: 100%|██████████| 3697/3697 [1:04:41<00:00,  1.05s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:27<00:00,  1.11s/it]
                   all       5000      36335       0.64      0.499       0.54      0.356

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       6/99      4.56G    0.04909    0.06974    0.02687        229        640: 100%|██████████| 3697/3697 [1:04:39<00:00,  1.05s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:27<00:00,  1.11s/it]
                   all       5000      36335      0.644      0.508      0.548      0.364

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       7/99      4.56G    0.04875    0.06928    0.02639        190        640: 100%|██████████| 3697/3697 [1:04:39<00:00,  1.05s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:27<00:00,  1.11s/it]
                   all       5000      36335      0.643      0.514      0.554      0.369

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       8/99      4.56G    0.04847    0.06912      0.026        168        640: 100%|██████████| 3697/3697 [1:04:47<00:00,  1.05s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:27<00:00,  1.11s/it]
                   all       5000      36335      0.649      0.517      0.559      0.372

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       9/99      4.56G    0.04829    0.06889     0.0256        195        640: 100%|██████████| 3697/3697 [1:04:52<00:00,  1.05s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:27<00:00,  1.10s/it]
                   all       5000      36335      0.651      0.519      0.561      0.375

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      10/99      4.56G     0.0482    0.06896    0.02554        235        640: 100%|██████████| 3697/3697 [1:04:51<00:00,  1.05s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:27<00:00,  1.10s/it]
                   all       5000      36335      0.654      0.517      0.563      0.377

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      11/99      4.56G    0.04797     0.0686    0.02526        197        640: 100%|██████████| 3697/3697 [1:04:52<00:00,  1.05s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:27<00:00,  1.10s/it]
                   all       5000      36335      0.653      0.521      0.566      0.379

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      12/99      4.56G     0.0479    0.06872    0.02504        168        640: 100%|██████████| 3697/3697 [1:04:52<00:00,  1.05s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:27<00:00,  1.10s/it]
                   all       5000      36335      0.649      0.525      0.567       0.38

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      13/99      4.56G    0.04781    0.06868    0.02499        213        640: 100%|██████████| 3697/3697 [1:04:50<00:00,  1.05s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:27<00:00,  1.10s/it]
                   all       5000      36335      0.652      0.523      0.569      0.381

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      14/99      4.56G    0.04772    0.06848    0.02489        171        640: 100%|██████████| 3697/3697 [1:04:53<00:00,  1.05s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:26<00:00,  1.10s/it]
                   all       5000      36335      0.641      0.532       0.57      0.382

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      15/99      4.56G    0.04763    0.06833    0.02477        198        640: 100%|██████████| 3697/3697 [1:04:53<00:00,  1.05s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:26<00:00,  1.10s/it]
                   all       5000      36335      0.651      0.527       0.57      0.383

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      16/99      4.56G    0.04758    0.06822    0.02473        240        640: 100%|██████████| 3697/3697 [1:04:52<00:00,  1.05s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:26<00:00,  1.10s/it]
                   all       5000      36335      0.642      0.533      0.572      0.384

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      17/99      4.56G    0.04752    0.06833    0.02455        140        640: 100%|██████████| 3697/3697 [1:04:51<00:00,  1.05s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:26<00:00,  1.10s/it]
                   all       5000      36335      0.653      0.527      0.572      0.384

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      32/99      5.25G    0.04661    0.06751    0.02325        152        640: 100%|██████████| 3697/3697 [1:04:45<00:00,  1.05s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:26<00:00,  1.09s/it]
                   all       5000      36335      0.657      0.543      0.584      0.394

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      33/99      5.25G    0.04647    0.06741    0.02317        231        640: 100%|██████████| 3697/3697 [1:04:46<00:00,  1.05s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:26<00:00,  1.09s/it]
                   all       5000      36335      0.659      0.543      0.584      0.394

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      34/99      5.25G    0.04641    0.06719    0.02308        167        640: 100%|██████████| 3697/3697 [1:04:51<00:00,  1.05s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:26<00:00,  1.09s/it]
                   all       5000      36335       0.66      0.543      0.585      0.394                   
```

```
python train.py --data coco.yaml --epochs 50 --weights yolov5m.pt --hyp data/hyps/hyp.m-relu-tune.yaml --batch-size 64
```

```
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       0/49      6.16G    0.04115    0.06202    0.01698        150        640: 100%|██████████| 1849/1849 [51:50<00:00,  1.68s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:26<00:00,  2.17s/it]
                   all       5000      36335      0.701      0.557      0.609      0.416

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       1/49      8.82G    0.03956    0.06001    0.01587        172        640: 100%|██████████| 1849/1849 [51:39<00:00,  1.68s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.14s/it]
                   all       5000      36335      0.695      0.561      0.606      0.416

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       2/49      8.82G    0.03969    0.05995    0.01596        243        640: 100%|██████████| 1849/1849 [51:39<00:00,  1.68s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.15s/it]
                   all       5000      36335      0.694      0.553        0.6      0.409

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       3/49      8.82G    0.04005    0.06077    0.01636        185        640: 100%|██████████| 1849/1849 [51:37<00:00,  1.68s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.14s/it]
                   all       5000      36335      0.698      0.551      0.601      0.411

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       4/49      8.82G    0.04003     0.0608    0.01642        229        640: 100%|██████████| 1849/1849 [51:38<00:00,  1.68s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.14s/it]
                   all       5000      36335      0.705      0.551      0.604      0.417

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       5/49      10.2G    0.04002    0.06061    0.01638        190        640: 100%|██████████| 1849/1849 [51:38<00:00,  1.68s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:26<00:00,  2.15s/it]
                   all       5000      36335      0.698      0.556      0.604      0.417

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       6/49      10.2G    0.03997    0.06054     0.0163        168        640: 100%|██████████| 1849/1849 [51:39<00:00,  1.68s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.14s/it]
                   all       5000      36335      0.692      0.564      0.607      0.419
```
