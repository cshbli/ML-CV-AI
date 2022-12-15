# YOLOv5 QAT

## Retrain the YOLOv5

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
