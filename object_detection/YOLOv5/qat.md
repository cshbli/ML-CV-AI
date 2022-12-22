# YOLOv5 QAT

## 1. Retrain the YOLOv5

### 1.0 Retrain with torch1.9.1 is OK

```
pip install torchvision==0.10.1
pip install -r requirements.txt
```

### 1.1 Change default activation to ReLU

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

- Once we changed the default_act to ReLU, we can't use auto batch size anymore. 
- We need specifiy the `batch-size`

### 1.2 Retrain on the COCO

- Make sure to change learning rate, otherwise it will long time to converge.
- Also we can change the default `batch-size` from 16 to 64

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

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       7/49      10.2G    0.03988    0.06046     0.0161        195        640: 100%|██████████| 1849/1849 [51:41<00:00,  1.68s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:26<00:00,  2.15s/it]
                   all       5000      36335      0.701      0.563      0.611      0.423

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       8/49      10.2G    0.03989    0.06048    0.01611        235        640: 100%|██████████| 1849/1849 [51:39<00:00,  1.68s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.15s/it]
                   all       5000      36335      0.705      0.564      0.613      0.424

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       9/49      10.2G    0.03974    0.06024    0.01594        197        640: 100%|██████████| 1849/1849 [51:41<00:00,  1.68s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:26<00:00,  2.15s/it]
                   all       5000      36335      0.705      0.565      0.615      0.426

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      10/49      10.2G    0.03971    0.06037    0.01588        168        640: 100%|██████████| 1849/1849 [51:38<00:00,  1.68s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.13s/it]
                   all       5000      36335      0.707      0.565      0.616      0.426

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      11/49      10.2G    0.03967    0.06023    0.01582        213        640: 100%|██████████| 1849/1849 [51:36<00:00,  1.67s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.13s/it]
                   all       5000      36335      0.706      0.567      0.617      0.427

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      12/49      10.2G    0.03958    0.06006     0.0157        171        640: 100%|██████████| 1849/1849 [51:35<00:00,  1.67s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.13s/it]
                   all       5000      36335      0.706      0.568      0.617      0.428

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      13/49      10.2G    0.03954    0.05978    0.01563        198        640: 100%|██████████| 1849/1849 [51:32<00:00,  1.67s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.13s/it]
                   all       5000      36335      0.709      0.567      0.617      0.428

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      14/49      10.2G    0.03948    0.05968    0.01557        240        640: 100%|██████████| 1849/1849 [51:30<00:00,  1.67s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.13s/it]
                   all       5000      36335      0.708      0.568      0.618      0.429

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      15/49      10.2G    0.03943     0.0597    0.01547        140        640: 100%|██████████| 1849/1849 [51:28<00:00,  1.67s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.13s/it]
                   all       5000      36335      0.709       0.57      0.618      0.429

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      16/49      10.2G    0.03936    0.05952    0.01537        145        640: 100%|██████████| 1849/1849 [51:25<00:00,  1.67s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.13s/it]
                   all       5000      36335      0.713      0.568      0.618       0.43

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      17/49      10.2G     0.0393    0.05943    0.01529        193        640: 100%|██████████| 1849/1849 [51:24<00:00,  1.67s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.13s/it]
                   all       5000      36335      0.705      0.571      0.619       0.43

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      18/49      10.2G    0.03926    0.05949    0.01524        192        640: 100%|██████████| 1849/1849 [51:23<00:00,  1.67s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:24<00:00,  2.12s/it]
                   all       5000      36335      0.707      0.569      0.619       0.43

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      19/49      10.2G    0.03922    0.05922    0.01519        162        640: 100%|██████████| 1849/1849 [51:23<00:00,  1.67s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.13s/it]
                   all       5000      36335      0.713      0.567       0.62       0.43

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      20/49      10.2G    0.03911    0.05934    0.01513        228        640: 100%|██████████| 1849/1849 [51:33<00:00,  1.67s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.13s/it]
                   all       5000      36335      0.707      0.569      0.619      0.431

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      21/49      10.2G    0.03903    0.05904      0.015        160        640: 100%|██████████| 1849/1849 [51:35<00:00,  1.67s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.13s/it]
                   all       5000      36335      0.701      0.572      0.619      0.431

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      22/49      10.2G    0.03905    0.05885    0.01492        156        640: 100%|██████████| 1849/1849 [51:35<00:00,  1.67s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.13s/it]
                   all       5000      36335      0.706      0.569      0.619      0.431

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      23/49      10.2G    0.03893    0.05877    0.01485        170        640: 100%|██████████| 1849/1849 [51:36<00:00,  1.67s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.13s/it]
                   all       5000      36335       0.71      0.568      0.619      0.431
```
