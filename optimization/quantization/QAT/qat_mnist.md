# Guide on Tensorflow quantization aware training 

## Requirements

tensorflow 1.15.3
python 3

```
source ~/venv/tf1.15.3/bin/activate
```

## Running training
```
python3 quantization_aware_training_mnist.py
```
Output:
- ckpt: checkpoint model

```
Step 1, Minibatch Loss= 74236.1797, Training Accuracy= 0.047
Step 10, Minibatch Loss= 35374.8594, Training Accuracy= 0.273
Step 20, Minibatch Loss= 10354.1523, Training Accuracy= 0.539
Step 30, Minibatch Loss= 8032.5986, Training Accuracy= 0.594
Step 40, Minibatch Loss= 4936.0820, Training Accuracy= 0.766
Step 50, Minibatch Loss= 5372.5059, Training Accuracy= 0.766
Step 60, Minibatch Loss= 2578.8789, Training Accuracy= 0.820
Step 70, Minibatch Loss= 3225.1899, Training Accuracy= 0.836
Step 80, Minibatch Loss= 2085.8237, Training Accuracy= 0.852
Step 90, Minibatch Loss= 2635.9307, Training Accuracy= 0.859
Step 100, Minibatch Loss= 1606.2928, Training Accuracy= 0.922
Step 110, Minibatch Loss= 2259.4971, Training Accuracy= 0.875
Step 120, Minibatch Loss= 1284.1897, Training Accuracy= 0.906
Step 130, Minibatch Loss= 1486.1177, Training Accuracy= 0.898
Step 140, Minibatch Loss= 1557.2573, Training Accuracy= 0.898
Step 150, Minibatch Loss= 577.8543, Training Accuracy= 0.922
Step 160, Minibatch Loss= 1359.2974, Training Accuracy= 0.914
Step 170, Minibatch Loss= 1862.6909, Training Accuracy= 0.891
Step 180, Minibatch Loss= 1168.3719, Training Accuracy= 0.906
Step 190, Minibatch Loss= 893.6849, Training Accuracy= 0.914
Step 200, Minibatch Loss= 1166.1388, Training Accuracy= 0.930
Step 210, Minibatch Loss= 1774.7173, Training Accuracy= 0.898
Step 220, Minibatch Loss= 482.9639, Training Accuracy= 0.953
Step 230, Minibatch Loss= 789.4235, Training Accuracy= 0.938
Step 240, Minibatch Loss= 946.9411, Training Accuracy= 0.945
Step 250, Minibatch Loss= 682.7535, Training Accuracy= 0.945
Step 260, Minibatch Loss= 736.4354, Training Accuracy= 0.914
Step 270, Minibatch Loss= 1196.4012, Training Accuracy= 0.930
Step 280, Minibatch Loss= 522.1609, Training Accuracy= 0.953
Step 290, Minibatch Loss= 586.9261, Training Accuracy= 0.961
Step 300, Minibatch Loss= 744.2021, Training Accuracy= 0.961
Step 310, Minibatch Loss= 480.6609, Training Accuracy= 0.969
Step 320, Minibatch Loss= 490.9370, Training Accuracy= 0.969
Step 330, Minibatch Loss= 234.9702, Training Accuracy= 0.969
Step 340, Minibatch Loss= 861.0356, Training Accuracy= 0.969
Step 350, Minibatch Loss= 955.4731, Training Accuracy= 0.953
Step 360, Minibatch Loss= 502.7508, Training Accuracy= 0.945
Step 370, Minibatch Loss= 798.5339, Training Accuracy= 0.938
Step 380, Minibatch Loss= 798.7742, Training Accuracy= 0.914
Step 390, Minibatch Loss= 977.4786, Training Accuracy= 0.945
Step 400, Minibatch Loss= 1050.9563, Training Accuracy= 0.953
Step 410, Minibatch Loss= 425.0377, Training Accuracy= 0.945
Step 420, Minibatch Loss= 982.8076, Training Accuracy= 0.945
Step 430, Minibatch Loss= 108.7343, Training Accuracy= 0.977
Step 440, Minibatch Loss= 99.2743, Training Accuracy= 0.984
Step 450, Minibatch Loss= 444.4600, Training Accuracy= 0.945
Step 460, Minibatch Loss= 1271.1218, Training Accuracy= 0.914
Step 470, Minibatch Loss= 919.8480, Training Accuracy= 0.945
Step 480, Minibatch Loss= 432.7097, Training Accuracy= 0.961
Step 490, Minibatch Loss= 102.3655, Training Accuracy= 0.984
Step 500, Minibatch Loss= 495.9827, Training Accuracy= 0.961
Optimization Finished!
Testing Accuracy: 0.98828125
```

## Export the frozen graph
```
python3 quantization_aware_training_mnist.py --mode export
```
Output:
- frozen_quant_graph.pb


