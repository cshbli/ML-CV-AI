# Install Tensorflow

## Installation

- Install OpenCV
```
pip install opencv-python
```

For Tensorflow 1.12, 

```
pip install opencv-python==3.4.10.35
```

- Install Scikit-learn
```
pip install scikit-learn
```

## Verification

### See list of conda environments
```
conda env list
```

### Activate your environment:
```
conda activate your_environment_name
```

### GPU

When using tensorflow2:
``````
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
``````
For tensorflow1, to find out which device is used, you can enable log device placement like this:
```
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
``````
