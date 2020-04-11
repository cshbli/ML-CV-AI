# Visualize Tensorflow graph with Tensorboard
## Frozen graph .pb file
```
import tensorflow as tf
from tensorflow.summary import FileWriter
from tensorflow.core.framework import graph_pb2

sess = tf.Session()
with tf.io.gfile.GFile("your-frozen-graph-file.pb", "rb") as f:
    graph_def = graph_pb2.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def)

FileWriter("__tb", sess.graph)    
print("Model Imported. Visualize by running: "
      "tensorboard --logdir={}".format("__tb"))
```      

## Checkpoint .meta file
The `.meta` file contains information about the different node in the tensorflow graph. Checkpoint meta file contains a serialized `MetaGraphDef` protocol buffer. The `MetaGraphDef` is designed as a serialization format that includes all of the information required to restore a training or inference process (including the `GraphDef` that describes the dataflow, and additional annotations that describe the variables, input pipelines, and other relevant information). For example, the `MetaGraphDef` is used by TensorFlow Serving to start an inference service based on your trained model. 

Assuming that you still have the Python code for your model, you do not need the `MetaGraphDef` to restore the model, because you can reconstruct all of the information in the `MetaGraphDef` by re-executing the Python code that builds the model. To restore from a checkpoint, you only need the checkpoint files that contain the trained weights, which are written periodically to the same directory. 

The values of the different variables in the graph at that moment are stored separately in the checkpoint folder in `checkpoint.data-xxxx-of-xxxx` file.

There is no concept of an input or output node in the normal checkpoint process, as opposed to the case of a frozen model. Freezing a model outputs a subset of the whole tensorflow graph. This subset of the main graph has only those nodes present on which the output node is dependent on. Because freezing a model is done for serving purposes, it converts the tensorflow variables to constants, eliminating the need for storing additional information like gradients of the different variables at each step.

You can restore your graph from the `.meta` file and visualize it in tensorboard.

```
import tensorflow as tf
from tensorflow.summary import FileWriter

sess = tf.Session()
tf.train.import_meta_graph("your-meta-graph-file.meta")
FileWriter("__tb", sess.graph)
```

This will create a `__tb` folder in your current directory and you can then view the graph by issuing the following command.

```
tensorboard --logdir __tb
```
