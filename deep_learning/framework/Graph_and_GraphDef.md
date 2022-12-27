# difference between Tensorflow's Graph and GraphDef

`Graph` or `Computional Graph` is the core concept of tensorflow to present computation. When you use tensorflow, you firstly create you own `Graph` and pass the `Graph` to tensorflow. As you may know, tensorflow support many front-end programming languages, like Python, C++, Java and Go and the core language is C++; how do the other languages transform the Graph to C++? They use a tool called `protobuf` which can generate specific language stubs, that's where the `GraphDef` come from. <b>It's a serialized version of `Graph`</b>.

You should read your *pb file using `GraphDef` and bind the `GraphDef` to a (default) `Graph`, then use a session to run the Graph for computation, like the following code:

```
import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename ='PATH_TO_PB.pb'
    with gfile.GFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # set name='', otherwise the default 'import' name scope will be 
        # added to all tensor names
        input_graph = tf.import_graph_def(graph_def, name='')
```        