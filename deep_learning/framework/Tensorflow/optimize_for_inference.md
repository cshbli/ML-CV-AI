# Tensorflow Model Optimize for Inference

The `optimize_for_inference` module takes a `frozen binary GraphDef` file as input and outputs the optimized Graph Def file which you can use for inference. And to get the frozen binary GraphDef file you need to use the module freeze_graph which takes a GraphDef proto, a SaverDef proto and a set of variables stored in a checkpoint file. The steps to achieve that is given below:

1. Saving tensorflow graph

```
 # make and save a simple graph
 G = tf.Graph()
 with G.as_default():
   x = tf.placeholder(dtype=tf.float32, shape=(), name="x")
   a = tf.Variable(5.0, name="a")
   y = tf.add(a, x, name="y")
   saver = tf.train.Saver()

with tf.Session(graph=G) as sess:
   sess.run(tf.global_variables_initializer())
   out = sess.run(fetches=[y], feed_dict={x: 1.0})

  # Save GraphDef
  tf.train.write_graph(sess.graph_def,'.','graph.pb')
  # Save checkpoint
  saver.save(sess=sess, save_path="test_model")
```  

2. Freeze graph

```
python -m tensorflow.python.tools.freeze_graph \
  --input_graph graph.pb \
  --input_checkpoint test_model \
  --output_graph graph_frozen.pb \
  --output_node_names=y
```

3. Optimize for inference

```
python -m tensorflow.python.tools.optimize_for_inference \
--input graph_frozen.pb \
--output graph_optimized.pb \
--input_names=x \
--output_names=y
```

4. Using Optimized graph

```
with tf.gfile.GFile('graph_optimized.pb', 'rb') as f:
   graph_def_optimized = tf.GraphDef()
   graph_def_optimized.ParseFromString(f.read())

G = tf.Graph()

with tf.Session(graph=G) as sess:
    y, = tf.import_graph_def(graph_def_optimized, return_elements=['y:0'])
    print('Operations in Optimized Graph:')
    print([op.name for op in G.get_operations()])
    x = G.get_tensor_by_name('import/x:0')
    out = sess.run(y, feed_dict={x: 1.0})
    print(out)

#Output
#Operations in Optimized Graph:
#['import/x', 'import/a', 'import/y']
#6.0
```

5. For multiple output names

If there are multiple output nodes, then specify : output_node_names = 'boxes, scores, classes' and import graph by,

```
boxes,scores,classes, = tf.import_graph_def(graph_def_optimized, return_elements=['boxes:0', 'scores:0', 'classes:0'])
```

## Another example

```
python -m tensorflow.python.tools.optimize_for_inference \
  --input HGB_U2D_Mi9_HTM_pillar_detection_x_20200825.pb \
  --output HGB_U2D_Mi9_HTM_pillar_detection_x_20200825_optimized.pb \
  --frozen_graph=True \
  --input_names=input_1 \
  --output_names="filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3,filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3,filtered_detections/map/TensorArrayStack/TensorArrayGatherV3"
```