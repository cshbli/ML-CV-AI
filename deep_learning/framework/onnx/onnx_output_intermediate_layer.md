# Get intermediary layer output of an ONNX model

## Expand the outputs of the original ONNX model

* Assume the original ONNX model name is "model.onnx"
* Assume the intermediate layer name is "layer_name"

We can expand the original ONNX model to add one more output.

```
import onnx
import onnxruntime
import numpy as np

model = onnx.load("model.onnx")
model_path = "model_with_layer.onnx"

intermediate_layer_value_info = onnx.helper.ValueInfoProto()
intermediate_layer_value_info.name = "layer_name"

model.graph.output.extend([intermediate_layer_value_info])
onnx.save(model, model_path)
```

## rt.InferenceSession will have one extra output

* Assume there are `n` outputs of the original model "model.onnx".
* There are going to to `n+1` outputs of the new model "model_with_layer.onnx".
* Assume the input name of the model is `input_name` and the input numpy array is `input`.

```
sess = onnxruntime.InferenceSession("model_with_layer.onnx")

outputs = sess.run(None, {'input_name': input})

intermediate_output = outputs[n]
```
