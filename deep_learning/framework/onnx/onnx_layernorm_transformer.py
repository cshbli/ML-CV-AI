import numpy as np

import onnx
import onnxruntime
import onnx.helper
from onnx import onnx_ml_pb2 as xpb2
import onnxoptimizer
import onnxsim
from onnx import TensorProto

_optimize = False
_simplify = False
_remove_initializer = False

np.random.seed(0)

# Optimizer passes:
OPTIMIZER_PASSES = ['eliminate_deadend',
                    'eliminate_duplicate_initializer',
                    'eliminate_identity',
                    'eliminate_if_with_const_cond',
                    'eliminate_nop_cast',
                    'eliminate_nop_dropout',
                    'eliminate_nop_flatten',
                    'eliminate_nop_monotone_argmax',
                    'eliminate_nop_pad',
                    'eliminate_nop_transpose',
                    'eliminate_unused_initializer',
                    'extract_constant_to_initializer',
                    'fuse_add_bias_into_conv',
                    'fuse_bn_into_conv',
                    'fuse_consecutive_concats',
                    'fuse_consecutive_log_softmax',
                    'fuse_consecutive_reduce_unsqueeze',
                    'fuse_consecutive_squeezes',
                    'fuse_consecutive_transposes',
                    'fuse_matmul_add_bias_into_gemm',
                    'fuse_pad_into_conv',
                    'fuse_transpose_into_gemm',
                    'lift_lexical_references']

# Create an onnx graph from scratch
graph1 = xpb2.GraphProto(name='onnx_graph')

# Create a BatchNormalization node
node1 = onnx.helper.make_node(op_type='LayerNormalization', inputs=[
                              'input', 'scale', 'bias'], outputs=['output'], name='LayerNorm')

# Add this Softmax node to the graph
graph1.node.append(node1)

# We should explicitly specify the named inputs to the graph
input1 = onnx.helper.make_tensor_value_info(
    'input', TensorProto.FLOAT, [1, 3200, 256])
graph1.input.append(input1)

#scale = onnx.helper.make_tensor_value_info('scale', TensorProto.FLOAT, [16])
scale = onnx.helper.make_tensor(
    'scale', TensorProto.FLOAT, [256], np.random.rand(256))
graph1.initializer.append(scale)

#bias = onnx.helper.make_tensor_value_info('bias', TensorProto.FLOAT, [16])
bias = onnx.helper.make_tensor(
    'bias', TensorProto.FLOAT, [256], np.random.rand(256))
graph1.initializer.append(bias)

# Similarly, we add the named output with its corresponding type and dimension
output1 = onnx.helper.make_tensor_value_info(
    'output', TensorProto.FLOAT, [1, 3200, 256])
graph1.output.append(output1)

# Create a model
model1 = onnx.helper.make_model(
    graph1, opset_imports=[onnx.helper.make_opsetid("", 17)])

# Standard ONNX checking
onnx.checker.check_model(model1)

# Optimize the model
if _optimize:
    try:
        model1 = onnxoptimizer.optimize(model1, passes=OPTIMIZER_PASSES)
    except Exception as e:
        print(str(e))

# Simply the model
if _simplify:
    model1, res = onnxsim.simplify(model1)

# Remove initializer from inputs
# From: onnxruntime/tools/python/remove_initializer_from_input.py
if _remove_initializer:
    graph1 = model1.graph
    inputs = graph1.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input
    for initializer in graph1.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

# Save the graph to a file
onnx.save(model1, "onnx_layernorm_transformer.onnx")

input1 = np.random.rand(1, 3200, 256)

example_input = {"input": input1.astype(np.float32)}
sess = onnxruntime.InferenceSession("onnx_layernorm_transformer.onnx")
results = sess.run(['output'], example_input)
print("Input: ", example_input.get("input"))
print("Output: ", results[0])

# Save the input data as data_0.npy
np.save('data_0.npy', input1.astype(np.float32))
