# PyTorch QAT With TorchVision Model - MobilenetV2

## Prepration before start

```{.python .input  n=1}
import bstnnx
import bstnnx.training
print(f"bstnnx version: {bstnnx.__version__}")
print(f"bstnnx git version: {bstnnx.version.git_version}")
print(f"bstnnx training version: {bstnnx.training.__version__}")
print(f"bstnnx training git version: {bstnnx.training.git_version}")

import torch
import torch.nn as nn
import torch.nn.functional as F
print(f"PyTorch version: {torch.__version__}")

assert bstnnx.__version__ >= '4.0.4', 'This notebook need to use bstnnx training >= 1.0.0 release'
assert bstnnx.training.__version__ >= '1.0.0', 'This notebook need to use bstnnx training >= 1.0.0 release'
assert '1.9.1' in torch.__version__, 'This notebook need to use pytorch 1.9.1 release'
```

```{.python .input  n=2}
import os
import onnx
import json
import time
import shutil
import numpy as np
from tqdm import tqdm
from typing import Dict
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
```

```{.python .input  n=3}
import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
```

```{.python .input  n=4}
# Determine whether it is dry run or not
dry_run = os.getenv('DRY_RUN')
if dry_run:
    print(f'DRY_RUN is set to 1, running in dry run mode')
dry_run = 1
```

```{.python .input  n=5}
model_name = 'mobilenet_v2_torch'
input_dir = "./"
input_dir = os.path.abspath(input_dir)
if not os.path.isdir(input_dir):
        raise RuntimeError(f"{input_dir} does not exists")
print(f"input_dir = {input_dir}")

def create_dir_if_need(output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    ## convert to absolute_path
    output_dir = os.path.abspath(output_dir)
    return output_dir

output_dir = "../results/job0"
output_dir = create_dir_if_need(output_dir)
print(f"output_dir = {output_dir}")

tmp_dir = "../tmp"
tmp_dir = create_dir_if_need(tmp_dir)
print(f"tmp_dir = {tmp_dir}")

```

## Pytorch offical quantization step by step flow

### 1. Create model and define dataset

```{.python .input  n=6}
from quantable_mobilenetv2 import mobilenet_v2
```

```{.python .input  n=7}
fp32_model = mobilenet_v2(pretrained=False, progress=True).to(device)
```

```{.python .input  n=8}
fp32_model
```

### 2. Train the model

```{.python .input  n=9}
import torchvision
from torchvision import datasets, transforms

from torch.utils.data import (DataLoader, TensorDataset)

NUM_WORKER = 1
IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 1

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Scale((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Scale((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if dry_run:
    batch_size = 1
    torch.manual_seed(2022)
    dummy_dataset = TensorDataset(torch.rand(batch_size, 3, 224, 224), torch.randint(0, 10, (batch_size,)))
    train_loader = DataLoader(dummy_dataset,
                              batch_size=batch_size)
    test_loader = DataLoader(dummy_dataset,
                             batch_size=1)
else:
    trainset = torchvision.datasets.CIFAR10(root='./cifar10',train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)

    testset = torchvision.datasets.CIFAR10(root='./cifar10',train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

```{.python .input  n=10}
#Training configurations
def train_one_epoch(net, device, train_loader, optimizer, criterion, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 40 == 39:
            print('[%d/%d] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (batch_idx, len(train_loader), train_loss/(batch_idx+1), 100.*correct/total, correct, total))
```

```{.python .input  n=11}
import torch.nn as nn
import torch.optim as optim

epochs = 15
lr = 0.001
momentum = 0.9
weight_decay = 5e-4
fp32_mobilenet_v2_pt_path = os.path.join(tmp_dir, "fp32_mobilenet_v2.pt")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(fp32_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

for epoch in range(1, epochs + 1):
    train_one_epoch(fp32_model, device, train_loader, optimizer, criterion, epoch)
```

```{.python .input  n=12}
# Since the model is pretrained, we just need to save it
torch.save(fp32_model.state_dict(), fp32_mobilenet_v2_pt_path)
```

### 3. Define Quantize Model Procedure

```{.python .input  n=13}
def quantize_model(model, device, backend='default', sample_data=None):
    model.to(device)
    model.train()
    
    if backend == 'default':
        activation_quant = quantizer.FakeQuantize.with_args(
            observer=quantizer.default_observer.with_args(dtype=torch.qint8), 
            quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, reduce_range=False)
        weight_quant = quantizer.FakeQuantize.with_args(
            observer=quantizer.default_observer.with_args(dtype=torch.qint8), 
            quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, reduce_range=False)

        # assign qconfig to model
        model.qconfig = quantizer.QConfig(activation=activation_quant, weight=weight_quant)
        
        # prepare qat model using qconfig settings
        prepared_model = quantizer.prepare_qat(model, inplace=False)
    elif backend == 'bst':
        bst_activation_quant = quantizer.FakeQuantize.with_args(
            observer=quantizer.MinMaxObserver.with_args(dtype=torch.qint8), 
            quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, reduce_range=False)
        bst_weight_quant = quantizer.FakeQuantize.with_args(
            observer=quantizer.MinMaxObserver.with_args(dtype=torch.qint8), 
            quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_affine, reduce_range=False)
        
        # 1) [bst_alignment] get b0 pre-bind qconfig adjusting Conv's activation quant scheme
        pre_bind_qconfig = quantizer.pre_bind(model, input_tensor=sample_data.to('cpu'))
        
        # 2) assign qconfig to model
        model.qconfig = quantizer.QConfig(activation=bst_activation_quant, weight=bst_weight_quant,
                                                    qconfig_dict=pre_bind_qconfig)
        
        # 3) prepare qat model using qconfig settings
        prepared_model = quantizer.prepare_qat(model, inplace=False)  
        
        # 4) [bst_alignment] link model observers
        prepared_model = quantizer.link_modules(prepared_model, auto_detect=True, input_tensor=sample_data.to('cpu'), inplace=False)
    
    prepared_model.eval()
    
    return prepared_model

def quant_aware_training(model, device, data_loader):
    epochs = 10
    lr = 0.001
    momentum = 0.9
    weight_decay = 5e-4

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # QAT takes time and one needs to train over a few epochs.
    # Train and check accuracy after each epoch
    for epoch in range(1, epochs + 1):
        # This should be replaced by actual train one epoch function
        train_one_epoch(model, device, data_loader, optimizer, criterion, epoch)

        if epoch > 0:
            # Freeze quantizer parameters
            model.apply(torch.quantization.disable_observer)
        if epoch > 0:
            # Freeze batch norm mean and variance estimates
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    best_loss = 0.0
    best_acc = 0.0
    
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 40 == 39:
                print('[%d/%d] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (batch_idx, len(test_loader), test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                
            acc = 100.*correct/total
            loss = test_loss/(batch_idx+1)
            
            if acc > best_acc:
                best_loss = loss
                best_acc = acc
    
    return best_loss, best_acc
                
```

### 4. Torch Quantize Model

```{.python .input  n=14}
fp32_model = mobilenet_v2(pretrained=False, progress=True).to(device)
loaded_dict_enc = torch.load(fp32_mobilenet_v2_pt_path, map_location=device)
fp32_model.load_state_dict(loaded_dict_enc)
```

```{.python .input  n=15}
float_loss, float_acc = test(fp32_model, device, test_loader)
```

```{.python .input  n=16}
print(f"Loss: {float_loss} | Accuracy: {float_acc}")
```

```{.python .input  n=17}
import torch.quantization as quantizer

fp32_model = mobilenet_v2(pretrained=False, progress=True).to(device)
loaded_dict_enc = torch.load(fp32_mobilenet_v2_pt_path, map_location=device)
fp32_model.load_state_dict(loaded_dict_enc)

fp32_model.fuse_model()
prepared_model = quantize_model(fp32_model, device)
```

### 5. Torch QAT

```{.python .input  n=18}
quant_aware_training(prepared_model, device, train_loader)
```

```{.python .input  n=19}
torch_qat_loss, torch_qat_acc = test(prepared_model, device, test_loader)
```

```{.python .input  n=20}
print(f"Loss: {torch_qat_loss} | Accuracy: {torch_qat_acc}")
```

## BST quantization step by step flow

### 3. Define Quantize Model Procedure

```{.python .input  n=21}
# switch the quantization framework
import bstnnx.training.PyTorch.QAT.core as quantizer
```

```{.python .input  n=22}
#redefine the test function for inference (support onnx)
from bstnnx.training.PyTorch.QAT.helper import runtime_helper

def to_list(results):
    list_results = []
    assert isinstance(results, Dict)
    for key, value in results.items():
        list_results.append(value)
    return list_results

def quant_aware_training(model, device, data_loader, sample_data):
    epochs = 10
    lr = 0.001
    momentum = 0.9
    weight_decay = 5e-4

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # QAT takes time and one needs to train over a few epochs.
    # Train and check accuracy after each epoch
    for epoch in range(1, epochs + 1):
        # This should be replaced by actual train one epoch function
        train_one_epoch(model, device, data_loader, optimizer, criterion, epoch)

        if epoch > 0:
            # Freeze quantizer parameters
            model.apply(torch.quantization.disable_observer)
            
            # Extra step: to align hardware, it will only be applied once for unaligned model
            quantizer.align_bst_hardware(model, sample_data)
        if epoch > 0:
            # Freeze batch norm mean and variance estimates
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            
def test(model, device, test_loader):
    test_loss = 0
    correct = 0
    total = 0
    best_loss = 0.0
    best_acc = 0.0
    
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if isinstance(model, nn.Module):
                model.eval()
                outputs = model(inputs)
            elif isinstance(model, runtime_helper.OnnxRep):
                outputs = to_list(runtime_helper.run_onnx_inference(model, inputs))[0]
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 40 == 39:
                print('[%d/%d] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (batch_idx, len(test_loader), test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
            acc = 100.*correct/total        
            loss = test_loss/(batch_idx+1)
            
            if acc > best_acc:
                best_loss = loss
                best_acc = acc
    
    return best_loss, best_acc
```

### 4. BST Quantize Model

```{.python .input  n=23}
fp32_model = mobilenet_v2(pretrained=False, progress=True, use_bstnn=True).to(device)
loaded_dict_enc = torch.load(fp32_mobilenet_v2_pt_path, map_location=device)
fp32_model.load_state_dict(loaded_dict_enc)
```

```{.python .input  n=24}
float_loss, float_acc = test(fp32_model, device, test_loader)
```

```{.python .input  n=25}
print(f"Loss: {float_loss} | Accuracy: {float_acc}")
```

```{.python .input  n=26}
fp32_model = mobilenet_v2(pretrained=False, progress=True, use_bstnn=True).to(device)
loaded_dict_enc = torch.load(fp32_mobilenet_v2_pt_path, map_location=device)
fp32_model.load_state_dict(loaded_dict_enc)

# define test data used for fusing model
random_data = np.random.rand(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH).astype("float32")
sample_data = torch.from_numpy(random_data).to(device)

# use CPU on input_tensor as our backend for parsing GraphTopology forced model to be on CPU
fp32_model.eval()
fused_model = quantizer.fuse_modules(fp32_model, auto_detect=True, input_tensor=sample_data.cpu())
prepared_model = quantize_model(fused_model, device, backend="bst", sample_data=sample_data)
```

```{.python .input  n=27}
prepared_model
```

### 5. Quant-aware-training

```{.python .input  n=28}
quant_aware_training(prepared_model, device, train_loader, sample_data)
```

### Extra step:  Export float ONNX model and Json and optimize them with hardware constraints

```{.python .input  n=29}
rand_in = np.random.rand(1, 1, 3, IMAGE_HEIGHT, IMAGE_WIDTH).astype("float32")
sample_in = tuple(torch.from_numpy(x) for x in rand_in)
stage_dict={}
stage_dict['simplify_onnx'] = True
onnx_model_path, quant_param_json_path = quantizer.export_onnx(prepared_model, 
                                                               sample_in, 
                                                               stage_dict=stage_dict, 
                                                               result_dir=tmp_dir)
```

##  Model conversion flow

```{.python .input  n=30}
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))
```

```{.python .input  n=31}
%%writetemplate $tmp_dir/test_e2e_qat.yaml

data_reader_method: random_data_reader
model_name: qat_frozen_model
model_path: {onnx_model_path}
batch_size: 1
size_limit: 10
non_image_input: True
enable_in_scale: True
orig_model_format: onnx
device_engine: 'A1000B0'
stage:
  - stage_name: pre_processing_stage
    priority: 100
  - stage_name: graph_optimization_stage
    run_built_in_optimization: True
    optimization_passes:
      - convert_gemm
      - convert_max_pool_to_dsp
      - convert_slice
      - convert_eltwise_add
      - convert_eltwise_mul
      - convert_global_avgpool
      - fuse_activation
      - convert_relu
      - fuse_conv_add
      - convert_to_skip_node
      - convert_resize_to_bst_resize
    optimization_parameters:
        fuse_conv_add:
            quant_params_json_path: {quant_param_json_path}
    priority: 200
  - stage_name: quantization_stage
    quantization_method: bst_standard_quantization_flow
    quant_params_json_path: {quant_param_json_path}
    priority: 300  
  - stage_name: graph_partition_stage
    priority: 400
  - stage_name: section_binding_stage
    priority: 500
  - stage_name: code_generation_stage
    priority: 600
  - stage_name: code_compilation_stage
    priority: 700
  - stage_name: run_emulation_stage
    profiling_mode: 2
    priority: 800
  - stage_name: partition_evaluation_stage
    priority: 900
  - stage_name: run_emulation_stage
    profiling_mode: 0
    priority: 1000
```

```{.python .input  n=32}
CONFIG_FILE_PATH = os.path.join(tmp_dir, "test_e2e_qat.yaml")
RESULT_DIR = output_dir
```

```{.python .input  n=33}
import bstnnx.frontend.stage_flow_control_main
RESULT_DIR = output_dir
extra_option = {
    "logging_level": "INFO",
    "priority_range": "100-1000,1200",
    "build_type": "html",
    "BSTNNX_NET_FW_DIR": "/bsnn/work/bstnnx_release/third_party/Net-FW-xos-v0-edp-gemm"
}

bstnnx.frontend.stage_flow_control_main.bstnnx_run(config=CONFIG_FILE_PATH, \
                                                      result_dir=RESULT_DIR, \
                                                      extra=extra_option)
```

# Sanity check with reference Model

```{.python .input}
bst_reference_model_path = os.path.join(output_dir, "300_QuantizationStage/quant_model.onnx")
```

```{.python .input}
device = 'cpu'
custom_bst_op_path = bstnnx.backend.custom_op.get_custom_op_lib_path()
onnx_rep = runtime_helper.OnnxRep(bst_reference_model_path, custom_op_lib=custom_bst_op_path)
bst_refernce_loss, bst_reference_acc  = test(onnx_rep, device, test_loader)
```

```{.python .input}
print(f"Float model - Loss: {float_loss} | Accuracy: {float_acc}")
print(f"Torch QAT model - Loss: {torch_qat_loss} | Accuracy: {torch_qat_acc}")
print(f"BST Reference model - Loss: {bst_refernce_loss} | Accuracy: {bst_reference_acc}")
```