﻿# Install PyTorch

## Installation

- Create a virtual environment for PyTorch

```
python3 -m venv ./venv/torch1.13.0
```

- Activate the venv

```
source ~/venv/torch1.13.0/bin/activate
```

- Update pip
```
python -m pip install --upgrade pip
```

- Install NumPy SciPy Pandas

```
pip install numpy scipy pandas
```

- Install PyTorch

```
pip install torch torchvision torchaudio
```

- Install Jupyter

```
pip install jupyter
```

- Install Matplotlib

```
pip install matplotlib
```

- Install ONNX
```
pip install onnx
```

- Install onnxruntime

```
pip install onnxruntime onnxruntime-gpu
```

- Install onnxoptimizer
```
pip install onnxoptimizer
```

- Install OpenCV

```
pip install opencv-python
```

- Install Scikit-learn

```
pip install scikit-learn
```

- Install graphviz

```
sudo apt install graphviz
```

```
pip install graphviz
```

- Install torchviz

```
pip install torchviz
```

### Fetch specified wheel 

- example:

```
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## Verification

To ensure that PyTorch was installed correctly, we can verify the installation by running sample PyTorch code. Here we will construct a randomly initialized tensor.

```
import torch
print(torch.__version__)

x = torch.rand(5, 3)
print(x)
```

The output should be something similar to:

```
tensor([[0.3380, 0.3845, 0.3217],
        [0.8337, 0.9050, 0.2650],
        [0.2979, 0.7141, 0.9069],
        [0.1449, 0.1132, 0.1375],
        [0.4675, 0.3947, 0.1426]])
```

Additionally, to check if your GPU driver and CUDA is enabled and accessible by PyTorch, run the following commands to return whether or not the CUDA driver is enabled:

```
import torch
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.get_device_name(0)
torch.cuda.current_device()
```

Sometimes especially while using docker, although torch.cuda.is_available() returns True. It is possible that some errors like the following will pop up:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

To make sure the CUDA is available:

```
x = torch.rand(5, 3).to('cuda')
print(x)
```
