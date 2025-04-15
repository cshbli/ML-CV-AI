import argparse
import numpy as np
import torch
import torch.onnx
import onnx
import onnxruntime
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.quantization import quantize_dynamic

def parse_args():
    parser = argparse.ArgumentParser(description='Feed Forward Network with configurable dimensions')
    parser.add_argument('--embed_dim', type=int, default=3584, 
                        help='Embedding dimension (default: 3584)')
    parser.add_argument('--hidden_dim', type=int, default=18944, 
                        help='Hidden dimension (default: 18944)')
    parser.add_argument('--dtype', type=str, choices=['float32', 'float16'], default='float32',
                        help='Data type for model weights (default: float32)')
    parser.add_argument('--save_path', type=str, default="ffn_model",
                        help='Path prefix for saving models (default: "ffn_model")')
    parser.add_argument('--only_test', action='store_true',
                        help='Only run test without saving models')
    parser.add_argument('--quantize', action='store_true',
                       help='Quantize the model to INT8')
    parser.add_argument('--load_weights', action='store_true',
                        help='Load weights from .npy files')
    parser.add_argument('--down_proj_weights', type=str, default=None, 
                        help='Path to .npy file for down_proj weights')
    parser.add_argument('--gate_proj_weights', type=str, default=None,
                        help='Path to .npy file for gate_proj weights')
    parser.add_argument('--up_proj_weights', type=str, default=None,
                        help='Path to .npy file for up_proj weights')
    parser.add_argument('--seq_len', type=int, default=1,
                        help='Fixed sequence length for ONNX model (default: 1)')
    return parser.parse_args()


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.g * x_norm

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim=3584, hidden_dim=18944):
        super(FeedForwardNetwork, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Gate and Up projections
        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=False)  # [3584, 18944]
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=False)    # [3584, 18944]
        
        # Down projection to return to embed_dim
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=False)  # [18944, 3584]
        # For Linear layers, the weight tensor shape is [out_features, in_features], i.e. [3584, 18944]

    def forward(self, x):
        # Apply RMSNorm
        # x_norm = self.rms_norm(x)  # [batch_size, seq_len, embed_dim]
        x_norm = x  # Assuming input is already normalized for simplicity
        
        # Gate and Up projections
        gate = self.gate_proj(x_norm)  # [batch_size, seq_len, 18944]
        up = self.up_proj(x_norm)      # [batch_size, seq_len, 18944]
        
        # Gated activation (element-wise multiplication + SiLU)
        hidden = F.silu(gate) * up     # [batch_size, seq_len, 18944]
        
        # Down projection
        output = self.down_proj(hidden)  # [batch_size, seq_len, 3584]
        
        # Residual connection
        output = x + output  # Add original input (pre-norm) to FFN output

        return output
    

def load_weights_from_npy(model, layer_name, npy_path):
    """
    Load weights from a .npy file into a specific layer of the model
    
    Parameters:
    - model: The PyTorch model
    - layer_name: Name of the layer (e.g., 'down_proj', 'gate_proj', 'up_proj')
    - npy_path: Path to the .npy file containing the weights
    """
    # Load weights from .npy file
    weights = np.load(npy_path)
    
    # Get the target layer
    if not hasattr(model, layer_name):
        raise ValueError(f"Model does not have a layer named '{layer_name}'")
    
    layer = getattr(model, layer_name)
    
    # Check if shapes match
    expected_shape = layer.weight.shape
    # For Linear layers, the shape is [out_features, in_features]
    if weights.shape != expected_shape:
        raise ValueError(f"Weight shape mismatch: Expected {expected_shape}, got {weights.shape}")
    
    # Convert numpy array to torch tensor and match the target data type, in case the data type is different
    # For example, if the weights are in float16 and the layer is in float32
    weights_tensor = torch.from_numpy(weights).to(layer.weight.dtype)
    
    # Assign to layer
    layer.weight.data = weights_tensor
    print(f"Successfully loaded weights for {layer_name} from {npy_path}")    

    
def save_model_and_convert_to_onnx(ffn, save_path="ffn_model", dtype=torch.float32, seq_len=32):    
    # Create dummy input for ONNX export with fixed sequence length
    dummy_input = torch.randn(1, seq_len, ffn.embed_dim).to(dtype)  # [batch_size, seq_len, embed_dim]
    
    # 1. Save PyTorch model
    torch.save({
        'model_state_dict': ffn.state_dict(),
        'embed_dim': ffn.embed_dim,
        'hidden_dim': ffn.hidden_dim,
        'seq_len': seq_len
    }, f"{save_path}.pt")
    print(f"PyTorch model saved as {save_path}.pt")
    
    # 2. Export to ONNX
    torch.onnx.export(
        ffn,                      # model being run
        dummy_input,              # model input 
        f"{save_path}.onnx",      # where to save the model
        export_params=True,       # store the trained parameters
        opset_version=12,         # ONNX version
        do_constant_folding=True, # optimize constant folding
        input_names=['input'],    # model's input names
        output_names=['output'],  # model's output names
        # This is important for dynamic shapes
        # dynamic_axes allows for variable input shapes
        # Here we set batch_size and sequence_length as dynamic dimensions
        # dynamic_axes={
        #     'input': {0: 'batch_size', 1: 'sequence_length'},
        #     'output': {0: 'batch_size', 1: 'sequence_length'}
        # }
    )
    print(f"ONNX model saved as {save_path}.onnx")
    
    # 3. Verify the ONNX model (optional)    
    onnx_model = onnx.load(f"{save_path}.onnx")
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully")


# Load the saved PyTorch model
def load_pytorch_model(path="ffn_model.pt"):
    checkpoint = torch.load(path)
    model = FeedForwardNetwork(
        embed_dim=checkpoint['embed_dim'],
        hidden_dim=checkpoint['hidden_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    return model    


# Load the ONNX model and run inference
def run_onnx_model(path="ffn_model.onnx", embed_dim=3584):
    # Create an ONNX Runtime session
    ort_session = onnxruntime.InferenceSession(path)
    
    # Prepare input
    input_data = torch.randn(1, 1, embed_dim).numpy()
    ort_inputs = {'input': input_data}
    
    # Run inference
    ort_outputs = ort_session.run(None, ort_inputs)
    print("ONNX model output shape:", ort_outputs[0].shape)


def quantize_model_to_int8(model, save_path="ffn_model_int8"):
    """Quantize model weights to INT8 and save the quantized model"""
    # Set model to evaluation mode
    model.eval()
    
    # Clone the model to avoid modifying the original
    model_fp32 = copy.deepcopy(model)
    
    # Prepare model for static quantization
    # 1. Fuse operations if needed (not needed for our FFN)
    # 2. Specify quantization configuration
    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # 3. Prepare the model for quantization
    model_prepared = torch.quantization.prepare(model_fp32)
    
    # 4. Calibrate with example inputs (recommended to use a small dataset)
    with torch.no_grad():
        # Example calibration - normally you'd use more data
        dummy_input = torch.randn(1, 1, model.embed_dim)
        model_prepared(dummy_input)
    
    # 5. Convert to quantized model
    model_int8 = torch.quantization.convert(model_prepared)
    
    # Save the quantized model
    torch.save({
        'model_state_dict': model_int8.state_dict(),
        'embed_dim': model_int8.embed_dim,
        'hidden_dim': model_int8.hidden_dim
    }, f"{save_path}.pt")
    print(f"Quantized PyTorch model saved as {save_path}.pt")
    
    return model_int8


# Quantize an existing ONNX model to INT8, using dynamic quantization
# This is a simpler method than static quantization
def quantize_onnx_model(onnx_model_path, quantized_model_path):
    """Quantize an existing ONNX model to INT8"""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    # Quantize the model
    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=quantized_model_path,
        weight_type=QuantType.QInt8
    )
    print(f"Quantized ONNX model saved as {quantized_model_path}")    


def quantize_onnx_model_static(onnx_model_path, quantized_model_path):
    """Quantize an existing ONNX model to INT8"""
    from onnxruntime.quantization import quantize_static, QuantType
    
    # Create a dummy calibration data reader
    class DummyCalibrationDataReader:
        def __init__(self, input_name, embed_dim):
            self.input_name = input_name
            self.embed_dim = embed_dim
            self.has_returned_data = False
            
        def get_next(self):
            if self.has_returned_data:
                return None
            
            dummy_input = torch.randn(1, 1, self.embed_dim).numpy()
            self.has_returned_data = True
            return {self.input_name: dummy_input}
            
        def rewind(self):
            self.has_returned_data = False
    
    # Create calibration reader with your model's embed_dim
    calibration_reader = DummyCalibrationDataReader(input_name='input', 
                                                    embed_dim=args.embed_dim)
    
    # Quantize the model
    quantize_static(
        model_input=onnx_model_path,
        model_output=quantized_model_path,
        calibration_data_reader=calibration_reader,
        weight_type=QuantType.QInt8
    )
    print(f"Quantized ONNX model saved as {quantized_model_path}")

# Hanlding Variable Length Inputs at Inference
# This function pads or truncates the input tensor to a fixed sequence length
def prepare_input_for_fixed_length_model(input_tensor, fixed_seq_len):
    """
    Pad or truncate input to match the fixed sequence length
    
    Parameters:
    - input_tensor: Input tensor of shape [batch_size, seq_len, embed_dim]
    - fixed_seq_len: Fixed sequence length expected by the model
    
    Returns:
    - Padded/truncated tensor of shape [batch_size, fixed_seq_len, embed_dim]
    """
    batch_size, seq_len, embed_dim = input_tensor.shape
    
    if seq_len > fixed_seq_len:
        # Truncate if input is longer than fixed length
        return input_tensor[:, :fixed_seq_len, :]
    elif seq_len < fixed_seq_len:
        # Pad with zeros if input is shorter
        padding = torch.zeros(batch_size, fixed_seq_len - seq_len, embed_dim, 
                             device=input_tensor.device, dtype=input_tensor.dtype)
        return torch.cat([input_tensor, padding], dim=1)
    else:
        # No change needed
        return input_tensor    


if __name__ == "__main__":
    args = parse_args()    
    
    ffn = FeedForwardNetwork(embed_dim=args.embed_dim, hidden_dim=args.hidden_dim)
    # Convert string dtype argument to torch dtype
    dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    # Convert model to desired dtype (float16 if you're using float16 weights)
    if dtype != torch.float32:
        ffn = ffn.to(dtype)
    print(f"Model dimensions: embed_dim={args.embed_dim}, hidden_dim={args.hidden_dim}")

    # Load weights if specified
    if args.load_weights:
        if args.down_proj_weights:
            load_weights_from_npy(ffn, 'down_proj', args.down_proj_weights)
        if args.gate_proj_weights:
            load_weights_from_npy(ffn, 'gate_proj', args.gate_proj_weights)
        if args.up_proj_weights:
            load_weights_from_npy(ffn, 'up_proj', args.up_proj_weights)

    x1 = torch.randn(1, 1, args.embed_dim).to(dtype)  # First token
    out1 = ffn(x1)    
    print("Output shape after first token:", out1.shape)    
    
    if not args.only_test:
        save_model_and_convert_to_onnx(ffn, save_path=args.save_path, dtype=dtype, seq_len=args.seq_len)

        if args.quantize:
            # # Method 1: Static quantization of PyTorch model            
            # quantized_model = quantize_model_to_int8(ffn, save_path=f"{args.save_path}_int8")
            
            # Method 2: Quantize the ONNX model
            quantize_onnx_model(f"{args.save_path}.onnx", f"{args.save_path}_int8.onnx")
            
            # # Method 3: Dynamic quantization (easier alternative)
            # quantized_dynamic = quantize_dynamic(
            #     model=copy.deepcopy(ffn),
            #     qconfig_spec={nn.Linear},  # Quantize only Linear layers
            # )
            # torch.save({
            #     'model_state_dict': quantized_dynamic.state_dict(),
            #     'embed_dim': quantized_dynamic.embed_dim,
            #     'hidden_dim': quantized_dynamic.hidden_dim
            # }, f"{args.save_path}_int8_dynamic.pt")
            # print(f"Dynamic quantized model saved as {args.save_path}_int8_dynamic.pt")
