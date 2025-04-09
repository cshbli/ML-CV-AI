import torch
import torch.onnx
import onnx
import onnxruntime
import torch.nn as nn
import torch.nn.functional as F

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

    
def save_model_and_convert_to_onnx():
    # Initialize model
    ffn = FeedForwardNetwork()
    
    # Create dummy input for ONNX export
    dummy_input = torch.randn(1, 1, 3584)
    
    # 1. Save PyTorch model
    torch.save({
        'model_state_dict': ffn.state_dict(),
        'embed_dim': ffn.embed_dim,
        'hidden_dim': ffn.hidden_dim
    }, "ffn_model.pt")
    print("PyTorch model saved as ffn_model.pt")
    
    # 2. Export to ONNX
    torch.onnx.export(
        ffn,                      # model being run
        dummy_input,              # model input 
        "ffn_model.onnx",         # where to save the model
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
    print("ONNX model saved as ffn_model.onnx")
    
    # 3. Verify the ONNX model (optional)    
    onnx_model = onnx.load("ffn_model.onnx")
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully")


# Load the saved PyTorch model
def load_pytorch_model():
    checkpoint = torch.load("ffn_model.pt")
    model = FeedForwardNetwork(
        embed_dim=checkpoint['embed_dim'],
        hidden_dim=checkpoint['hidden_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    return model    


# Load the ONNX model and run inference
def run_onnx_model():
    # Create an ONNX Runtime session
    ort_session = onnxruntime.InferenceSession("ffn_model.onnx")
    
    # Prepare input
    input_data = torch.randn(1, 1, 3584).numpy()
    ort_inputs = {'input': input_data}
    
    # Run inference
    ort_outputs = ort_session.run(None, ort_inputs)
    print("ONNX model output shape:", ort_outputs[0].shape)

# Test
def test_ffn():
    ffn = FeedForwardNetwork()
    x1 = torch.randn(1, 1, 3584)  # First token
    out1 = ffn(x1)
    print("Output shape after first token:", out1.shape)

if __name__ == "__main__":
    test_ffn()
    save_model_and_convert_to_onnx()
