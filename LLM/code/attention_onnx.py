import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import numpy as np
import argparse
import os

class GroupedQueryAttentionONNX(nn.Module):
    def __init__(self, embed_dim=3584, num_q_heads=28, num_groups=4, head_dim=128, max_seq_len=2048):
        super(GroupedQueryAttentionONNX, self).__init__()
        self.embed_dim = embed_dim
        self.num_q_heads = num_q_heads
        self.num_groups = num_groups
        self.head_dim = head_dim
        self.num_heads_per_group = num_q_heads // num_groups        
        
        self.q_proj = nn.Linear(embed_dim, num_q_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, num_groups * head_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, num_groups * head_dim, bias=True)
        self.out_proj = nn.Linear(num_q_heads * head_dim, embed_dim, bias=False)        
        
        # Create a stateless forward to handle KV cache as inputs/outputs
    
    def forward(self, x, past_key=None, past_value=None, use_cache=True):
        """
        Stateless forward pass that takes KV cache as inputs and returns them as outputs
        
        Parameters:
        - x: Input tensor [batch_size, seq_len, embed_dim]
        - past_key: Past key cache [batch_size, num_groups, past_seq_len, head_dim]
        - past_value: Past value cache [batch_size, num_groups, past_seq_len, head_dim]
        - use_cache: Whether to use KV cache
        
        Returns:
        - output: Output tensor [batch_size, seq_len, embed_dim]
        - present_key: Updated key cache [batch_size, num_groups, total_seq_len, head_dim]
        - present_value: Updated value cache [batch_size, num_groups, total_seq_len, head_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        past_seq_len = 0 if past_key is None else past_key.size(2)
        
        # Project query, key, value
        q = self.q_proj(x).view(batch_size, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(x).view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(x).view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
        
        # Handle KV cache
        if use_cache and past_key is not None:
            k = torch.cat([past_key, k_new], dim=2)
            v = torch.cat([past_value, v_new], dim=2)
        else:
            k = k_new
            v = v_new
            
        # Set up present KV cache for return
        present_key, present_value = k, v
        
        # Expand grouped keys and values to match query heads
        # In ONNX, we need to avoid dynamic indexing, so we'll expand manually
        k_expanded = torch.zeros(batch_size, self.num_q_heads, k.size(2), self.head_dim, device=x.device)
        v_expanded = torch.zeros(batch_size, self.num_q_heads, v.size(2), self.head_dim, device=x.device)
        
        # This loop avoids dynamic indexing which can be problematic in ONNX
        for i in range(self.num_groups):
            for j in range(self.num_heads_per_group):
                head_idx = i * self.num_heads_per_group + j
                k_expanded[:, head_idx] = k[:, i]
                v_expanded[:, head_idx] = v[:, i]
        
        # Attention calculation
        attn_scores = torch.matmul(q, k_expanded.transpose(-1, -2)) / (self.head_dim ** 0.5)
        
        # Causal masking - for ONNX export, use a fixed causal mask
        if k.size(2) > 1:  # Only apply mask if we have more than one token
            mask_seq_len = q.size(2) + past_seq_len
            # Create causal mask for the current sequence plus past
            mask = torch.triu(torch.ones(mask_seq_len, mask_seq_len, device=x.device), diagonal=1)
            # Select the portion relevant to current q and k
            mask = mask[past_seq_len:, :k.size(2)]
            # Add batch and head dimensions
            mask = mask.unsqueeze(0).unsqueeze(1).expand(batch_size, self.num_q_heads, -1, -1)
            # Apply mask
            attn_scores = attn_scores.masked_fill(mask.bool(), float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v_expanded)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)
        
        output = x + attn_output  # Residual connection

        return output, present_key, present_value
    

def load_weights_from_npy(model, layer_name, npy_path):
    """Load weights from a .npy file into a specific layer of the model"""
    weights = np.load(npy_path)
    
    if not hasattr(model, layer_name):
        raise ValueError(f"Model does not have a layer named '{layer_name}'")
    
    layer = getattr(model, layer_name)
    
    expected_shape = layer.weight.shape
    if weights.shape != expected_shape:
        raise ValueError(f"Weight shape mismatch: Expected {expected_shape}, got {weights.shape}")
    
    weights_tensor = torch.from_numpy(weights).to(layer.weight.dtype)
    layer.weight.data = weights_tensor
    print(f"Successfully loaded weights for {layer_name} from {npy_path}")


def export_attention_to_onnx(attention_layer, save_path, seq_len=32, batch_size=1, dtype=torch.float32, use_past=True):
    """Export the attention layer to ONNX format with fixed sequence length"""
    attention_layer.eval()
    attention_layer = attention_layer.to(dtype)
    
    if use_past:
        # For decoder-style inference with KV cache
        # Create inputs
        x = torch.randn(batch_size, seq_len, attention_layer.embed_dim, dtype=dtype)
        past_key = torch.randn(batch_size, attention_layer.num_groups, 0, attention_layer.head_dim, dtype=dtype)
        past_value = torch.randn(batch_size, attention_layer.num_groups, 0, attention_layer.head_dim, dtype=dtype)
        
        # Trace with KV cache inputs/outputs
        torch.onnx.export(
            attention_layer,
            (x, past_key, past_value, True),  # inputs
            f"{save_path}_with_past.onnx",
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input', 'past_key', 'past_value', 'use_cache'],
            output_names=['output', 'present_key', 'present_value'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'past_key': {0: 'batch_size', 2: 'past_seq_len'},
                'past_value': {0: 'batch_size', 2: 'past_seq_len'},
                'output': {0: 'batch_size'},
                'present_key': {0: 'batch_size', 2: 'total_seq_len'},
                'present_value': {0: 'batch_size', 2: 'total_seq_len'}
            }
        )
        print(f"ONNX model with KV cache saved as {save_path}_with_past.onnx")
        
    # Also export a version without past for initial token processing
    x = torch.randn(batch_size, seq_len, attention_layer.embed_dim, dtype=dtype)
    
    # Define a wrapper class for the no-past case
    class WrapperNoPast(nn.Module):
        def __init__(self, attn_layer):
            super(WrapperNoPast, self).__init__()
            self.attn = attn_layer
            
        def forward(self, x):
            output, _, _ = self.attn(x, None, None, True)
            return output
    
    wrapper = WrapperNoPast(attention_layer)
    
    torch.onnx.export(
        wrapper,
        (x,),  # inputs
        f"{save_path}_no_past.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"ONNX model without KV cache saved as {save_path}_no_past.onnx")
    
    # Also export a PyTorch model for reference
    torch.save({
        'model_state_dict': attention_layer.state_dict(),
        'embed_dim': attention_layer.embed_dim,
        'num_q_heads': attention_layer.num_q_heads,
        'num_groups': attention_layer.num_groups,
        'head_dim': attention_layer.head_dim
    }, f"{save_path}.pt")
    print(f"PyTorch model saved as {save_path}.pt")


def main():
    parser = argparse.ArgumentParser(description="Export Grouped Query Attention to ONNX")
    parser.add_argument('--embed_dim', type=int, default=1536, help='Embedding dimension')
    parser.add_argument('--num_q_heads', type=int, default=24, help='Number of query heads')
    parser.add_argument('--num_groups', type=int, default=4, help='Number of KV groups')
    parser.add_argument('--head_dim', type=int, default=64, help='Dimension of each attention head')
    parser.add_argument('--seq_len', type=int, default=32, help='Sequence length for export')
    parser.add_argument('--save_path', type=str, default="attention_model", help='Path to save models')
    parser.add_argument('--dtype', type=str, choices=['float32', 'float16'], default='float32', help='Data type')
    
    # Add arguments for weight loading
    parser.add_argument('--load_weights', action='store_true', help='Load weights from .npy files')
    parser.add_argument('--q_proj_weights', type=str, default=None, help='Path for q_proj weights')
    parser.add_argument('--k_proj_weights', type=str, default=None, help='Path for k_proj weights')
    parser.add_argument('--v_proj_weights', type=str, default=None, help='Path for v_proj weights')
    parser.add_argument('--out_proj_weights', type=str, default=None, help='Path for out_proj weights')
    
    args = parser.parse_args()
    
    # Set dtype
    dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    
    # Create the attention layer
    attention = GroupedQueryAttentionONNX(
        embed_dim=args.embed_dim,
        num_q_heads=args.num_q_heads,
        num_groups=args.num_groups,
        head_dim=args.head_dim
    )
    
    # Load weights if specified
    if args.load_weights:
        if args.q_proj_weights:
            load_weights_from_npy(attention, 'q_proj', args.q_proj_weights)
        if args.k_proj_weights:
            load_weights_from_npy(attention, 'k_proj', args.k_proj_weights)
        if args.v_proj_weights:
            load_weights_from_npy(attention, 'v_proj', args.v_proj_weights)
        if args.out_proj_weights:
            load_weights_from_npy(attention, 'out_proj', args.out_proj_weights)
    
    # Export to ONNX
    export_attention_to_onnx(
        attention_layer=attention,
        save_path=args.save_path,
        seq_len=args.seq_len,
        dtype=dtype
    )


if __name__ == "__main__":
    main()