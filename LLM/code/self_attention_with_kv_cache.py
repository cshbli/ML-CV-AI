import torch
import torch.nn as nn

class SelfAttentionWithKVCache(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(SelfAttentionWithKVCache, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model  # Embedding dimension (e.g., 64)
        self.n_heads = n_heads  # Number of attention heads (e.g., 8). Each head has its own Q, K, V matrices.
        self.d_k = d_model // n_heads  # Dimension per head (e.g., 64 / 8 = 8)        
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # Output projection
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        # KV cache (will be initialized during forward pass)
        self.k_cache = None
        self.v_cache = None
    
    def reset_cache(self):
        """Reset the KV cache (e.g., for a new sequence)."""
        self.k_cache = None
        self.v_cache = None
    
    def forward(self, x, mask=None, use_cache=True):
        """
        x: Input tensor of shape (batch_size, seq_len, d_model), during inference, seq_len is typically 1 (one token at a time)
        mask: Optional attention mask (batch_size, 1, seq_len) or None
        use_cache: Whether to use and update the KV cache
        """
        batch_size, seq_len, _ = x.size()
        
        # Project inputs to Q, K, V
        q = self.W_q(x)  # (batch_size, seq_len, d_model)
        k = self.W_k(x)  # (batch_size, seq_len, d_model)
        v = self.W_v(x)  # (batch_size, seq_len, d_model)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # Shapes: (batch_size, n_heads, seq_len, d_k)
        
        # Update KV cache if using it
        if use_cache:
            if self.k_cache is None:
                # First call: Initialize cache with current k, v
                self.k_cache = k
                self.v_cache = v
            else:
                # Append new k, v to cache (along sequence dimension)
                self.k_cache = torch.cat([self.k_cache, k], dim=2)
                self.v_cache = torch.cat([self.v_cache, v], dim=2)
            k = self.k_cache
            v = self.v_cache
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # scores: (batch_size, n_heads, seq_len, cached_seq_len)
        
        # Apply causal mask (optional, for autoregressive models)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax over attention scores
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        # out: (batch_size, n_heads, seq_len, d_k)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_o(out)
        
        return out

# Example usage
def main():
    # Hyperparameters
    batch_size = 2
    seq_len = 3
    d_model = 64
    n_heads = 8
    
    # Initialize model
    attn = SelfAttentionWithKVCache(d_model, n_heads)
    
    # Dummy input (e.g., first token)
    x = torch.randn(batch_size, 1, d_model)  # Single token at a time
    print("Step 1: First token")
    out = attn(x, use_cache=True)
    print("Output shape:", out.shape)  # (2, 1, 64)
    print("KV cache shape:", attn.k_cache.shape)  # (2, 8, 1, 8)
    
    # Next token
    x2 = torch.randn(batch_size, 1, d_model)
    print("\nStep 2: Second token")
    out = attn(x2, use_cache=True)
    print("Output shape:", out.shape)  # (2, 1, 64)
    print("KV cache shape:", attn.k_cache.shape)  # (2, 8, 2, 8)
    
    # Reset cache for a new sequence
    attn.reset_cache()

if __name__ == "__main__":
    main()