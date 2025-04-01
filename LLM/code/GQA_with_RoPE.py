import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048, base=10000):
        super(RotaryPositionEmbedding, self).__init__()
        # Head Dimension: RoPE assumes that the head dimension is even. (128 is fine here).
        self.head_dim = head_dim
        self.base = base
        
        # Precompute frequency terms
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("theta", theta)
        
        # Precompute position indices
        # Precomputed RoPE tables (cos, sin) assume a max sequence length; for very long sequences, you might compute angles on-the-fly.
        positions = torch.arange(max_seq_len).float()
        angles = positions[:, None] * theta[None, :]
        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))

    def forward(self, x, offset=0):
        # x: [batch, num_heads, seq_len, head_dim]
        batch, num_heads, seq_len, head_dim = x.shape
        assert head_dim == self.head_dim
        
        # Get cos and sin for current sequence positions
        cos = self.cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
        sin = self.sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
        
        # Split x into even and odd dimensions
        x_even = x[..., 0::2]  # [batch, num_heads, seq_len, head_dim//2]
        x_odd = x[..., 1::2]
        
        # Apply rotation
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        
        # Interleave back
        x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1).view(batch, num_heads, seq_len, head_dim)
        return x_rot

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim=3584, num_q_heads=28, num_groups=4, head_dim=128, max_seq_len=2048):
        super(GroupedQueryAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_q_heads = num_q_heads
        self.num_groups = num_groups
        self.head_dim = head_dim
        self.num_heads_per_group = num_q_heads // num_groups
        
        self.q_proj = nn.Linear(embed_dim, num_q_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, num_groups * head_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, num_groups * head_dim, bias=True)
        self.out_proj = nn.Linear(num_q_heads * head_dim, embed_dim, bias=False)
        
        self.rope = RotaryPositionEmbedding(head_dim, max_seq_len)
        self.kv_cache = None

    def forward(self, x, use_cache=True, past_seq_len=0):
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        # [batch, num_q_heads, seq_len, head_dim]
        
        if use_cache and self.kv_cache is not None:
            past_k, past_v = self.kv_cache
            k_new = self.k_proj(x).view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
            v_new = self.v_proj(x).view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
            k = torch.cat([past_k, k_new], dim=2)
            v = torch.cat([past_v, v_new], dim=2)
        else:
            k = self.k_proj(x).view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
            # [batch, num_groups, seq_len, head_dim]
        
        if use_cache:
            self.kv_cache = (k, v)

        # Apply RoPE to Q and K
        q = self.rope(q, offset=past_seq_len)
        k = self.rope(k, offset=0)  # Keys include past positions, start from 0
        
        # Expand K and V for GQA
        group_idx = torch.arange(self.num_groups, device=x.device).repeat_interleave(self.num_heads_per_group)
        k_expanded = k[:, group_idx, :, :]
        v_expanded = v[:, group_idx, :, :]
        
        # Attention computation
        attn_scores = torch.matmul(q, k_expanded.transpose(-1, -2)) / (self.head_dim ** 0.5)
        # Causal Masking: For autoregressive models, add a causal mask to attn_scores before softmax.
        if past_seq_len > 0:
            # Mask out the future tokens in the attention scores
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(x.device)
            attn_scores = attn_scores.masked_fill(mask == 1, float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v_expanded)
        
        # Output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)
        return output

# Test
def test_gqa_with_rope():
    gqa = GroupedQueryAttention()
    x1 = torch.randn(1, 1, 3584)  # First token
    out1 = gqa(x1, use_cache=True, past_seq_len=0)
    print("Output shape:", out1.shape)
    x2 = torch.randn(1, 1, 3584)  # Second token
    out2 = gqa(x2, use_cache=True, past_seq_len=1)
    print("Output shape:", out2.shape)

if __name__ == "__main__":
    test_gqa_with_rope()