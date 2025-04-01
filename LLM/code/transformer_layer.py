import torch
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

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048, base=10000):
        super(RotaryPositionEmbedding, self).__init__()
        self.head_dim = head_dim
        self.base = base
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("theta", theta)
        positions = torch.arange(max_seq_len).float()
        angles = positions[:, None] * theta[None, :]
        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))

    def forward(self, x, offset=0):
        batch, num_heads, seq_len, head_dim = x.shape
        cos = self.cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
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
        
        self.rms_norm = RMSNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, num_q_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, num_groups * head_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, num_groups * head_dim, bias=True)
        self.out_proj = nn.Linear(num_q_heads * head_dim, embed_dim, bias=False)
        
        self.rope = RotaryPositionEmbedding(head_dim, max_seq_len)
        self.kv_cache = None

    def forward(self, x, use_cache=True, past_seq_len=0):
        batch_size, seq_len, embed_dim = x.shape
        x_norm = self.rms_norm(x)
        
        q = self.q_proj(x_norm).view(batch_size, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        
        if use_cache and self.kv_cache is not None:
            past_k, past_v = self.kv_cache
            k_new = self.k_proj(x_norm).view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
            v_new = self.v_proj(x_norm).view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
            k = torch.cat([past_k, k_new], dim=2)
            v = torch.cat([past_v, v_new], dim=2)
        else:
            k = self.k_proj(x_norm).view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
            v = self.v_proj(x_norm).view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
        
        if use_cache:
            self.kv_cache = (k, v)

        q = self.rope(q, offset=past_seq_len)
        k = self.rope(k, offset=0)
        
        group_idx = torch.arange(self.num_groups, device=x.device).repeat_interleave(self.num_heads_per_group)
        k_expanded = k[:, group_idx, :, :]
        v_expanded = v[:, group_idx, :, :]
        
        attn_scores = torch.matmul(q, k_expanded.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v_expanded)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)
        
        output = x_norm + attn_output  # Residual connection

        return output

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim=3584, hidden_dim=18944):
        super(FeedForwardNetwork, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # RMSNorm on the input
        self.rms_norm = RMSNorm(embed_dim)
        
        # Gate and Up projections
        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=False)  # [3584, 18944]
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=False)    # [3584, 18944]
        
        # Down projection to return to embed_dim
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=False)  # [18944, 3584]

    def forward(self, x):
        # Apply RMSNorm
        x_norm = self.rms_norm(x)  # [batch_size, seq_len, embed_dim]
        
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

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim=3584, num_q_heads=28, num_groups=4, head_dim=128, max_seq_len=2048, hidden_dim=18944):
        super(TransformerLayer, self).__init__()
        self.gqa = GroupedQueryAttention(embed_dim, num_q_heads, num_groups, head_dim, max_seq_len)
        self.ffn = FeedForwardNetwork(embed_dim, hidden_dim)

    def forward(self, x, use_cache=True, past_seq_len=0):
        # GQA output
        attn_output = self.gqa(x, use_cache, past_seq_len)
        # FFN output
        output = self.ffn(attn_output)
        return output

# Test
def test_transformer_layer():
    layer = TransformerLayer()
    x1 = torch.randn(1, 1, 3584)  # First token
    out1 = layer(x1, use_cache=True, past_seq_len=0)
    print("Output shape after first token:", out1.shape)
    x2 = torch.randn(1, 1, 3584)  # Second token
    out2 = layer(x2, use_cache=True, past_seq_len=1)
    print("Output shape after second token:", out2.shape)

if __name__ == "__main__":
    test_transformer_layer()