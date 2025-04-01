import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim=3584, num_q_heads=28, num_groups=4, head_dim=128):
        super(GroupedQueryAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_q_heads = num_q_heads
        self.num_groups = num_groups
        self.head_dim = head_dim
        self.num_heads_per_group = num_q_heads // num_groups  # 7 in this case
        
        # Query projection: one per head
        self.q_proj = nn.Linear(embed_dim, num_q_heads * head_dim, bias=False)
        # Key/Value projections: one per group
        self.k_proj = nn.Linear(embed_dim, num_groups * head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_groups * head_dim, bias=False)
        # Output projection
        self.out_proj = nn.Linear(num_q_heads * head_dim, embed_dim, bias=False)
        
        # KV cache (initialized as None, will be updated during forward pass)
        self.kv_cache = None

    def forward(self, x, use_cache=True, past_seq_len=0):
        batch_size, seq_len, embed_dim = x.shape
        
        # Query projection: [batch, seq_len, num_q_heads * head_dim]
        q = self.q_proj(x)
        q = q.view(batch_size, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        # Shape: [batch, num_q_heads, seq_len, head_dim]

        if use_cache and self.kv_cache is not None:
            # Use cached keys and values for past tokens
            past_k, past_v = self.kv_cache
            # Compute new keys and values only for the current token(s)
            k_new = self.k_proj(x)
            # Shape: [batch, seq_len, num_groups * head_dim]            
            v_new = self.v_proj(x)
            # Shape: [batch, seq_len, num_groups * head_dim]
            # Reshape and transpose to group keys/values
            k_new = k_new.view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
            # Shape: [batch, num_groups, seq_len, head_dim]
            v_new = v_new.view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
            # Shape: [batch, num_groups, seq_len, head_dim]
            # Concatenate past and new keys/values along sequence dimension
            k = torch.cat([past_k, k_new], dim=2)  # [batch, num_groups, past_seq_len + seq_len, head_dim]
            v = torch.cat([past_v, v_new], dim=2)  # [batch, num_groups, past_seq_len + seq_len, head_dim]
        else:
            # Compute keys and values for all tokens (no cache or reset cache)
            k = self.k_proj(x)
            v = self.v_proj(x)
            k = k.view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
            # Shape: [batch, num_groups, seq_len, head_dim]

        # Update KV cache if enabled
        if use_cache:
            self.kv_cache = (k, v)

        # Expand keys and values to match the number of query heads per group
        # Repeat each group's K and V for all heads in that group
        group_idx = torch.arange(self.num_groups, device=x.device).repeat_interleave(self.num_heads_per_group)
        k_expanded = k[:, group_idx, :, :]  # [batch, num_q_heads, past_seq_len + seq_len, head_dim]
        v_expanded = v[:, group_idx, :, :]  # [batch, num_q_heads, past_seq_len + seq_len, head_dim]

        # Attention computation
        attn_scores = torch.matmul(q, k_expanded.transpose(-1, -2)) / (self.head_dim ** 0.5)
        # Shape: [batch, num_q_heads, seq_len, past_seq_len + seq_len]
        if past_seq_len > 0:
            # Mask out the future tokens in the attention scores
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(x.device)
            attn_scores = attn_scores.masked_fill(mask == 1, float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)
        # Shape: [batch, num_q_heads, seq_len, past_seq_len + seq_len]
        attn_output = torch.matmul(attn_probs, v_expanded)
        # Shape: [batch, num_q_heads, seq_len, head_dim]

        # Reshape and project back to embed_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_q_heads * self.head_dim)
        # Shape: [batch, seq_len, num_q_heads * head_dim]
        output = self.out_proj(attn_output)  # [batch, seq_len, embed_dim]

        return output

# Example usage
def test_gqa_with_kv_cache():
    # Hyperparameters
    embed_dim = 3584
    num_q_heads = 28
    num_groups = 4
    head_dim = 128
    batch_size = 1

    # Initialize model
    gqa = GroupedQueryAttention(embed_dim, num_q_heads, num_groups, head_dim)

    # Simulate autoregressive decoding
    # Step 1: Process first token
    x1 = torch.randn(batch_size, 1, embed_dim)  # First token
    output1 = gqa(x1, use_cache=True, past_seq_len=0)
    print("Output shape after first token:", output1.shape)  # [1, 1, 3584]
    print("KV cache shape:", [x.shape for x in gqa.kv_cache])  # [1, 4, 1, 128] for K and V

    # Step 2: Process second token with cache
    x2 = torch.randn(batch_size, 1, embed_dim)  # Second token
    output2 = gqa(x2, use_cache=True, past_seq_len=1)
    print("Output shape after second token:", output2.shape)  # [1, 1, 3584]
    print("KV cache shape:", [x.shape for x in gqa.kv_cache])  # [1, 4, 2, 128] for K and V

if __name__ == "__main__":
    test_gqa_with_kv_cache()