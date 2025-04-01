# GQA: Grouped Query Attention

GQA, or Grouped Query Attention, is an optimization technique used in large language models (LLMs) to improve the efficiency of the multi-head attention mechanism, which is a core component of transformer architectures. It builds on the standard multi-head attention (MHA) and multi-query attention (MQA) approaches, striking a balance between computational efficiency and model performance. Here’s a detailed explanation:

## Background: Multi-Head Attention (MHA)
In a standard transformer (e.g., as introduced in "Attention is All You Need"), multi-head attention computes attention scores using separate query (Q), key (K), and value (V) projections for each attention head. For a model with h heads:

- Each head has its own Q, K, and V matrices.
- Attention is computed as softmax(QK^T / sqrt(d_k))V per head, where d_k is the key dimension.
- This requires storing h sets of keys and values in the KV cache during inference, which can be memory-intensive, especially for long sequences.

<b>Downside</b>: The KV cache grows linearly with the number of heads (h) and sequence length, making it a bottleneck for memory and speed in large models.

## Multi-Query Attention (MQA)
MQA is a precursor to GQA. It reduces memory usage by:

- Using a single set of K and V projections shared across all heads.
- Keeping separate Q projections per head.
- Result: The KV cache size is reduced from h * sequence_length * d_k to 1 * sequence_length * d_k, improving inference speed and memory efficiency.

<b>Trade-off</b>: Sharing K and V across heads can reduce expressiveness, potentially hurting model quality, especially for tasks requiring fine-grained attention.

## Grouped Query Attention (GQA)
GQA is a middle ground between MHA and MQA. Instead of having one K and V for all heads (MQA) or one per head (MHA), GQA:

- Groups the heads into a smaller number of groups (e.g., g groups where g < h).
- Each group shares a single K and V projection, while Q remains unique per head.
- The number of groups (g) is a hyperparameter:
    - g = 1 is equivalent to MQA.
    - g = h is equivalent to MHA.

### How It Works
Suppose a model has h = 32 heads and g = 4 groups:
- The 32 heads are divided into 4 groups of 8 heads each.
- Each group uses one shared K and V, but each of the 8 heads in the group has its own Q.
- KV cache size is reduced to g * sequence_length * d_k (e.g., 4x smaller than MHA with 32 heads).

### Attention Computation
- For each group:
    - Compute attention using the shared K and V with the individual Q from each head in the group.
    - Attention = softmax(Q_i K_g^T / sqrt(d_k)) V_g, where i is the head index and g is the group index.
- Concatenate results across heads as in MHA.

### Benefits of GQA
1. Memory Efficiency: Reduces the KV cache size by a factor of h/g compared to MHA, making it more practical for long-context models or  resource-constrained environments.
2. Speed: Fewer unique K and V computations and smaller cache mean faster inference, especially on GPUs where memory bandwidth is a bottleneck.
3. Preserves Quality: Unlike MQA (which shares K and V across all heads), GQA retains more flexibility by allowing multiple groups, mitigating the accuracy loss seen in MQA.

### Trade-offs
- Hyperparameter Tuning: The number of groups (g) needs to be chosen carefully. Too few groups (e.g., g = 1, like MQA) may degrade performance; too many (e.g., g = h, like MHA) negate the efficiency gains.
- Complexity: Slightly more complex to implement than MQA, though still simpler than full MHA.

### Use in LLMs
GQA was popularized by models like Grok (from xAI) and has been adopted in other efficient LLMs (e.g., LLaMA variants, Mistral). For example:

- A 70B-parameter model with 64 heads might use g = 8 groups, reducing the KV cache size by 8x compared to MHA while maintaining near-MHA quality.
- It’s particularly useful for autoregressive decoding, where the KV cache dominates memory usage during generation.

## Code Example

Below is an example implementation of Grouped Query Attention (GQA) with a Key-Value (KV) cache in PyTorch. This code simulates a single attention layer with GQA, including the KV cache mechanism commonly used in autoregressive models (e.g., transformers for language generation) to avoid recomputing key and value vectors for past tokens.

Assumptions
- Embedding dimension: 3584 (from your previous example)
- Number of query heads: 28
- Number of groups: 4 (so 7 query heads per group)
- Key/Value dimension per group: 128
- Batch size: 1 (for simplicity)
- Sequence length: Variable (to demonstrate KV caching)
- KV cache stores past keys and values to enable efficient incremental decoding.

[Example Code](./code/GQA.py)

### Explanation of the Code
1. Model Structure:
    - q_proj: Linear layer projecting the input to 28 query heads (3584 → 28 × 128).
    - k_proj and v_proj: Linear layers projecting the input to 4 groups of keys/values (3584 → 4 × 128).
    - out_proj: Projects the concatenated attention output back to the embedding dimension.
2. KV Cache:
    - Stored as a tuple (past_k, past_v) in self.kv_cache.
    - When use_cache=True, the forward pass concatenates cached keys/values from past tokens with new keys/values for the current token(s).
    - Without cache (or on first step), keys/values are computed for all input tokens.
3. GQA Logic:
    - Queries are computed for all 28 heads.
    - Keys and values are computed for 4 groups, then expanded (via repeat_interleave) to align with the 28 query heads (7 heads per group share the same K/V).
    - Attention is computed as usual, with shared K/V within each group.