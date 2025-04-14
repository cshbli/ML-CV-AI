# llama.cpp Quantization

- <b>Bit Precision</b>: `Q8` > `Q6` > `Q5` > `Q4`. Lower bits mean less memory but more accuracy loss.
- <b>K-Quantization</b>: The `_K` suffix indicates advanced block-wise quantization with the K-means-like technique, which is more sophisticated than plain quantization (like `_0`). It groups weights and assigns scale factors per block, reducing quantization errors.
- <b>Variant Suffixes (`_S`, `_M`, `_L`)</b>: These tweak the block size or complexity of the K-quantization:
  - `_S` (small): More aggressive memory savings, potentially faster but less accurate.
  - `_M` (medium): Balanced approach.
  - `_L` (large): Larger blocks or more precision-preserving, at the cost of slightly more memory.

## Q8_0 Quantization

The Q8_0 quantization in the GGUF format, is a method to compress neural network weights from high-precision formats (e.g., 16-bit floating-point, FP16) to 8-bit integers (INT8: -128 to 127). This reduces model size and speeds up inference while maintaining near-original accuracy. The "Q8_0" designation indicates an 8-bit quantization with a straightforward, uniform scaling approach (no additional optimizations like K-quant methods). 

Q8_0 Specifics:

- Q: Quantized.
- 8: 8-bit integers (INT8, range: -128 to 127).
- _0: Uniform quantization with per-block scaling, no advanced clustering or asymmetric offsets (unlike K-quant methods like Q8_K).

### How GGUF Q8_0 Quantization Works

#### 1. Input Weights

The starting point is a model’s weights in FP16 or FP32 (common for pre-trained LLMs like Llama). These weights are typically floating-point numbers in a range like [-10.0, 10.0] (varies by layer).

#### 2. Block-Based Quantization

Q8_0 divides weights into small blocks (typically 32 weights per block in llama.cpp) to improve accuracy. Each block is quantized independently:
  - Local scaling within a block preserves precision better than global scaling across all weights, as weight distributions vary by layer or tensor.

#### 3. Scaling and Quantization

  For each block:
  - Find the Range:
    - Compute the absolute maximum value (max_abs) of the weights in the block:
    ```
    max_abs = max(|w_i|) for w_i in block
    ```
    Example: For weights [0.5, -1.2, 0.8, 1.0], max_abs = 1.2.
  - Compute Scale: The scale factor maps the floating-point range to the INT8 range (-128 to 127):
    ```
    scale = max_abs / 127
    ```
    Example: If max_abs = 1.2, scale = 1.2 / 127 ≈ 0.0094488.
  - Quantize Weights:
    - Convert each weight w_i to an 8-bit integer q_i:
    ```
    q_i = round(w_i / scale)
    ```
    - q_i is clamped to [-128, 127].
    - Example: For w_i = 0.5, q_i = round(0.5 / 0.0094488) ≈ 53.
    - The quantized block stores [q_1, q_2, ..., q_n] (n = block size, e.g., 32).
  - Store Scale:
  The scale is stored as a single FP16 or FP32 value per block to allow dequantization during inference.
  Example: For the block, store [53, -127, 85, 106, ...] (INT8) and scale = 0.0094488 (FP16).

#### 4. Dequantization During Inference
To use the model:

- For each block, reconstruct approximate floating-point weights:
```
w_i ≈ q_i * scale
```
Example: q_i = 53, scale = 0.0094488 → w_i ≈ 53 * 0.0094488 ≈ 0.5007864 (close to original 0.5).
- These weights are used in matrix operations (e.g., attention, feed-forward layers).

#### 5. Storage in GGUF
In a GGUF file (e.g., model-Q8_0.gguf):

- Tensor Data: Weights are stored as INT8 arrays, grouped by blocks.
- Scale Data: Each block’s scale factor is stored alongside the quantized weights.
- Metadata: Includes quantization type (Q8_0), block size (e.g., 32), and tensor shapes.
- Example tensor layout (simplified):
```
Tensor: blk.0.attn_q.weight
Shape: [4096, 4096]
Data: [INT8, INT8, ..., INT8]  # Quantized weights
Scales: [FP16, FP16, ...]     # One scale per 32 weights
```

### Summary
- Block Size: llama.cpp typically uses 32 weights per block for Q8_0, balancing precision and overhead. Each block requires 32 bytes (INT8) + 2 bytes (FP16 scale) = 34 bytes.
- Memory Savings:
  - FP16: 2 bytes per weight → 7B parameters ≈ 14 GB.
  - Q8_0: ~1 byte per weight (32 INT8 + 1 FP16 scale per 32 weights ≈ 1.0625 bytes/weight) → 7B parameters ≈ 7.44 GB.
- Quality: Q8_0 is nearly lossless because:
  - 8 bits provide 256 levels (-128 to 127), sufficient for most weight distributions.
  - Per-block scaling adapts to local ranges, unlike global quantization.
  - Typical perplexity loss vs. FP16 is <1% for LLMs like Llama.
