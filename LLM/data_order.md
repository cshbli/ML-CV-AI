# Multi-Dimensional Matrix Data Order

## Introduction to Matrix Data Order
In computing, multi-dimensional arrays (matrices, tensors) represent data with multiple axes (e.g., rows, columns, depth). The data order (or memory layout) defines how these elements are arranged in a contiguous block of memory. This affects performance (e.g., cache efficiency), interoperability between libraries, and how algorithms process the data.

Two fundamental concepts underpin data order:

- Row-Major Order: Elements are stored row-by-row (left-to-right, top-to-bottom).
- Column-Major Order: Elements are stored column-by-column (top-to-bottom, left-to-right).

Higher-dimensional arrays extend these principles, leading to formats like WHD, DWH, NCHW, etc.

## Key Data Order Formats
### WHD (Width-Height-Depth)
- Definition: Width (W) varies fastest, then Height (H), then Depth (D) varies slowest.
- Memory Layout: Row-major within each depth slice; depth is the outermost dimension.
- Example (2x3x2 matrix, W=2, H=3, D=2):

```
Depth 0: [[ 0,  1], [ 2,  3], [ 4,  5]]
Depth 1: [[ 6,  7], [ 8,  9], [10, 11]]
Memory: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
```
  - Indices: (d, h, w) → Memory offset = d * (H * W) + h * W + w.

- Use Case: Common in image processing where width and height represent spatial dimensions, and depth is channels (e.g., RGB).
- Intuition: Think of stacking 2D images (width × height) along a depth axis.

### DWH (Depth-Width-Height)
- Definition: Depth (D) varies fastest, then Width (W), then Height (H) varies slowest.
- Memory Layout: Depth slices are contiguous, followed by row-major width and height.
- Example (2x2x3 matrix, D=2, W=2, H=3):
```
Height 0: [[0, 2], [1, 3]]
Height 1: [[4, 6], [5, 7]]
Height 2: [[8, 10], [9, 11]]
Memory: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
```
- Indices: (h, w, d) → Memory offset = h * (W * D) + w * D + d.
- Use Case: Used in some deep learning frameworks or when depth (channels) is the primary focus.
- Intuition: Depth elements are grouped together, like interleaving RGB values per pixel.

### ONNX NCHW (Batch-Channel-Height-Width)
- Definition: Batch (N) varies slowest, then Channels (C), Height (H), and Width (W) varies fastest.
- Memory Layout: Row-major within each channel, with channels grouped by batch.
- Example (N=1, C=2, H=2, W=3):
```
Batch 0, Channel 0: [[0, 1, 2], [3, 4, 5]]
Batch 0, Channel 1: [[6, 7, 8], [9, 10, 11]]
Memory: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
```
- Indices: (n, c, h, w) → Offset = n * (C * H * W) + c * (H * W) + h * W + w.
- Use Case: Standard in ONNX (Open Neural Network Exchange) and frameworks like PyTorch for convolutional neural networks (CNNs).
- Why NCHW?
  - Channels (C) are grouped together, optimizing for convolution operations where filters slide over spatial dimensions (H, W).
  - Batch (N) is outermost for parallel processing of multiple samples.

### Python Default Data Order (NumPy)
- Definition: Python’s NumPy uses row-major order (C-style) by default.
- Memory Layout: The rightmost index varies fastest, and the leftmost varies slowest.
- Example (2x3 array):
```
import numpy as np
arr = np.array([[0, 1, 2], [3, 4, 5]])
# Memory: [0, 1, 2, 3, 4, 5]
```
- Indices: (i, j) → Offset = i * W + j.
- Higher Dimensions: For a 3D array (e.g., 2x3x4), it’s (D, H, W)-like:
```
arr = np.arange(24).reshape(2, 3, 4)
# Memory: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
```
- Use Case: Default for Python scientific computing; aligns with WHD-like ordering.
- Note: NumPy can use column-major (Fortran-style) with order='F', but it’s not default.

### Row-Major Order
- Definition: Elements are stored row-by-row; the rightmost index changes fastest.
- Languages: C, C++, Python (NumPy), Java use row-major by default.
- Example (2x3 matrix):
```
[[0, 1, 2],
 [3, 4, 5]]
Memory: [0, 1, 2, 3, 4, 5]
```
- Offset: For (i, j) in a 2D array of width W, offset = i * W + j.
- Advantages: Cache-friendly for row-wise access; intuitive for image-like data.

### Column-Major Order (Comparison)
- Definition: Elements are stored column-by-column; the leftmost index changes fastest.
- Languages: Fortran, MATLAB, R use column-major.
- Example (2x3 matrix):
```
[[0, 1, 2],
 [3, 4, 5]]
Memory: [0, 3, 1, 4, 2, 5]
```
- Offset: For (i, j) in a 2D array of height H, offset = j * H + i.
- Relevance: Less common in Python/ML but important for interoperability.

## Comparison Table

|Format|	Order (Slowest → Fastest)|	Example Memory Layout (2x2x3)|	Typical Use Case|
|---|---|---|---|
|WHD|	D → H → W	|[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]	|Image processing|
|DWH|	H → W → D	|[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]	|Channel-first processing|
|NCHW (ONNX)	|N → C → H → W	|[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]	|Deep learning (CNNs)|
|Python/NumPy	|D → H → W (row-major)	|[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]	|General scientific computing|
|Row-Major	|Rows → Columns	|[0, 1, 2, 3, 4, 5] (2D)	|C-style languages|
|Column-Major	|Columns → Rows	|[0, 3, 1, 4, 2, 5] (2D)	|Fortran-style languages|

## Implications and Practical Considerations
### Performance
- Cache Locality: Row-major (e.g., WHD, NCHW) benefits from spatial locality when accessing adjacent elements (e.g., pixels), common in CPUs/GPUs.
- DWH: Better for channel-wise operations but less cache-efficient for spatial traversals.
- NCHW vs. NHWC: NCHW (PyTorch default) optimizes for convolutions; NHWC (TensorFlow default) suits some hardware (e.g., GPUs with channel-last memory).

### Interoperability
- ONNX: Uses NCHW, so converting from NHWC (e.g., TensorFlow) requires transposition.
- Python/NumPy: Aligns with WHD-like row-major, making it natural for image data but requiring adjustment for NCHW in ML frameworks.

### Code Examples (Python/NumPy)
- WHD to NCHW Conversion:
```
import numpy as np
whd = np.arange(12).reshape(2, 3, 2)  # D=2, H=3, W=2
nchw = whd.transpose(0, 2, 1)[np.newaxis, ...]  # N=1, C=2, H=3, W=2
print(nchw.shape)  # (1, 2, 3, 2)
```

- DWH to NCHW:
```
dwh = np.arange(12).reshape(3, 2, 2)  # H=3, W=2, D=2
nchw = dwh.transpose(2, 0, 1)[np.newaxis, ...]  # N=1, C=2, H=3, W=2
```

## Context in Machine Learning
- NCHW in ONNX/PyTorch: Channels (C) are grouped for efficient convolution kernel application.
- NHWC in TensorFlow: Hardware optimizations (e.g., CUDA) sometimes favor channel-last.
- Conversion: Frameworks often provide utilities (e.g., torch.permute, tf.transpose) to switch orders.

## Conclusion
- WHD: Intuitive for images, row-major, Python-friendly.
- DWH: Channel-first, less common but useful in specific contexts.
- NCHW: ML standard (ONNX, PyTorch), optimized for CNNs.
- Python/NumPy: Row-major by default, aligns with WHD.
- Row-Major: Dominant in C-style ecosystems; column-major is its Fortran counterpart.

Understanding data order is critical for performance tuning, debugging, and framework interoperability. 

For Python ML, NCHW and NumPy’s row-major are most relevant, with conversions as needed.
