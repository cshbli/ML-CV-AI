# GGML GGUF

## GGML Tensor
- ggml.h
```
    // n-dimensional tensor
    struct ggml_tensor {
        enum ggml_type type;  // GGML_TYPE_F32     = 0, 
                              // GGML_TYPE_F16     = 1,        
                              // GGML_TYPE_Q8_0    = 8,

        struct ggml_backend_buffer * buffer;

        int64_t ne[GGML_MAX_DIMS]; // number of elements
        size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = ggml_type_size(type)
                                   // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                   // nb[i] = nb[i-1] * ne[i-1]

        // compute data
        enum ggml_op op;  // GGML_OP_NONE = 0,
                          // GGML_OP_ADD,
                          // GGML_OP_RMS_NORM,

        // op params - allocated as int32_t for alignment
        int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

        int32_t flags;

        struct ggml_tensor * src[GGML_MAX_SRC];

        // source tensor and offset for views
        struct ggml_tensor * view_src;
        size_t               view_offs;

        void * data;

        char name[GGML_MAX_NAME];

        void * extra; // extra things e.g. for ggml-cuda.cu

        char padding[8];
    };
```    

The `ggml_tensor` is the core data structure in GGML (the tensor library used by llama.cpp) that represents n-dimensional arrays of values for neural network operations. It serves as the fundamental building block for all computations in the library.

In llama.cpp, tensors are used to represent everything from model weights to activations during inference. The structure is designed to be flexible enough to support various operations while maintaining performance through careful memory management and support for hardware acceleration.

### Data Storage
- Stores values of different types (FP32, FP16, quantized formats, etc.)
- Supports up to 4 dimensions with configurable shapes
- Manages memory layout through strides (`nb`) for efficient operations

### Computational Graph Integration
- Records which operation created this tensor (`op`)
- Stores references to source tensors (`src`) used to create it
- Contains operation parameters used during computation
- Enables building a computational graph for inference and training

### Memory Optimization
- Supports views (references to subsections of other tensors)
- Can belong to different backend buffers for device-specific memory
- Contains metadata necessary for quantization and special operations

### Tensor Classification
- Flags mark tensors as inputs, outputs, parameters or loss values
- Contains name field for human-readable identification

## GGML Backend Buffer

- ggml-backend-impl.h

```
    struct ggml_backend_buffer {
        struct ggml_backend_buffer_i  iface;
        ggml_backend_buffer_type_t    buft;
        void * context;
        size_t size;
        enum ggml_backend_buffer_usage usage;   // GGML_BACKEND_BUFFER_USAGE_ANY = 0,
                                                // GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1,
                                                // GGML_BACKEND_BUFFER_USAGE_COMPUTE = 2,
    };
```    

The `ggml_backend_buffer` is a core component in GGML's hardware abstraction layer that manages memory for tensor operations across different computing devices. It has several key responsibilities:

### Memory Management for Different Devices
- Provides a unified interface for memory allocation on various backends (CPU, GPU, etc.)
- Handles device-specific memory characteristics and requirements
- Stores metadata about the memory block including size and usage pattern

### Tensor Operations
- Offers device-specific implementations for common tensor operations:
  - Initializing tensors within the buffer
  - Setting tensor data (copying from host to device)
  - Getting tensor data (copying from device to host)
  - Clearing memory
  - Copying between tensors (potentially across different backends)

### Buffer Abstraction Layer
- Creates a device-agnostic way to interact with memory
- Contains function pointers to backend-specific implementations
- Allows GGML to support heterogeneous computing environments

### Key Fields
- `iface`: Function pointers for buffer operations
- `buft`: The buffer type (defines allocation strategy and memory type)
- `context`: Implementation-specific data
- `size`: Total memory size

This abstraction enables GGML to work efficiently with different hardware like CPUs, GPUs, and potentially other accelerators while maintaining a consistent API.
