# llama.cpp Deep Dive

## 

```mermaid
flowchart TD
  start([start]) --> backend_load["ggml_backend_load_all"]
  backend_load --> model_load["llama_model_load_from_file"]
  model_load --> get_vocab["llama_model_get_vocab"]
  get_vocab --> tokenize["llama_tokenize"]
  tokenize --> default_context(["llama_context_default_params"])
  default_context --> sampler_chain_default("llama_sampler_chain_default_params")
  sampler_chain_default --> sampler_chain_init("llama_sampler_chain_init")
  sampler_chain_init --> sampler_chain_add("llama_sampler_chain_add")
  sampler_chain_add --> get_one_batch("llama_batch_get_one")
  get_one_batch --> loop_check{"Is n_pos + batch.n_tokens 
    < n_prompt + n_predict?"}

  subgraph main_loop["Main Loop"]
    loop_check -->|Yes| decode["Process batch with model:
    llama_decode(ctx, batch)"]
    decode -->|Success| update_pos["Update position: 
    n_pos += batch.n_tokens"]
    
    decode -->|Failure| error_decode["Log decode error"]
    error_decode --> exit([Exit with error])

    update_pos --> sample["Sample next token:
    new_token_id = llama_sampler_sample()"]
    
    sample --> check_eog{"Is token
    end-of-generation?"}
    
    check_eog -->|Yes| exit_loop([Exit loop])
    
    check_eog -->|No| convert["Convert token to text:
    llama_token_to_piece()"]
    
    convert -->|Success| print["Output text piece"]
    
    convert -->|Failure| error_convert["Log conversion error"]
    error_convert --> exit
    
    print --> prepare["Prepare the next batch with
    the sampled token<br>llama_batch_get_one"]
    
    prepare --> increment["Increment token counter:
    n_decode += 1"]
    
    increment --> loop_check

    loop_check -->|No| exit_loop
    
    exit_loop --> finish([End])
    
  end

  classDef process fill:#f9f,stroke:#333,stroke-width:2px;

  class decode process

```

## `ggml_backend_load_all()` in `ggml-backend-reg.cpp`

This function dynamically discovers and loads all available hardware acceleration backends for the GGML framework in llama.cpp, enabling optimal performance across different hardware.

```mermaid
flowchart TD
    start([ggml_backend_load_all]) --> load_from_path["Call ggml_backend_load_all_from_path(nullptr)"]
    
    load_from_path --> load_backends["Load multiple backend types in priority order"]
    
    load_backends --> blas["Try to load BLAS backend"]
    blas --> cann["Try to load CANN backend"]
    cann --> cuda["Try to load CUDA backend"]
    cuda --> hip["Try to load HIP backend"]
    hip --> kompute["Try to load Kompute backend"]
    kompute --> metal["Try to load Metal backend"]
    metal --> rpc["Try to load RPC backend"]
    rpc --> sycl["Try to load SYCL backend"]
    sycl --> vulkan["Try to load Vulkan backend"]
    vulkan --> opencl["Try to load OpenCL backend"] 
    opencl --> musa["Try to load MUSA backend"]
    musa --> cpu["Try to load CPU backend"]
    
    cpu --> check_env["Check GGML_BACKEND_PATH 
    environment variable"]
    
    check_env --> has_env{"ENV var set?"}
    has_env -->|Yes| load_custom["Load custom backend
    from GGML_BACKEND_PATH"]
    has_env -->|No| finish
    
    load_custom --> finish([End])
```

### Key Features

1. **Dynamic Backend Discovery**
   - Searches for backend libraries with naming patterns like:
     - Linux/macOS: `libggml-{type}-*.so`
     - Windows: `ggml-{type}-*.dll`

2. **Search Paths**
   - Executable directory
   - Current working directory
   - Custom path specified by parameter (when using `ggml_backend_load_all_from_path()`)

3. **Backend Selection Process**
   - Loads candidate libraries
   - Calls each backend's `ggml_backend_score()` function
   - Selects the backend with the highest score (best compatibility/performance)

   For example, when called with name = "cuda", it might find:
    ```
    ggml-cuda.dll           (base version)
    ggml-cuda-sm75.dll     (score: 75)
    ggml-cuda-sm86.dll     (score: 86)
    ```
  And would load the sm86 version as it has the highest score, indicating better performance for that GPU architecture.

  This allows for optimized versions of backends to be automatically selected based on the specific hardware capabilities of the system.

4. **Supported Backends**
   - **GPU**: CUDA (NVIDIA), Metal (Apple), Vulkan, OpenCL, HIP (AMD), CANN (Huawei)
   - **CPU**: BLAS, specialized CPU implementations
   - **Other**: RPC (remote), SYCL, Kompute, MUSA

5. **Custom Backend Support**
   - Checks `GGML_BACKEND_PATH` environment variable for custom backends

This function is crucial for llama.cpp's hardware acceleration capabilities, allowing it to adapt to the specific hardware available on the user's system.

## `llama_model_default_params()` in `llama-model.cpp`

This function creates and returns a default configuration structure for loading a language model in llama.cpp.

```mermaid
flowchart TD
    start([Start]) --> create["Create llama_model_params structure with defaults"]
    create --> check_metal{"Running with Metal?<br>(Apple GPU)"}
    check_metal -->|Yes| offload["Set n_gpu_layers = 999<br>(offload all layers to GPU)"]
    check_metal -->|No| keep["Keep n_gpu_layers = 0<br>(CPU only by default)"]
    offload --> return
    keep --> return
    return --> finish([Return structure])
```

## Default Parameter Values

The function initializes the following configuration parameters:

| Parameter | Default Value | Purpose |
|-----------|---------------|---------|
| `devices` | `nullptr` | No specific devices selected |
| `n_gpu_layers` | `0` (CPU) or `999` (Metal) | Number of layers to offload to GPU |
| `split_mode` | `LLAMA_SPLIT_MODE_LAYER` | How to split model across devices |
| `main_gpu` | `0` | Primary GPU device ID |
| `tensor_split` | `nullptr` | Custom tensor splitting ratios |
| `progress_callback` | `nullptr` | Function for loading progress updates |
| `progress_callback_user_data` | `nullptr` | User data for callbacks |
| `kv_overrides` | `nullptr` | Override model parameters |
| `vocab_only` | `false` | Load only vocabulary (not weights) |
| `use_mmap` | `true` | Use memory mapping for model loading |
| `use_mlock` | `false` | Lock model in RAM to prevent swapping |
| `check_tensors` | `false` | Validate tensor data during loading |

On Apple devices with Metal support, the function automatically configures GPU offloading for optimal performance, assuming Apple GPUs typically have sufficient VRAM.

## `llama_model_load_from_file()` in `llama.cpp`

This function loads an LLM (Large Language Model) from a GGUF file into memory and prepares it for inference.

```mermaid
flowchart TD
    start([llama_model_load_from_file]) --> create_model["Create new llama_model instance"]
    
    create_model --> has_devices{"params.devices<br>specified?"}
    
    has_devices -->|Yes| use_specified["Use specified devices"]
    has_devices -->|No| auto_detect["Detect available devices"]
    
    auto_detect --> collect_gpus["Find GPU devices"]
    collect_gpus --> collect_rpc["Find RPC backend devices"]
    collect_rpc --> combine["Combine devices (RPC first)"]
    
    use_specified --> check_split_mode
    combine --> check_split_mode{"Split mode?"}
    
    check_split_mode -->|NONE| use_main_gpu["Use only main_gpu device"]
    check_split_mode -->|LAYER| keep_all["Keep all devices"]
    
    use_main_gpu --> log_devices
    keep_all --> log_devices["Get device info (name, memory)"]
    
    log_devices --> load_arch["Load model architecture"]

    subgraph llama_model_load[llama_model_load]
    
    
    load_arch --> load_hparams["Load hyperparameters"]
    load_hparams --> load_vocab["Load vocabulary"]
    load_vocab --> load_stats["Load model stats"]
    
    load_stats --> vocab_only{"params.vocab_only?"}
    
    vocab_only -->|Yes| skip_tensors["Skip loading weights"]
    vocab_only -->|No| load_tensors["Load model tensors/weights"]
    
    skip_tensors --> check_status
    load_tensors --> check_status{"Check loading<br>status"}
    
    check_status -->|Success| return_model["Return model pointer"]
    check_status -->|Error| free_model["Free model"]
    check_status -->|Cancelled| free_model
    
    free_model --> return_null["Return nullptr"]
    end
```

### Key Operations

1. **Initialization**:
   - Takes a model path and parameter struct
   - Creates a model instance with specified parameters

2. **Device Selection**:
   - Uses explicitly provided devices or detects available hardware
   - Prioritizes GPU devices based on split mode setting
   - Handles RPC (remote procedure call) servers for distributed inference

3. **Loading Process**:
   - Architecture: Model structure definition
   - Hyperparameters: Configuration values like embedding size, layer count
   - Vocabulary: Tokenizer data for text conversion
   - Statistics: Memory usage, parameter counts
   - Tensors: Actual model weights (unless in vocab_only mode)

This is the primary entry point for loading models in llama.cpp and supports various configurations including memory mapping, tensor checking, and GPU offloading.

## `llama_tokenize()` in `llama-vocab.cpp`

The `llama_tokenize()` function converts natural language text into tokens (integer IDs) that can be processed by a language model.

```cpp
int32_t llama_tokenize(
    const struct llama_vocab * vocab,
                  const char * text,
                     int32_t   text_len,
                 llama_token * tokens,
                     int32_t   n_tokens_max,
                        bool   add_special,
                        bool   parse_special)
```

### Key Operations

1. **Text Preprocessing**:
   - Takes the input text and processes it according to the tokenizer's rules
   - Creates fragments for further processing

2. **Special Token Handling**:
   - If `parse_special` is `true`, identifies and separates special tokens like control tokens
   - Special tokens have direct token IDs without being broken down further

3. **Tokenization Algorithm**:
   - Applies the appropriate tokenization algorithm based on the vocabulary type:
     - **SPM (SentencePiece)**: Uses byte-pair encoding with unigram language model
     - **BPE (Byte-Pair Encoding)**: Merges common pairs of characters/subwords
     - **WPM (WordPiece)**: Similar to BPE but with different merging criteria
     - **UGM (Unigram Model)**: Optimizes word segmentation using Viterbi algorithm
     - **RWKV**: Custom tokenizer for RWKV models

4. **Special Token Addition**:
   - If `add_special` is `true`, adds appropriate special tokens:
     - BOS (beginning of sequence) token at the start if configured
     - EOS (end of sequence) token at the end if configured

5. **Output Handling**:
   - Returns the number of tokens if successful
   - Returns negative number if `n_tokens_max` is too small (indicates required buffer size)

### Example Flow

For a simple example with SPM tokenizer:
```
"Hello world" → (preprocess) → "▁Hello▁world" → (tokenize) → [1, 15043, 2787]
```

Where:
- `1` might be the BOS token (if `add_special` is true)
- `15043` and `2787` are the token IDs for "Hello" and "world" respectively

This function is a core component of the inference pipeline, bridging the gap between human-readable text and the numeric representation that language models process.

## `llama_context_default_params()` in `llama-context.cpp`

This function creates and returns a default configuration structure (`llama_context_params`) for language model inference in llama.cpp. It initializes sensible defaults that can later be customized by users.

```mermaid
flowchart TD
    start([Start]) --> create["Create new llama_context_params
    structure with default values"]
    create --> return["Return default parameters"]
```

### Key Parameter Groups

1. **Context Window Settings**
   - `n_ctx = 512`: Maximum context size (tokens)
   - `n_batch = 2048`: Batch size for token processing
   - `n_ubatch = 512`: Micro-batch size for efficiency
   - `n_seq_max = 1`: Maximum concurrent sequences

2. **Computational Resources**
   - `n_threads` and `n_threads_batch`: Default thread counts
   - `offload_kqv = true`: Offload computation to GPU if available
   - `flash_attn = false`: Optimized attention disabled by default

3. **Model Behavior Controls**
   - RoPE configuration parameters (`rope_freq_base`, `rope_freq_scale`)
   - YaRN parameters for extended context (`yarn_ext_factor`, etc.)
   - `defrag_thold = -1.0f`: KV cache defragmentation threshold

4. **Memory Optimization**
   - `type_k = GGML_TYPE_F16`: Half precision for key cache
   - `type_v = GGML_TYPE_F16`: Half precision for value cache

5. **Output Controls**
   - `logits_all = false`: Only compute logits for the last token
   - `embeddings = false`: Output logits, not embeddings
   - `pooling_type = UNSPECIFIED`: Inherit from model

These default parameters provide a reasonable starting point for language model inference, prioritizing memory efficiency while still allowing the model to function effectively.

## `llama_init_from_model()` in `llama-context.cpp`

This function creates a new inference context from a loaded model, setting up all resources needed for model execution.

```mermaid
flowchart TD
    start([Start]) --> validate_model["Check if model is valid"]
    
    validate_model -->|Invalid| error_model["Return NULL with error message"]
    validate_model -->|Valid| validate_params["Validate context parameters"]
    
    validate_params -->|Invalid| error_params["Return NULL with error message"]
    validate_params -->|Valid| check_compatibility["Check hardware/model compatibility"]
    
    check_compatibility --> adjust_flash_attn["Adjust flash attention settings if needed"]
    adjust_flash_attn --> create["Create new llama_context object"]    
    
    create --> init_backends["Initialize computation backends
    (GPU, ACCEL, CPU)"]

    subgraph llama_context["llama_context"]
    
      init_backends --> create_kv_cache["Create KV cache memory structure"]
      create_kv_cache --> setup_compute_buffers["Allocate computation buffers"]
      
      setup_compute_buffers --> setup_scheduler["Initialize computation scheduler"]
      setup_scheduler --> reserve_worst_case["Reserve memory for worst-case graph"]
      
      reserve_worst_case --> loginfo["Log memory usage and graph statistics"]
      loginfo --> return_ctx["Return context pointer"]
    end
    
    create -->|Exception| error_init["Log error and return NULL"]
    error_init --> return_null["Return NULL"]
    error_model --> return_null
    error_params --> return_null
    
```

### Key Operations

1. **Validation**
   - Ensures model pointer is not NULL
   - Checks batch size parameters are valid
   - Verifies context size parameters are valid

2. **Hardware Compatibility**
   - Disables flash attention for incompatible models (e.g., Grok)
   - Verifies key/value head dimensions for flash attention
   - Checks quantization compatibility with flash attention

3. **Resource Allocation**
   - Creates backend interfaces for GPU, CPU, and acceleration devices
   - Allocates memory for key-value cache
   - Sets up computation buffers and scheduler

4. **Configuration**
   - Configures RoPE parameters (frequency scaling, etc.)
   - Sets up threading and callbacks
   - Handles model type specific settings

5. **Performance Optimization**
   - Sets up pipeline parallelism when appropriate
   - Configures GPU offloading based on model size
   - Reserves memory for efficient graph execution

This function is the bridge between a loaded model and the ability to perform inference with it, handling all the necessary setup for efficient execution.

## `llama_sampler_chain_default_params()` in `llama.cpp`

This function initializes and returns a default configuration structure for text sampling in llama.cpp.

```cpp
struct llama_sampler_chain_params llama_sampler_chain_default_params() {
    struct llama_sampler_chain_params result = {
        /*.no_perf                     =*/ true,
    };

    return result;
}
```

### Purpose

The function creates default parameters for a "sampler chain" - a sequence of sampling strategies that determine how the next token is selected during text generation.

Currently, the structure has only one parameter:

- `no_perf = true`: Disables performance tracking for the sampling process by default

### Usage Context

In llama.cpp, token sampling is a critical part of text generation that determines how to select the next token from probability distributions. The sampler chain allows for multiple sampling methods to be applied in sequence, such as:

1. Temperature scaling
2. Top-K filtering
3. Top-P (nucleus) sampling
4. Repetition penalties
5. Frequency penalties

This function creates a starting point with conservative defaults that users can then customize for their specific generation needs.

# `llama_sampler_chain_init()` in `llama-sampling.cpp`

This function initializes a composite sampler chain that allows multiple sampling strategies to be applied sequentially during text generation.

```mermaid
flowchart TD
    start([Start]) --> create["Create new llama_sampler with
    sampler chain interface"]
    
    create --> allocate["Allocate llama_sampler_chain context:
    - params: User-provided parameters
    - samplers: Empty vector
    - t_sample_us: 0
    - n_sample: 0"]
    
    allocate --> return["Return sampler pointer"]
```

### Purpose

The sampler chain acts as a container and orchestrator for multiple text sampling strategies. During token generation, these strategies are applied in sequence, with each one modifying the token probability distribution before passing it to the next sampler.

## Key Components

1. **Interface Setup**:
   - Sets up function pointers for standard operations (accept, apply, reset, clone)
   - Establishes behavior for the chain as a whole

2. **Context Initialization**:
   - Creates an empty container for individual samplers
   - Sets up performance tracking (disabled by default)

3. **Chain Pattern Implementation**:
   - Forms the foundation of a chain-of-responsibility pattern
   - Each sampler in the chain modifies token probabilities in sequence

## Usage Flow

After initialization, specific samplers are added to the chain with `llama_sampler_chain_add()`. For example:

```cpp
sampler = llama_sampler_chain_init(params);
llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));     // Temperature
llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));      // Top-K filtering
llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1)); // Top-P sampling
```

This design makes sampling highly customizable through composition of different strategies.

## `llama_sampler_chain_add()` in `llama-sampling.cpp`

The `llama_sampler_chain_add()` function adds a sampler to a sampler chain, building a sequence of text generation sampling strategies.

```cpp
void llama_sampler_chain_add(struct llama_sampler * chain, struct llama_sampler * smpl) {
    auto * p = (llama_sampler_chain *) chain->ctx;
    p->samplers.push_back(smpl);
}
```

```mermaid
flowchart LR
    start([Start]) --> cast["Cast chain->ctx to<br>llama_sampler_chain*"]
    cast --> add["Add sampler to chain<br>p->samplers.push_back()"]
    add --> finish([End])
```

### Purpose

This function implements the "Chain of Responsibility" pattern for token sampling, allowing multiple sampling strategies to be combined in sequence during text generation. Each sampler in the chain modifies the token probability distribution before passing it to the next sampler.

### Parameters

- `chain`: A pointer to the sampler chain created with `llama_sampler_chain_init()`
- `smpl`: A pointer to the sampler to add to the chain

### Usage Example

```cpp
// Create a sampling chain
llama_sampler* chain = llama_sampler_chain_init(params);

// Add samplers in sequence (order matters)
llama_sampler_chain_add(chain, llama_sampler_init_temp(0.8f));      // Temperature
llama_sampler_chain_add(chain, llama_sampler_init_top_k(40));       // Top-K filtering
llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95f, 1)); // Top-P (nucleus)
llama_sampler_chain_add(chain, llama_sampler_init_min_p(0.05f, 1)); // Min-P filtering

// The chain ownership is transferred - individual samplers will be freed when the chain is freed
```

This design allows for highly customizable sampling strategies through composition, with each component focusing on a specific aspect of probability manipulation.

## `llama_batch_get_one()` in `llama-batch.cpp`

This function creates a minimal batch structure for token processing, designed for simple, single-sequence inference cases.

```mermaid
flowchart LR
    tokens[Token Array] --> batch[Create minimal batch]
    n_tokens[Token Count] --> batch
    batch --> return[Return batch structure]
```

### Purpose

`llama_batch_get_one()` is a utility function that quickly creates a basic `llama_batch` structure from an array of token IDs. It's the simplest way to prepare tokens for inference.

```cpp
struct llama_batch llama_batch_get_one(
             llama_token * tokens,
                 int32_t   n_tokens) {
    return {
        /*n_tokens       =*/ n_tokens,
        /*tokens         =*/ tokens,
        /*embd           =*/ nullptr,
        /*pos            =*/ nullptr,
        /*n_seq_id       =*/ nullptr,
        /*seq_id         =*/ nullptr,
        /*logits         =*/ nullptr,
    };
}
```

### Key Features

1. **Minimal Configuration**:
   - Sets only the token array and count
   - All other batch fields are set to `nullptr`

2. **Default Behavior** (when used without modification):
   - Positions will be assigned sequentially (0, 1, 2...)
   - All tokens will be treated as a single sequence
   - Only the final token will generate logits (probabilities for next token)
   - No embedding inputs are provided

This function provides a streamlined way to feed tokens into the model for the common case of processing a single sequence of text.

## `llama_token_to_piece()` in `llama-vocab.cpp`

The `llama_token_to_piece()` function converts a numeric token ID back into its corresponding text representation (called a "piece"). This function is crucial for detokenization — the process of converting the model's token IDs back into human-readable text.

```cpp
int32_t llama_token_to_piece(
    const struct llama_vocab * vocab, // Vocabulary containing token information
    llama_token token,                // The token ID to convert
    char * buf,                       // Output buffer for the text
    int32_t length,                   // Size of output buffer
    int32_t lstrip,                   // Number of leading spaces to strip
    bool special                      // Whether to allow special tokens
)
```

### Key Operations

1. **Special Token Filtering**: 
   - If `special=false` and the token is a special token (control or unknown), returns 0 (no output)
   - This allows the caller to control whether special tokens appear in the output

2. **Text Retrieval**:
   - Looks up the token's text representation from the vocabulary
   - Uses a cache if available for performance optimization

3. **Tokenizer-Specific Processing**:
   - **SPM/WPM/UGM**: Unescapes whitespace (converts "\xe2\x96\x81" back to actual space)
   - **BPE**: Handles decoding of escaped sequences
   - **RWKV**: Uses custom escaping format for tokens

4. **Buffer Management**:
   - Checks if the provided buffer is large enough for the result
   - Handles leading space removal via `lstrip` parameter
   - Returns negative value if buffer is too small (indicating required size)

### Return Value

- **Positive**: Number of bytes written to the buffer
- **Negative**: Required buffer size (as a negative number) if provided buffer is too small
- **Zero**: When the token produces no output (e.g., filtered special tokens)

This function is essential for converting token sequences generated by the model back into human-readable text during inference.

## Code Structure

```mermaid
graph TD
    A[llama_model] --> B[llama_hparams]
    A --> C[llama_model_loader]
    A --> D[llama_layer]
    A --> E[llama_layer_posnet]
    A --> F[llama_layer_convnext]
    
    D --> G[ggml_tensor]
    E --> G
    F --> G
    
    C --> G
    C -.->|loads| D
    C -.->|loads| E
    C -.->|loads| F
    C -.->|loads| B

    subgraph "Model Core"
        A
        B[llama_hparams<br/>Model hyperparameters]
    end

    subgraph "Layer Types"
        D[llama_layer<br/>Transformer layer]
        E[llama_layer_posnet<br/>Positional network]
        F[llama_layer_convnext<br/>ConvNeXt layer]
    end

    subgraph "Loading & Memory"
        C[llama_model_loader<br/>Loads model from files]
        G[ggml_tensor<br/>Tensor data structure]
    end
```


```mermaid
graph TD;
    A["llama_context::decode()<br> <sub>in llama-context.cpp"</sub>] --> B["llama_context::graph_build()<br> <sub>in llama-context.cpp"</sub>];
    B --> C["llama_context::graph_compute()<br> <sub>in llama-context.cpp"</sub>]

    subgraph "graph_build()"
        B -->|LLM Architecture|D{Which LLM?}
        D -->|DeepSeek|E[llm_build_deepseek<br> <sub>in llama-model.cpp</sub>]
        D -->|LLaMA|F[llm_build_llma<br> <sub>in llama-model.cpp</sub>]
        D -->|Qwen2|G[llm_build_qwen2<br> <sub>in llama-model.cpp<sub>]
        D -->|Qwen2VL|H[llm_buildqwen2vl<br> <sub>in llama-model.cpp<sub>]
    end    

    subgraph "graph_compute()"
        C --> I[ggml_backend_sched_graph_compute_async<br> <sub>in ggml-backend.cpp</sub]
        I --> J[ggml_backend_sched_alloc_graph<br> <sub>in ggml-backend.cpp</sub]
        J --> K[ggml_backend_sched_compute_splits<br> <sub>in ggml-backend.cpp</sub]
        K --> L[ggml_backend_graph_compute_async<br> <sub>in ggml-backend.cpp</sub>]
        L --> O[backend->iface.graph_compute]
    end    

    subgraph "ggml_backend_sched_alloc_graph"
        M[ggml_backend_sched_split_graph] --> N[ggml_backend_sched_alloc_splits]
    end
```

### llama_context::decode() in llama-context.cpp

- Input Processing
- KV Cache Management
- Inference Pipeline
    - <b>graph_build()</b>
    - <b>graph_compute()</b>
- Output Generation
- Preformance Optimization

### llama_model::build_graph() in llama-model.cpp

The `llama_model::build_graph()` function is a crucial component in llama.cpp that constructs the computational graph needed for inferencing with the language model. Here's what it does:

1. **Dynamic Graph Construction** - Creates a computational graph tailored to a specific model architecture (Llama, Mamba, RWKV, etc.)

2. **Architecture-Specific Builder Selection** - Uses a factory pattern to select the appropriate builder class based on the model's architecture type:
   ```cpp
   switch (arch) {
       case LLM_ARCH_LLAMA:
           llm = std::make_unique<llm_build_llama>(*this, params, gf);
           break;
       case LLM_ARCH_MAMBA:
           llm = std::make_unique<llm_build_mamba>(*this, params, gf);
           break;
       // Many more architectures...
   }
   ```

3. **Graph Building** - When a builder is instantiated, its constructor automatically:
   - Creates the complete neural network flow
   - Adds all tensor operations to the graph (`gf`)
   - Sets up the connections between layers
   - Configures attention mechanisms, feed-forward networks, etc.

4. **Pooling Layer Addition** - After the main architecture is built, it adds any required pooling layer:
   ```cpp
   llm->build_pooling(gf, cls, cls_b, cls_out, cls_out_b);
   ```

5. **Result Production** - Returns a pointer to the graph result, which contains:
   - `t_logits` - The tensor containing the model's output logits
   - `t_embd` - The tensor containing the model's embedding output

This function enables llama.cpp to handle many model architectures (70+ in the code) through a uniform interface while applying architecture-specific optimizations and computational patterns.

### `llm_build_llama` - The Core Graph Builder for Llama Models

`llm_build_llama` is a crucial class in llama.cpp that constructs the computational graph for Llama-architecture models. It inherits from `llm_graph_context` and is responsible for translating the model's weights and architecture into a computational graph that can be executed by the GGML backend.

1. **Graph Construction**: It builds a complete forward pass graph for Llama models by connecting tensor operations in the proper sequence.

2. **Input Processing**:
   - Sets up token embeddings
   - Handles positional information
   - Prepares KV caching for efficient inference

3. **Layer Processing**: For each transformer layer, it builds:
   - RMS normalization
   - Self-attention mechanism with RoPE (Rotary Position Embeddings)
   - Feed-forward network with SwiGLU activation
   - Residual connections

4. **Attention Mechanism**:
   - Calculates Query, Key, and Value matrices
   - Applies RoPE to handle positional information
   - Builds the appropriate attention pattern with proper scaling
   - Handles multi-query attention patterns

5. **MoE Support**: For mixture-of-experts models, it constructs:
   - Expert routing gates
   - Multiple FFN experts
   - Expert combination logic

6. **Output Processing**:
   - Final normalization
   - Projection to vocabulary logits
   - Applies any final scaling needed

#### Implementation Details:

The class follows a builder pattern, constructing the graph incrementally through operations on GGML tensors. It uses callbacks to mark important tensors in the graph, which enables debugging and optimization. The core of the implementation is the layer-by-layer construction, with careful attention to architecture-specific details like scaling factors and normalization types.

This builder is specialized for Llama-style models but shares patterns with other architecture builders in the codebase.

### `graph_compute()` Function in llama-context.cpp

The `graph_compute()` method in `llama_context` is responsible for executing the computational graph built for inferencing. It's a crucial part of the execution pipeline that actually runs the model computations.

1. **Thread Management**   
2. **CPU Backend Configuration**
3. **Multi-Backend Thread Configuration**
4. **Asynchronous Graph Execution**:
   ```cpp
   auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
   ```
   - Submits the computational graph to the scheduler for asynchronous execution
   - This allows overlapping of I/O and computation for better performance

This function represents the point where all the tensor operations defined in the graph are actually executed on the hardware, whether that's CPU, GPU, or other accelerators.

### `ggml_backend_sched_alloc_graph()` Function in ggml-backend.cpp

This function is responsible for memory allocation for computational graphs in llama.cpp's heterogeneous backend system. It's a critical component that prepares the memory needed for model inference across different hardware devices.

1. **Graph Splitting**:
   ```cpp
   ggml_backend_sched_split_graph(sched, graph);
   ```
   - Divides the computation graph into subgraphs that can execute efficiently on different backends (CPU, CUDA, Metal, etc.)
   - Assigns each tensor and operation to the optimal backend
   - Creates necessary copies for tensors that need to be shared between backends
   - Builds connection points between subgraphs

2. **Memory Allocation**:
   ```cpp
   if (!ggml_backend_sched_alloc_splits(sched)) {
       return false;
   }
   ```
   - Allocates actual memory for each tensor in the computational graph
   - Tries to reuse previous allocations when possible
   - Creates new allocations when backend assignments have changed
   - Ensures tensors that need to communicate are properly aligned and accessible
   - Optimizes memory usage across all available backends

This function helps llama.cpp efficiently utilize multiple computing devices simultaneously by handling all the complex memory management required for heterogeneous computing. It's called prior to inference to ensure all memory is properly set up before computation begins.

### `ggml_backend_sched_compute_splits()` 

This function is critical for executing the computational graph after it has been split across multiple backends (like CPU, CUDA, Metal, etc.). Here's what it does:

1. **Execute Each Graph Split**: 
   - Processes each split of the computational graph that was previously divided by `ggml_backend_sched_split_graph()`
   - Each split runs on its designated backend (CPU, GPU, etc.)

2. **Handle Cross-Backend Transfers**:   

3. **Manage Synchronization**:   

4. **Support Async Execution**:   

5. **Pipeline Parallelism**:   

6. **Callback Integration**:   

This function is the execution engine that makes llama.cpp's multi-backend architecture work. It enables efficient utilization of heterogeneous hardware by coordinating tensor movement and computation across different devices while maintaining correct execution order and data consistency.

### `ggml_backend_graph_compute_async()` Function in ggml-backend.cpp

This function is a critical component in the GGML backend system that handles asynchronous execution of computational graphs across different hardware accelerators. Here's what it does:

```cpp
enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    return backend->iface.graph_compute(backend, cgraph);
}
```

1. **Dispatches Computation**: Forwards the graph computation request to the appropriate backend implementation (CPU, CUDA, Metal, etc.)

2. **Non-Blocking Operation**: Unlike its synchronous counterpart, it launches computation but returns immediately without waiting for completion

3. **Backend-Specific Execution**: Each backend implements its own version of `graph_compute()` that knows how to:
   - Schedule operations on the specific hardware
   - Manage memory transfers
   - Optimize for the particular accelerator

4. **Status Reporting**: Returns an enumerated status value indicating success or specific failure modes

#### Key Differences from Synchronous Version:

For comparison, the synchronous version is:

```cpp
enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}
```

The asynchronous version enables overlapping computation with other operations, which is crucial for maximizing performance in complex applications like LLM inference where you might want to process the next batch of tokens while still generating the current ones.

This function is part of the backend abstraction layer that allows llama.cpp to run efficiently across diverse hardware platforms while maintaining a unified programming interface.

## Llama Archtiecture

Llama.cpp’s backbone is the original Llama models, which is also based on the transformer architecture.

<img src="llama_architecture.avif">

The main difference between the LLaMa architecture and the transformers’:

- <b>Pre-normalization (GPT3):</b> used to improve the training stability by normalizing the input of each transformer sub-layer using the RMSNorm approach instead of normalizing the output.
- <b>SwigGLU activation function (PaLM):</b> the original non-linearity ReLU activation function is replaced by the SwiGLU activation function, which leads to performance improvements.
- <b>Rotary embeddings (GPTNeao):</b> the rotary positional embeddings (RoPE) was added at each layer of the network after removing the absolute positional embeddings.

### Decoder-Only Architecture

A decoder-only architecture refers to a type of transformer model that consists solely of the decoder component from the original transformer framework, omitting the encoder entirely. This design is tailored for tasks where the model generates output autoregressively—predicting the next token in a sequence based on all previous tokens—without requiring a separate input encoding phase. It’s widely used in large language models (LLMs) like GPT, LLaMA, and DeepSeek-R1.

<img src="./images/f6133c18-bfaf-4578-8c5a-e5ac7809f65b_1632x784.png">

#### The Original Transformer Context
The transformer architecture, introduced in "Attention is All You Need" (Vaswani et al., 2017), has two main components:

- Encoder: Processes the input sequence (e.g., a sentence to translate) into a contextualized representation. It uses bidirectional self-attention, allowing each token to attend to all tokens in the input.
- Decoder: Generates the output sequence (e.g., the translated sentence) autoregressively, using causal (masked) self-attention to ensure each token only attends to previous positions, plus cross-attention to the encoder’s output.

In tasks like machine translation, the encoder-decoder setup excels because the encoder captures the full input context, and the decoder generates the output step-by-step.

## High Level Flow

At its core, an LLM only predicts a single token each time. The generation of a complete sentence (or more) is achieved by repeatedly applying the LLM model to the same prompt, with the previous output tokens appended to the prompt. This type of model is referred to as an autoregressive model.

<img src="./images/llama_cpp_high_level_flow.png.webp">

Following the diagram, the flow is as follows:

1. The `tokenizer` splits the prompt into a list of `tokens`. Some words may be split into multiple tokens, based on the model’s `vocabulary`. Each token is represented by a unique number.
2. Each numerical token is converted into an `embedding`. An embedding is a vector of fixed size that represents the token in a way that is more efficient for the LLM to process. All the embeddings together form an `embedding` matrix.
3. The embedding matrix serves as the input to the `Transformer`. The Transformer is a neural network that acts as the core of the LLM. The Transformer consists of a chain of multiple layers. Each layer takes an input matrix and performs various mathematical operations on it using the model parameters, the most notable being the self-attention mechanism. The layer’s output is used as the next layer’s input.
4. A final neural network converts the output of the Transformer into `logits`. Each possible next token has a corresponding logit, which represents the probability that the token is the “correct” continuation of the sentence.
5. One of several `sampling` techniques is used to choose the next token from the list of logits.
6. The chosen token is returned as the output. To continue generating tokens, the chosen token is appended to the list of tokens from step (1), and the process is repeated. This can be continued until the desired number of tokens is generated, or the LLM emits a special end-of-stream (EOS) token.

## Tensors

It is important to distinguish between two types of tensors. 
- There are tensors that hold actual data, containing a multi-dimensional array of numbers. 
- On the other hand, there are tensors that only represent the result of a computation between one or more other tensors, and do not hold data until actually computed.

```
    // ggml.h
    // n-dimensional tensor
    struct ggml_tensor {
        enum ggml_type type;

        struct ggml_backend_buffer * buffer;

        int64_t ne[GGML_MAX_DIMS]; // number of elements
        size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = ggml_type_size(type)
                                   // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                   // nb[i] = nb[i-1] * ne[i-1]

        // compute data
        enum ggml_op op;

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

`nb` is a bit more sophisticated. It contains the stride: the number of bytes between consequetive elements in each dimension. In the first dimension this will be the size of the primitive element. In the second dimension it will be the row size times the size of an element, and so on. For example, for a 4x3x2 tensor:

<img src="./images/llama_cpp_tensor_stride.png.webp">

The purpose of using a stride is to allow certain tensor operations to be performed without copying any data. For example, the transpose operation on a two-dimensional that turns rows into columns can be carried out by just flipping `ne` and `nb` and pointing to the same underlying data:

```
// ggml.c
// ggml_transpose

struct ggml_tensor * ggml_transpose(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    struct ggml_tensor * result = ggml_view_tensor(ctx, a);
    ggml_format_name(result, "%s (transposed)", a->name);

    result->ne[0] = a->ne[1];
    result->ne[1] = a->ne[0];

    result->nb[0] = a->nb[1];
    result->nb[1] = a->nb[0];

    result->op     = GGML_OP_TRANSPOSE;
    result->src[0] = a;

    return result;
}
```

In the above function, <b>result</b> is a new tensor initialized to point to the same multi-dimensional array of numbers as the source tensor <b>a</b>. By exchanging the dimensions in <b>ne</b> and the strides in <b>nb</b>, it performs the transpose operation without copying any data.

### Tensor operations and views

As mentioned before, some tensors hold data, while others represent the theoretical result of an operation between other tensors. Going back to `struct ggml_tensor`:

- `op` may be any supported operation between tensors. Setting it to `GGML_OP_NONE` marks that the tensor holds data. Other values can mark an operation. For example, `GGML_OP_MUL_MAT` means that this tensor does not hold data, but only represents the result of matrix multiplication between two other tensors.
- `src` is an array of pointers to the tensors between which the operation is to be taken. For example, if `op == GGML_OP_MUL_MAT`, then `src` will contain pointers to the two tensors to be multiplied. If `op == GGML_OP_NONE`, then `src` will be empty.
- `data` points to the actual tensor’s data, or `NULL` if this tensor is an operation. It may also point to another tensor’s data, and then it’s known as a view. For example, in the `ggml_transpose()` function above, the resulting tensor is a view of the original, just with flipped dimensions and strides. `data` points to the same location in memory.

The matrix multiplication function illustrates these concepts well:

```
// ggml.c
struct ggml_tensor * ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}
```

In the above function, `result` does not contain any data. It is merely a representation of the theoretical result of multiplying `a` and `b`.

### Computing tensors
The `ggml_mul_mat()` function above, or any other tensor operation, does not calculate anything but just prepares the tensors for the operation. A different way to look at it is that it builds up a computation graph where each tensor operation is a node, and the operation’s sources are the node’s children. In the matrix multiplication scenario, the graph has a parent node with operation `GGML_OP_MUL_MAT`, along with two children.

The computation graph is run using `ggml_graph_compute()`, which runs `ggml_compute_forward()` on each node in a depth-first order. `ggml_compute_forward()` does the heavy lifting of calculations. It performs the mathetmatical operation and fills the tensor’s data pointer with the result.

|function| |source location|
|---|---|---|
|ggml_graph_compute() ||ggml_cpu.c|
|ggml_metal_graph_compute() || ggml_metal.m|
|ggml_compute_forward()  | |ggml_cpu.c|
||ggml_compute_forward_soft_max|ggml_cpu.c|
|ggml_cuda_compute_forward() || ggml-cuda.cu|
||ggml_cuda_op_soft_max|softmax.cu|


## Computation Acceleration

|Backend|Supported Platforms|Notes|
|---|---|---|
|CPU|All|mutli-threading and SIMD(AVX, AVX2, AVX512, AMX)|
|CUDA|NVIDIA GPUs|Compute Unified Device Architecture|
|HIP|AMD GPUs|Heterogeneous-Interface Parallel Programming|
|Metal|Apple GPUs|
|CANN|Huawei Ascend AI processors|Compute Architecture for Neural Networks|
|MUSA|Moore Threads MTT GPU|Multiverse Unified System Architecture|
|Vulkan|Cross-platform (NVIDIA, AMD, Apple, Intel) |Low-level API for rendering and compute|
| |Cross-platform (Windows, Linux, MacOS, Android) |On platform like Android, it is the de facto standard for GPU computing|
|Kompute|High-leve framework for <b>Vulkan</b>-based GPU computing|
|OpenCL|Cross-platform (NVIDIA, AMD, Intel) |Enables acceleration on devices that don't support CUDA or Metal|
|SYCL|C++ for Heterogeneous computing (Intel GPUs)|high-level programming model builds on the foundation of <b>OpenCL</b>|
|BLAS|Cross-platform math libraries for matrix operations|Basic Linear Algebra Subprograms|
| | Intel MKL: Intel Math Kernel Library| using AVX/AVX2/AVX512 instructions for SIMD|
| | OpenBLAS: Optimized BLAS library| Open-source, optimized for various CPUs (Intel, AMD, ARM)|
| | cuBLAS: NVIDIA CUDA BLAS library| NVIDIA GPUs|
| | rocBLAS: AMD ROC BLAS library| AMD GPUs using the HIP platform|
|BLIS|BLAS-like library for high-performance dense linear algebra|
|RPC|Distributed or remote computing|remote procedure call|

### AVX/AVX2/AVX-512/AMX Comparison Table

- Advanced Vector Extensions (AVX) instruction set
- Advanced Matrix Extensions (AMX). Intel-specific extension. 
- AMD CPUs also support AVX/AVX2/AVX-512

|Feature|	AVX|	AVX2|	AVX-512|	AMX|
|---|---|---|---|---|
|Bit-width|	256 bits|	256 bits|	512 bits|	Tile-based (Matrix ops)|
|Registers|	YMM|	YMM|	ZMM|	Tile registers|
|Focus|	Floating-point|	Floating-point & integers|	Vector ops, AI, HPC|	Matrix multiplication|
|Performance|	Moderate|	Higher|	Very high|	Specialized for AI/ML|
|Applications|	Multimedia, HPC|	HPC, ML, video encoding|	AI, scientific computing|	AI, deep learning|

#### Neon Instruction Set

The `NEON instruction set` is an advanced SIMD (Single Instruction, Multiple Data) extension to the ARM architecture, designed to accelerate parallel processing tasks like multimedia, signal processing, and machine learning on ARM-based CPUs. Introduced with ARMv7-A in 2005 and enhanced in later versions (e.g., ARMv8-A in the M3 chip), NEON is ARM’s equivalent to Intel’s AVX or SSE, enabling CPUs to perform the same operation on multiple data elements simultaneously.

Example:
```
#include <arm_neon.h>
#include <stdio.h>

void addFloats(float* a, float* b, float* result, int n) {
    for (int i = 0; i < n; i += 4) {
        // Load 4 floats from each array into 128-bit registers
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        // Add them in parallel
        float32x4_t vr = vaddq_f32(va, vb);
        // Store result
        vst1q_f32(&result[i], vr);
    }
}

int main() {
    float a[4] = {1.0, 2.0, 3.0, 4.0};
    float b[4] = {5.0, 6.0, 7.0, 8.0};
    float result[4];
    addFloats(a, b, result, 4);
    for (int i = 0; i < 4; i++) {
        printf("%f ", result[i]); // Outputs: 6.0 8.0 10.0 12.0
    }
    return 0;
}
```

- Compile: `gcc -o neon_add neon_add.c -march=armv8-a+simd` (on ARM64).
- How It Works: `vld1q_f32` loads 4 floats, `vaddq_f32` adds them, and  `vst1q_f32` stores the result—all in one cycle per vector.

#### NEON vs. AVX

- NEON: Leaner, power-efficient, ubiquitous in ARM (phones, Apple Silicon).
- AVX: Wider registers, more complex ops, higher power draw.

|Feature|	NEON (ARM)	| AVX (Intel)|
|---|---|---|
|Register Size|	128-bit	|256-bit (AVX), 512-bit (AVX-512)
|Introduced	|2005 (ARMv7-A)|	2011 (Sandy Bridge)
|Scope	|ARM CPUs (e.g., M3)|	x86 CPUs (e.g., i9)
|Instructions|	Broad SIMD ops	|Broader + FMA, AVX-512 adds more
|Use Case|	Mobile, embedded, ML	|Desktop, server, HPC

### Key Differences Between Metal and CoreML

|Feature|	Metal|	CoreML|
|---|---|---|
|Purpose|	High-performance graphics and GPU compute API.|	Framework for integrating and running machine learning models.|
|Focus Area|	Rendering and compute tasks for games, AR, and GPU workloads.|	On-device inference for machine learning models.|
|Target Developer|	Game developers, graphics/rendering experts, GPU compute developers.|	App developers incorporating AI/ML features.|
|Device Usage|	Optimizes GPU for graphics or parallel computations.|	Leverages CPU, GPU, and Neural Engine for ML inference.|
|Example Application|	High-performance 3D games, ARKit apps, or video editing.|	Object detection, text classification, or image processing in apps.

#### MLX
- MLX is a NumPy-like array framework designed for efficient and flexible machine learning on Apple silicon, brought to you by Apple machine learning research.

- The Python API closely follows NumPy with a few exceptions. MLX also has a fully featured C++ API which closely follows the Python API.

### iOS Inference Framework Comparison Table

|Feature|	CoreML|	TensorFlow Lite (TFLite)|	ONNX Runtime|
|---|---|---|---|
|Native to iOS|	Yes|	No|	No|
|Cross-Platform|	No|	Yes|	Yes|
|Performance on iOS|	Excellent (fully optimized).|	Good (with Core ML/GPU delegates).|	Moderate (less optimized).|
|Ease of Integration|	Best for iOS developers.|	Requires TensorFlow Lite APIs.|	Requires ONNX Runtime setup.|
|Hardware Utilization|	Fully optimized (Neural Engine, GPU).|	Can use GPU/CoreML delegate.|	Less efficient without conversion.|
|Model Format Support|	Only CoreML format.|	TensorFlow and TFLite formats.|	Any framework supporting ONNX.|
|Workflow Simplicity|	Simple (Xcode and Swift integration).|	Moderate (requires extra setup).|	More complex (manual tuning needed).|

### Android Inference Framework Comparison Table

|Feature|	TensorFlow Lite|	ONNX Runtime|	ExecuTorch|	Others (MNN/NCNN)|
|---|---|---|---|---|
|Optimization for Android|	Excellent (designed for mobile).|	Good (general-purpose, NNAPI support).|	Moderate (not as optimized).|	Excellent (mobile-first frameworks).|
|Hardware Acceleration|	NNAPI, GPU Delegate, Hexagon DSP.|	NNAPI, GPU Delegate.|	NNAPI, GPU Delegate.|	NNAPI, GPU (varies by framework).|
|Model Compatibility|	Best with TensorFlow-trained models.|	Supports PyTorch, TensorFlow, etc.|	Best with PyTorch-trained models.|	Limited (specific use cases).|
|Ease of Use|	Simple for TensorFlow users.|	Requires ONNX conversion.|	Simple for PyTorch users.|	Moderate to complex.|
|Cross-Platform|	Yes (Android, iOS, embedded).|	Yes (Android, iOS, Linux, Windows).|	Yes (Android, iOS, Linux, Mac, embedded).|	Primarily Android (some iOS support).|
|Binary Size|	Smallest.|	Moderate.|	Larger.|	Very small.|

- MNN (Alibaba)
- NCNN (Tencent)

## Build a project outside of the source tree
- Please see this [example](https://github.com/ggerganov/llama.cpp/tree/master/examples/simple-cmake-pkg)

## What is a layer?

### Transformer Architecture Basics
LLaMA (Large Language Model Meta AI) follows the standard transformer architecture, which consists of multiple stacked "layers." Each layer in a transformer model is typically a <b>transformer block</b> that processes input data sequentially. For LLaMA, which is a decoder-only model (like GPT), these layers are responsible for generating or understanding text by applying a series of computations.

A single transformer layer (or block) in LLaMA generally includes:

- <b>Multi-Head Self-Attention</b>: This mechanism allows the model to weigh the importance of different tokens in the input sequence relative to each other.
- <b>Feed-Forward Neural Network (FFN)</b>: A fully connected network applied independently to each token’s representation, typically with a hidden size larger than the input/output size (e.g., 4x the model dimension in LLaMA).
- <b>Normalization Layers</b>: Layer normalization (e.g., RMSNorm in LLaMA) is applied before or after the attention and FFN components to stabilize training and inference.
- <b>Residual Connections</b>: These add the input of the layer to its output, helping with gradient flow during training.

Each layer transforms the input embeddings (or hidden states) and passes them to the next layer, progressively refining the representation of the text.

### Layers in llama.cpp
llama.cpp is a C++ implementation of LLaMA designed for efficient inference on CPUs (and optionally GPUs via extensions). In this codebase, a "layer" corresponds to one of these transformer blocks, and the model is composed of multiple such layers stacked together. The exact number of layers depends on the specific LLaMA model variant (e.g., 7B, 13B, 70B), with larger models having more layers.

For example:

- LLaMA 7B: 32 layers
- LLaMA 13B: 40 layers
- LLaMA 70B: 80 layers

In the llama.cpp source code (e.g., llama.h or llama.cpp), layers are represented structurally. The model weights are organized by layer, with each layer containing weights for:

- Self-attention (query, key, value, and output projection matrices: wq, wk, wv, wo).
- Feed-forward network (e.g., w1, w2, w3 for the SwiGLU activation in LLaMA).
- Normalization parameters (e.g., RMSNorm weights).
When llama.cpp performs inference, it processes the input token embeddings through each layer sequentially, applying the attention and FFN computations as defined by the transformer block.

### What Does "Layer" Mean Practically in llama.cpp?

- <b>Computationally</b>: A layer is a unit of processing. During inference, llama.cpp iterates over all layers (e.g., 32 times for LLaMA 7B) to compute the output for a given input.
- <b>Memory</b>: Each layer has associated weights, and the total memory footprint of the model scales with the number of layers and their size (determined by the hidden dimension and number of attention heads).
- <b>Quantization</b>: When quantizing models in llama.cpp (e.g., to 4-bit or 8-bit precision), the weights of each layer are quantized individually, which can affect performance and memory usage.
- <b>Parallelism</b>: Features like layer offloading (to GPU) or splitting layers across multiple threads rely on treating layers as discrete units.

### Example in Context
If you see a log or configuration in llama.cpp mentioning "layers," it might refer to:

- How many transformer blocks are loaded into memory (e.g., --n-layers for offloading).
- The progress of computation (e.g., "processing layer 10 of 32").

### Summary
In llama.cpp, a layer is a single transformer block in the LLaMA model, consisting of self-attention, a feed-forward network, and normalization, with associated weights. The model’s depth (number of layers) defines its capacity, and llama.cpp processes these layers iteratively during inference. 

## Layer Offloading

Llama.cpp allows you to specify how many model layers (e.g., transformer layers in an LLM) are offloaded to the GPU, with the remainder staying on the CPU.

- Default Behavior: Without GPU support compiled in or if `-ngl` is set to 0, all computations run on the CPU using optimized tensor operations (via the GGML library). With GPU support enabled, you can offload some or all layers to the GPU, depending on VRAM capacity and your settings.

### How Layers Are Distributed
1. User Specification:

    - You explicitly set the number of layers to offload. For example, `-ngl 32` offloads 32 layers to the GPU, and any additional layers stay on the CPU.
    - If you set `-ngl` higher than the model’s total layers (e.g., `-ngl 100` for a 33-layer model like LLaMA 7B), it offloads all layers to the GPU, assuming VRAM permits.
    - Setting `-ngl -1` in some contexts (like Python bindings) attempts to offload all layers automatically.

2. Model Architecture:

    - LLMs like LLaMA consist of stacked transformer layers (e.g., 32 for 7B, 40 for 13B). Llama.cpp offloads these layers sequentially from the bottom up. So, if you offload 20 out of 32 layers, the first 20 run on the GPU, and the last 12 run on the CPU.
    - Key-value caches (used for context in generation) can also be offloaded alongside layers, increasing VRAM usage.

3. VRAM Management:

    - The GPU handles the offloaded layers’ computations and stores their weights in VRAM. If VRAM is insufficient, you’ll encounter errors unless you reduce -ngl to fit within your GPU’s memory (e.g., a 7B model in 4-bit quantization needs ~6-8 GB VRAM fully offloaded).
    - Remaining layers and their weights stay in system RAM, processed by the CPU.

4. Backend Integration:

    - CUDA (NVIDIA GPUs): With CUDA support compiled (e.g., `make LLAMA_CUBLAS=1`), llama.cpp uses NVIDIA’s cuBLAS library for GPU acceleration. Layers offloaded to the GPU benefit from parallel matrix operations.
    - Metal (Apple GPUs): On macOS, Metal support offloads layers to the integrated GPU, optimized for Apple Silicon.
    - SYCL (Intel GPUs): For Intel GPUs, SYCL enables layer offloading with similar logic.
    - The CPU uses GGML’s SIMD-optimized routines for any layers not offloaded.

Performance Implications

- Full GPU Offload: Offloading all layers (e.g., -ngl 32 for a 7B model with enough VRAM) maximizes GPU parallelism, yielding the fastest inference (e.g., 20-50 tokens/s on an NVIDIA RTX 3090 with a 7B model in Q4).
- Partial Offload: Splitting layers (e.g., 20 on GPU, 12 on CPU) balances VRAM constraints but introduces a performance hit due to data transfer between GPU and CPU RAM over the PCIe bus. This hybrid mode is slower than full GPU or full CPU but allows larger models on limited VRAM (e.g., 13B on a 12 GB GPU).
- CPU Only: With no offload (`-ngl 0`), inference relies solely on CPU cores, typically achieving 5-15 tokens/s on modern CPUs (e.g., Ryzen 9) for a 7B model.

Multi-GPU Support

For systems with multiple GPUs, llama.cpp can split layers across them using the `--split-mode` and `--tensor-split` options:

- --split-mode layer (default): Distributes layers across GPUs (e.g., 16 layers per GPU on two GPUs for a 32-layer model).
- --split-mode row: Splits tensor rows across GPUs, less common for layer-based models.
- --tensor-split 0.5,0.5: Allocates 50% of the workload to each of two GPUs (adjust fractions based on GPU count and VRAM).

## llama.swiftui example

### Using the old llama.cpp

- Exactly follow the `Instruction` on this [link](https://github.com/ggerganov/llama.cpp/discussions/4508)

- Clone the project
```
    git clone https://github.com/ggerganov/llama.cpp
    git checkout 0e18b2e
```
- Open the examples/llama.swiftui with Xcode
- Enable Release build

### Using the latest llama.cpp

- Please refer to [here](https://github.com/ggerganov/llama.cpp/issues/11578) on how to setup.

1. Compile the library first:

```
cmake -DCMAKE_SYSTEM_NAME=iOS \
      -DCMAKE_OSX_ARCHITECTURES=arm64 \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DGGML_METAL=OFF \
      -DSIMD_SUM_F32_DISABLED=ON \
      -S . \
      -B build

cmake --build build --config Release 
```

2. Select all `.dylib` files from the newly created build directory. Open Xcode, and put them all under the `Frameworks` folder.

3. In `General > Frameworks, Libraries and Embedded Content`, make sure all of them are flagged as <b>Embedded & Sign</b>.

4. Update Header Search Paths in your project settings:
    - Click on your project in Xcode navigator
    - Select the llama.swiftui target
    - Go to "Build Settings" tab
    - Search for "Header Search Paths"
    - Add these paths:
    ```
    $(SRCROOT)/../../include
    $(SRCROOT)/../../ggml/include
    ```

## References
- [Llama.cpp Tutorial: A Complete Guide to Efficient LLM Inference and Implementation](https://www.datacamp.com/tutorial/llama-cpp-tutorial)
- [Understanding how LLM inference works with llama.cpp](https://www.omrimallis.com/posts/understanding-how-llm-inference-works-with-llama-cpp/)