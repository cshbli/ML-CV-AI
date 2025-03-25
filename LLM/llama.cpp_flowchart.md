# llama.cpp Flowchart

## End-to-End flowchart

```mermaid
flowchart TD
  start([start]) --> backend_load["ggml_backend_load_all"]
  backend_load --> default_params["llama_model_default_params"]
  default_params --> model_load["llama_model_load_from_file"]
  model_load --> get_vocab["llama_model_get_vocab"]
  get_vocab --> tokenize["llama_tokenize"]
  tokenize --> default_context(["llama_context_default_params"])
  default_context --> context_init(["llama_init_from_model"])
  context_init --> sampler_chain_default("llama_sampler_chain_default_params")
  context_init -.-> kv_cache_init("llama_kv_cache_unified::init")
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

  classDef CoreProcess fill:#f9f,stroke:#333,stroke-width:2px;
  classDef process fill:#faa,stroke:#333,stroke-width:1px;

  class decode,kv_cache_init CoreProcess;
  class model_load,tokenize,context_init,sample,convert process;  
```

### llama_decode function deep dive
Please check [llama.cpp_flowchart_decode.md](llama.cpp_flowchart_decode.md) for more detailed information about this function.

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

## llama_model_load_from_file() in llama.cpp

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
    keep_all --> log_devices["Get device info (name, memory)<br>ggml_backend_dev_memory"]
    
    log_devices --> model_loader_init["Create new llama_model_loader instance<br>llama_model_loader"]    

    subgraph llama_model_load[llama_model_load]
    
      model_loader_init --> load_arch["Load model architecture<br>model.load_arch()"]
      load_arch --> load_hparams["Load hyperparameters<br>model.load_hparams"]
      load_hparams --> load_vocab["Load vocabulary<br>model.load_vocab"]
      load_vocab --> load_stats["Load model stats<br>model.load_stats"]
      
      load_stats --> vocab_only{"params.vocab_only?"}
      
      vocab_only -->|Yes| skip_tensors["Skip loading weights"]
      vocab_only -->|No| load_tensors["Load model tensors/weights<br>model.load_tensors"]
      
      skip_tensors --> check_status
      load_tensors --> check_status{"Check loading<br>status"}
      
      check_status -->|Success| return_model["Return model pointer"]
      check_status -->|Error| free_model["Free model"]
      check_status -->|Cancelled| free_model
      
      free_model --> return_null["Return nullptr"]
    end

    classDef CoreProcess fill:#f9f,stroke:#333,stroke-width:2px;
    class load_tensors CoreProcess;
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

## `llama_model_loader` in llama-model-loader.cpp

The `llama_model_loader` constructor initializes the infrastructure for loading large language models from GGUF files, setting up metadata access without immediately loading the full weights.

```mermaid
flowchart TD
    start([Start]) --> parse_overrides["Process parameter overrides
    (if provided)"]
    
    parse_overrides --> load_meta["Load main GGUF file metadata:
    - Create gguf_context
    - Extract architecture information
    - Initialize file handles"]
    
    load_meta --> index_tensors["Build tensor index:
    - Map tensor names to file locations
    - Track tensor sizes and offsets
    - Count total elements and bytes"]
    
    index_tensors --> check_split{"Is model split?
    (n_split > 1)"}
    
    check_split -->|Yes| load_splits["Load split metadata:
    1. Verify main file is idx=0
    2. Generate list of split filenames
    3. Load metadata from each split
    4. Add tensors from each to unified index"]
    
    check_split -->|No| analyze_model["Analyze model information:
    - Count tensors of each type
    - Determine model quantization format
    - Set LLAMA_FTYPE based on predominant type"]
    
    load_splits --> analyze_model
    
    analyze_model --> log_metadata["Log detailed metadata:
    - Print all key-value pairs
    - Output tensor type statistics
    - Show quantization information"]
    
    log_metadata --> check_mmap["Check if mmap is supported
    on current platform"]
    
    check_mmap --> finish([End constructor])
```

### Key Features

1. **Metadata Extraction**:
   - Loads model architecture, hyperparameters, and configuration
   - Builds a map of all tensors with their locations and sizes
   - Supports parameter overrides via command line or API

2. **Multi-File Support**:
   - Handles models split across multiple files (sharded models)
   - Validates file consistency and ordering
   - Creates a unified view of tensors across all files

3. **Memory Efficiency**:
   - Delays actual weight loading until needed
   - Prepares for memory mapping when supported
   - Only loads metadata initially, not the full model

4. **Analysis & Diagnostics**:
   - Determines model quantization format
   - Outputs detailed statistics and information
   - Validates model consistency

This constructor is the first step in loading a model, creating the scaffolding that enables efficient, on-demand loading of the actual weight data when needed for inference.

## `llama_model::load_tensors()` in `llama-model.cpp`

This function is responsible for loading model weights (tensors) from disk into memory and allocating them optimally across available computing devices.

```mermaid
flowchart TD
    start([Start]) --> setup["Set up buffer types for
    CPU and GPU devices"]
    
    setup --> determine["Determine memory distribution:
    - Calculate split points for multi-device usage
    - Assign layers to appropriate devices
    - Handle GPU offloading decisions"]
    
    determine --> create_tensors["Create tensors based on architecture:
    - Token embeddings
    - Attention layers (Q,K,V)
    - Feed-forward networks (FFN)
    - Layer normalization
    - Output projection"]
    
    create_tensors --> select_backend["Select appropriate backend for each tensor:
    - Check hardware compatibility
    - Choose optimal memory placement
    - Handle tensor operations (MUL_MAT, ADD, etc.)"]
    
    select_backend --> allocate["Allocate memory buffers:
    - Create backend-specific buffers
    - Set up memory mapping if enabled
    - Configure memory locking if requested"]
    
    allocate --> load_data["Load weight data from files:
    - Copy data to appropriate buffers
    - Track progress via callbacks
    - Handle multi-file models"]
    
    load_data --> finalize["Finalize loading:
    - Register tensors by name
    - Report memory usage statistics
    - Handle GPU offloading information"]
    
    finalize --> finish([Return success/failure])

    classDef CoreProcess fill:#f9f,stroke:#333,stroke-width:2px;
    class load_data CoreProcess
```

### Key Features

1. **Architecture-Specific Loading**: Handles 50+ model architectures (LLaMA, Falcon, Gemma, etc.) with specialized tensor layouts

2. **Multi-Device Support**: Distributes model layers across:
   - Multiple GPUs
   - CPU + GPU combinations
   - Various specialized accelerators

3. **Memory Optimization**:
   - Uses memory mapping for efficiency when possible
   - Properly aligns tensors for hardware acceleration
   - Minimizes copying between host and device memory

4. **Operation-Based Tensor Management**:
   - Assigns tensors to appropriate devices based on operations (matrix multiply, addition, etc.)
   - Handles specialized operations like RoPE (Rotary Position Embedding)

This function enables efficient inference by ensuring model weights are optimally placed across available computing resources, critical for running large language models on consumer hardware.

## `llama_model_loader::load_all_data()` in `llama-model-loader.cpp`

This function loads the actual weight data from files into memory for all tensors in a model. It's a crucial part of the model loading process that transfers data from storage to computation buffers.

```mermaid
flowchart TD
    start([Start]) --> setup["Set up loading resources:
    - Read buffers
    - Validation futures
    - Async upload buffers"]
    
    setup --> prepare_gpu["Check for GPU upload capabilities:
    - Pinned memory support
    - Event synchronization
    - Host-to-device transfer"]
    
    prepare_gpu --> loop["Process each tensor in context"]
    
    loop --> report["Call progress callback
    size_done/size_data"]
    
    report -->|"Continue"| check_method{"Loading method?"}
    report -->|"Cancelled"| return_false["Return false (cancelled)"]
    
    check_method -->|"Memory mapping"| mmap["Memory-mapped loading:
    1. Get mapping for tensor's file
    2. Point tensor data to mapped address 
    3. Optionally validate in parallel"]
    
    check_method -->|"Direct reading"| check_buffer{"Tensor in
    host memory?"}
    
    check_buffer -->|"Yes"| direct_load["Read directly from file
    to tensor memory"]
    
    check_buffer -->|"No"| check_async{"Can use
    async uploads?"}
    
    check_async -->|"Yes"| async_load["Load chunks through
    pinned memory buffers:
    1. Read chunk to host buffer
    2. Async transfer to GPU
    3. Record sync event"]
    
    check_async -->|"No"| simple_copy["Read to temporary buffer
    then copy to tensor"]
    
    mmap --> update_counters["Update size_done counter"]
    direct_load --> update_counters
    async_load --> update_counters
    simple_copy --> update_counters
    
    update_counters --> more_tensors{"More tensors?"}
    more_tensors -->|"Yes"| loop
    more_tensors -->|"No"| cleanup["Cleanup:
    1. Sync & free events
    2. Free temp buffers
    3. Check validation results"]
    
    cleanup --> check_done{"All data loaded?"}
    
    check_done -->|"Yes"| final_cleanup["Unmap unused memory regions"]
    check_done -->|"No"| return_true
    
    final_cleanup --> final_callback["Call progress callback (100%)"]
    
    final_callback -->|"Continue"| return_true["Return true (success)"]
    final_callback -->|"Cancelled"| return_false
```

### Key Features

1. **Multiple Loading Methods**:
   - **Memory Mapping**: Zero-copy access to model data directly from files
   - **Direct Loading**: Reading data from files into pre-allocated memory
   - **Asynchronous GPU Uploads**: Efficient pinned-memory transfers for GPU tensors

2. **Performance Optimizations**:
   - Parallel validation of tensor data
   - Asynchronous uploads for GPU memory
   - Pinned memory staging buffers for efficient transfers
   - Unmapping of unused memory regions

3. **Progress Reporting and Cancellation**:
   - Reports loading progress through callback
   - Honors cancellation requests at multiple points

This function is critical for memory efficiency in llama.cpp, enabling several techniques that allow large models to run on consumer hardware.

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

## `llama_context::llama_context()` Constructor Analysis

The `llama_context` constructor is the fundamental initialization function in llama.cpp that prepares the model for inference. It takes a model reference and parameters, then configures everything needed for efficient model execution.

```mermaid
flowchart TD
    start([Start]) --> params["Initialize parameters:
    - Context window size
    - Batch sizes
    - Threading configuration
    - Position encoding settings"]
    
    params --> validate["Validate parameters:
    - Check context sizes
    - Ensure batch size is valid
    - Verify configuration compatibility
    - Handle special cases per architecture"]
    
    validate --> log["Log configuration:
    - Context parameters
    - Sequence settings
    - Attention mechanism
    - RoPE settings"]
    
    log --> backends["Initialize backends:
    - Set up GPU backends from model devices
    - Add acceleration backends (BLAS, etc.)
    - Initialize CPU backend as fallback
    - Configure threading functions"]
    
    backends --> memory["Set up memory systems:
    - Create output buffers
    - Initialize KV cache
    - Allocate computation buffers
    - Set up appropriate memory types"]
    
    memory --> compute["Configure compute infrastructure:
    - Create computation context
    - Set up scheduler
    - Initialize pipeline parallelism (if multiple devices)
    - Optimize backend selection"]
    
    compute --> reserve["Reserve worst-case resources:
    - Build maximum size graph
    - Reserve memory for computation
    - Set up buffer allocation
    - Log memory usage stats"]
    
    reserve --> finish([End])
```

## Key Components:

1. **Parameter Processing**:
   - Handles context window sizing
   - Configures RoPE (Rotary Position Embedding) parameters
   - Sets up model-specific adaptations

2. **Hardware Acceleration**:
   - Creates a prioritized list of backends (GPU → Accelerators → CPU)
   - Configures thread pools for parallel execution
   - Sets up pipeline parallelism for multi-device execution

3. **Memory Management**:
   - Allocates KV cache based on model requirements
   - Creates output buffers for logits/embeddings
   - Sets up computation buffers optimized for each backend
   - Configures memory sharing between devices when beneficial

4. **Inference Engine Setup**:
   - Builds graph scheduler for distributing computation
   - Reserves worst-case computation graph
   - Sets up callbacks and abort handlers
   - Configures tensor operation routing

This constructor is the foundation for all model inference, establishing the resources and execution framework that enables efficient language model inference across diverse hardware setups.

## `llama_kv_cache_unified::init()` in `llama-kv-cache.cpp`

This function initializes the key-value (KV) cache for a language model, which is critical for efficient inference by storing previously computed keys and values.

```mermaid
flowchart TD
    start([Start]) --> set_flags["Set initial state:
    - has_shift = false
    - detect if model is recurrent
    - determine v_trans based on model type
    - set can_shift flag based on architecture"]
    
    set_flags --> log_info["Log initialization parameters:
    - kv_size, offload, type_k, type_v
    - n_layer, can_shift"]
    
    log_info --> init_counters["Initialize counters:
    - head = 0 (starting position)
    - size = kv_size (total capacity)
    - used = 0 (no cells used yet)
    - Set data types"]
    
    init_counters --> setup_cells["Set up cells storage:
    - Clear cells array
    - Resize to kv_size"]
    
    setup_cells --> create_contexts["Create GGML contexts:
    - One context per buffer type
    - Set up memory for tensor overhead"]
    
    create_contexts --> create_tensors["Create K/V tensors for each layer:
    1. Determine dimensions based on model
    2. Choose buffer type (GPU/CPU)
    3. Create tensors with proper names
    4. Push to k_l and v_l vectors"]
    
    create_tensors --> allocate_memory["Allocate memory:
    - For each context/buffer type
    - Allocate tensor memory
    - Clear memory to prevent NaNs
    - Log buffer sizes"]
    
    allocate_memory -->|Success| return_true["Return true"]
    allocate_memory -->|Failure| return_false["Return false"]
```

### Key Components

1. **Model-Specific Configuration**:
   - Detects if model is recurrent (like Mamba or RWKV)
   - Sets up appropriately for transformer vs. state-space models
   - Configures value transposition based on model needs

2. **Memory Management**:
   - Creates appropriate buffer types (CPU/GPU) based on offload parameter
   - Allocates memory efficiently for different hardware
   - Supports GPU offloading for specific layers

3. **Tensor Organization**:
   - Creates tensors for each layer's keys and values
   - Handles grouped query attention (GQA) dimensions
   - Sets up proper naming for debugging

4. **Optimization Features**:
   - Configures "can_shift" capability for context extension
   - Sets up memory layouts optimized for specific models
   - Provides clear logging for debugging

This initialization function is a critical part of llama.cpp's efficiency, as proper KV cache setup dramatically improves inference speed by avoiding redundant computations for previously seen tokens.

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

## `llama_decode()` in `llama-context.cpp`

The `llama_decode()` function is the core inference engine of llama.cpp, processing tokens through the model to generate logits or embeddings.

```mermaid
flowchart TD
    start([Input: batch of tokens]) --> validate["Validate inputs
    - Check if n_tokens > 0
    - Verify token IDs are valid"]
    
    validate --> batch_prepare["Prepare batch for processing
    - Allocate memory if needed
    - Start performance timing"]
    
    batch_prepare --> split["Split into micro-batches
    - Handle based on n_ubatch parameter
    - Prepare for sequential processing"]
    
    split --> kv_update["Update KV cache
    - Defragment if needed
    - Find slots for new tokens"]
    
    kv_update --> loop{"Process all 
    micro-batches"}
    
    loop -->|"For each micro-batch"| graph_build["Build computation graph
    - Create tensor operations
    - Connect forward pass layers"]
    
    graph_build --> allocate["Allocate computation buffers
    - Optimize memory usage
    - Prepare for execution"]
    
    allocate --> execute["Execute computation
    - Run through transformer layers
    - Process through attention & FFN"]
    
    execute --> extract["Extract outputs
    - Get logits (token probabilities)
    - Get embeddings (if requested)"]
    
    extract --> update_kv["Update KV cache state
    - Move head pointer
    - Update sequence tracking"]
    
    update_kv --> next_batch{"More batches?"}
    next_batch -->|Yes| loop
    next_batch -->|No| finalize["Finalize processing
    - Update output mappings
    - Check for defragmentation needs"]
    
    finalize --> reset["Reset scheduler state"]
    
    reset --> return["Return status code:
    0: Success
    2: Aborted
    -1: Invalid input
    -2: Memory allocation failed
    -3: Computation failed"]
```

### Key Components

1. **Batch Processing**:
   - Handles both token IDs and direct embedding inputs
   - Processes tokens in micro-batches for efficiency
   - Supports both causal and non-causal attention modes

2. **KV Cache Management**:
   - Maintains key-value pairs for efficient sequence processing
   - Handles sequence IDs and positions for multi-sequence batches
   - Implements cache defragmentation and shifting

3. **Computation Optimization**:
   - Uses hardware acceleration (GPU, CPU optimizations)
   - Implements pipeline parallelism when appropriate
   - Optimizes memory usage with scheduler

4. **Multiple Output Modes**:
   - Token logits for text generation
   - Token embeddings for representation learning
   - Sequence embeddings with different pooling types

The function is the central execution point that brings together all the components of the model (attention, feed-forward networks, embedding lookups) to perform the actual language model inference.

### llama_decode function deep dive
Please check [llama.cpp_flowchart_decode.md](llama.cpp_flowchart_decode.md) for more detailed information about this function.

## `llama_sampler_sample()` in `llama-sampling.cpp`

This function serves as the core token selection mechanism for llama.cpp, taking a model's logits and applying sampling strategies to choose the next token.

```mermaid
flowchart TD
    start([Start]) --> get_logits["Get logits for position idx<br>llama_get_logits_ith"]
    
    get_logits --> get_vocab["Get vocabulary information
    llama_get_model(ctx)
    lama_model_get_vocab(model)
    llama_vocab_n_tokens(vocab)"]
    
    get_vocab --> create_token_data["Create token data array for all tokens with logits but no probabilities yet"]
    
    create_token_data --> setup_array["Set up token data array structure"]
    
    setup_array --> apply_sampler["Apply sampling strategy chain<br>llama_sampler_apply"]
    
    apply_sampler --> check_selected["Verify valid selection
    GGML_ASSERT(cur_p.selected >= 0 && 
                cur_p.selected < cur_p.size)"]
    
    check_selected --> get_token["Extract selected token ID"]
    
    get_token --> notify_sampler["Notify sampler of selection<br>llama_sampler_accept"]
    
    notify_sampler --> return_token["Return selected token"]
    
    subgraph "llama_sampler_apply (hidden details)"
        sa1["Apply temperature"]
        sa2["Apply top_k filtering"]
        sa3["Apply top_p filtering"]
        sa4["Apply repetition penalties"]
        sa5["Apply grammar constraints"]
        sa6["elect final token"]
        
        sa1 --> sa2 --> sa3 --> sa4 --> sa5 --> sa6
    end
    
    apply_sampler -.-> sa1
```

### Key Steps

1. **Get Raw Logits**: Retrieve the model's raw output scores for the specified position

2. **Prepare Token Data**: Convert logits into a structured array containing all tokens in the vocabulary

3. **Apply Sampling Strategy**: Call `llama_sampler_apply()` which modifies token probabilities based on the configured sampling chain:
   - Temperature scaling
   - Top-K filtering
   - Top-P (nucleus) sampling 
   - Repetition penalties
   - Grammar constraints
   - etc.

4. **Select Token**: The sampling chain selects the final token and sets it in `cur_p.selected`

5. **Update Sampler State**: Call `llama_sampler_accept()` to update the sampler's internal state with the selected token (for repetition tracking, etc.)

6. **Return Token**: Return the selected token ID for use in the generation process

This is the critical function that transforms raw model outputs into the next token in the generated sequence.

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
