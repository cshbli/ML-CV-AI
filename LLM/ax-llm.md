# ax-llm

https://github.com/AXERA-TECH/ax-llm/tree/prefill

# `LLM::Run()` Function Analysis - LLM Text Generation Engine

This function executes a complete language model inference pipeline on the AX650 NPU, generating text from either a prompt string or pre-processed embeddings.

```mermaid
flowchart TD
    start([Start]) --> init["Initialize:
    - Reset stop flag
    - Set up attention masks
    - Prepare embedding vectors"]
    
    init --> prefill["PREFILL PHASE (all input tokens at once):
    For each model layer:
      1. Load layer model if dynamic
      2. Configure inputs (embed, indices, masks)
      3. Run prefill inference
      4. Store K/V cache outputs
      5. Unload layer if dynamic"]
    
    prefill --> first_token["Generate first token:
    1. Process last prefill embedding 
    2. Run post-processing model
    3. Apply sampling strategy 
    4. Select next token
    5. Measure Time To First Token (TTFT)"]
    
    first_token --> gen_loop["GENERATION LOOP:
    For each new token position:"]
    
    gen_loop --> get_embed["Convert token to embedding"]
    
    get_embed --> layer_loop["Process through all layers:
    For each model layer:
      1. Load layer if dynamic
      2. Set up inputs & K/V caches
      3. Run decode inference
      4. Update K/V caches
      5. Process embedding
      6. Unload layer if dynamic"]
    
    layer_loop --> update_mask["Update attention mask"]
    
    update_mask --> sample["Run post-processing:
    1. Apply sampling strategy
    2. Select next token"]
    
    sample --> check_eos{"Is token
    end-of-sequence?"}
    
    check_eos -->|Yes| finish_gen["Break generation loop"]
    check_eos -->|No| check_callback{"Callback
    configured?"}
    
    check_callback -->|Yes| run_callback["Call progress callback:
    - Convert tokens to text
    - Report generation speed
    - Send partial results"]
    check_callback -->|No| update_progress["Update progress bar"]
    
    run_callback --> check_continue{"Continue
    generating?"}
    update_progress --> check_continue
    
    check_continue -->|Yes| gen_loop
    check_continue -->|No| decode["Decode all tokens to text"]
    finish_gen --> decode
    
    decode --> log_stats["Log generation statistics:
    - Token generation rate
    - Total generation time"]
    
    log_stats --> return["Return generated text"]
    
    return --> finish([End])
```

## Key Features

1. **Two-Phase Execution**
   - **Prefill**: Processes all input tokens in parallel for efficiency
   - **Decode**: Generates new tokens one by one in autoregressive fashion

2. **Memory Optimization**
   - Dynamically loads/unloads model layers to reduce memory usage
   - Uses memory mapping for efficient file access
   - Carefully manages K/V caches for attention state

3. **Hardware Acceleration**
   - Uses AX650 NPU for computation
   - Handles cache invalidation for CPU/NPU memory coherence
   - Optimizes memory transfers between host and device

4. **Text Generation Controls**
   - Implements sampling with temperature/top-k/top-p
   - Supports early stopping
   - Detects end-of-sequence tokens

5. **Progress Reporting**
   - Provides real-time generation statistics
   - Supports callback for partial results
   - Measures token generation speed

This implementation represents a complete transformer inference engine optimized for edge hardware, balancing memory constraints with performance requirements.

## `LLM::Init()` Function Analysis

This function initializes a Large Language Model (LLM) for inference on an AX650 hardware accelerator, setting up all necessary components for model execution.

```mermaid
flowchart TD
    start([Start]) --> store_attr["Store LLM attributes"]
    
    store_attr --> tokenizer["Initialize tokenizer:
    - Load vocabulary
    - Configure special tokens (BOS/EOS)"]
    
    tokenizer -->|Success| embed["Initialize embedding selector:
    - Load token embeddings
    - Configure memory mapping"]
    tokenizer -->|Failure| return_false["Return false"]
    
    embed -->|Success| layer_setup["Set up model layers:
    - Resize layer containers
    - Prepare file paths"]
    embed -->|Failure| return_false
    
    layer_setup --> dynamic_check{"Dynamic loading
    enabled?"}
    
    dynamic_check -->|No| load_layers["Load all layers immediately:
    - Initialize each axmodel
    - Track remaining memory"]
    
    dynamic_check -->|Yes| prepare_layers["Prepare for dynamic loading:
    - Either memory map files
    - Or load into buffer vectors"]
    
    load_layers --> post_model["Initialize post-processing model"]
    prepare_layers --> post_model
    
    post_model -->|Success| extract_info["Extract model configuration:
    - max_token_len
    - kv_cache size/num
    - prefill_token_num"]
    post_model -->|Failure| return_false
    
    extract_info --> validate["Validate parameters:
    max_token_len <= kv_cache_num"]
    
    validate -->|Valid| postprocess["Load post-processing configuration"]
    validate -->|Invalid| return_false
    
    postprocess --> return_true["Return true"]
```

### Key Components

1. **Model Loading**:
   - Either loads all model layers at once or prepares for dynamic loading
   - Supports memory mapping for efficient file access
   - Loads embedding tables for token representation

2. **Memory Optimization**:
   - Dynamic layer loading to reduce memory footprint
   - Memory mapping for efficient file access
   - Tracks remaining memory for debugging

3. **Hardware Configuration**:
   - Sets up models for AX650 acceleration
   - Configures appropriate memory layouts
   - Prepares input/output tensors for hardware

4. **Pipeline Setup**:
   - Tokenizer for text conversion
   - Embedding selector for token embedding
   - Layer processors for transformer operations
   - Post-processor for output generation

This function creates the foundation for the entire inference pipeline, from text input to generated output.

## `ax_runner_ax650::init()` Function Analysis

This function initializes a model/layer for inference on the AX650 NPU hardware accelerator, setting up the necessary resources for model/layer execution.

```mermaid
flowchart TD
    start([Start]) --> check_method{"Loading method?"}
    
    check_method -->|Memory Map| mmap["Create memory map of model file"]
    check_method -->|Standard Read| read["Read entire model file into buffer"]
    
    mmap -->|Success| validate_mmap["Validate memory map data"]
    mmap -->|Failure| error_mmap["Log mmap error<br>Return -1"]
    
    read -->|Success| proceed_read["Pass file data to buffer-based init"]
    read -->|Failure| error_read["Log read_file error<br>Return -1"]
    
    validate_mmap -->|Valid| call_buffer_init["Call buffer-based init()<br>with mapped memory"]
    validate_mmap -->|Invalid| error_mmap
    
    proceed_read --> call_buffer_init
    
    call_buffer_init --> handle_creation["Create handle if not exists"]
    
    handle_creation --> check_engine{"Engine initialized?"}
    
    check_engine -->|No| init_engine["Configure NPU attributes
    - Set hardware mode
    - Initialize AX_SYS
    - Initialize AX_ENGINE"]
    check_engine -->|Yes| create_handle
    
    init_engine -->|Success| create_handle["Create engine handle<br>with model data"]
    init_engine -->|Failure| error_init["Return error code"]
    
    create_handle -->|Success| run_sub_init["Run sub-initialization:
    1. Create engine context
    2. Get I/O information
    3. Allocate memory for I/O
    4. Set up tensor structures"]
    create_handle -->|Failure| error_handle["Log error<br>Return error code"]
    
    run_sub_init --> cleanup_mmap{"Using mmap?"}
    run_sub_init --> cleanup_buffer{"Using buffer?"}
    
    cleanup_mmap -->|Yes| close_mmap["Close memory mapped file"]
    cleanup_buffer -->|Yes| free_buffer["Free model buffer"]
    
    close_mmap --> return["Return result code"]
    free_buffer --> return
    error_mmap --> finish([End])
    error_read --> finish
    error_init --> finish
    error_handle --> finish
    return --> finish
```

### Key Components

1. **Flexible Loading Strategies**:
   - Memory mapping for efficient loading of large models
   - Standard file reading for compatibility with all systems

2. **Hardware Initialization**:
   - One-time engine initialization with appropriate NPU settings
   - Handle creation for the specific model

3. **Resource Allocation**:
   - Context creation for model execution
   - Memory allocation for inputs and outputs
   - Tensor organization for consistent access patterns

4. **I/O Management**:
   - Identification of model inputs and outputs
   - Memory allocation with appropriate alignment
   - Organization of tensor shapes and sizes

This function is the entry point for setting up inference on the AX650 hardware, enabling efficient execution of neural network models.

## `sub_init()` Function Analysis

This function initializes the execution context and I/O resources for a model on the AX650 NPU hardware after the model handle has been created.

```mermaid
flowchart TD
    start([Start]) --> create_context["Create engine context:
    1. AX_ENGINE_CreateContext()
    2. AX_ENGINE_CreateContextV2()"]
    
    create_context -->|Success| get_io_count["Get I/O group count:
    AX_ENGINE_GetGroupIOInfoCount()"]
    
    create_context -->|Failure| return_error["Return error code"]
    
    get_io_count -->|Success| resize_containers["Resize data structures:
    - io_info vector
    - io_data vector
    - input/output tensor groups"]
    
    get_io_count -->|Failure| return_error
    
    resize_containers --> check_prepare{"Is I/O already 
    prepared?"}
    
    check_prepare -->|No| group_loop["Process each I/O group
    (Loop through io_count)"]
    
    check_prepare -->|Yes| skip_prepare["Skip preparation
    (I/O already set up)"]
    
    group_loop --> get_group_info["Get group I/O info:
    AX_ENGINE_GetGroupIOInfo()"]
    
    get_group_info --> prepare_group_io["Allocate memory for group I/O:
    prepare_io() with DEFAULT input and
    CACHED output strategy"]
    
    prepare_group_io -->|Success| more_groups{"More groups?"}
    prepare_group_io -->|Failure| return_error
    
    more_groups -->|Yes| group_loop
    more_groups -->|No| tensor_loop["Build tensor metadata
    for all groups"]
    
    tensor_loop --> process_outputs["For each output:
    1. Create tensor metadata
    2. Copy shape information
    3. Store address pointers
    4. Add to group output list"]
    
    process_outputs --> process_inputs["For each input:
    1. Create tensor metadata
    2. Copy shape information 
    3. Store address pointers
    4. Add to group input list"]
    
    process_inputs --> more_tensor_groups{"More groups?"}
    
    more_tensor_groups -->|Yes| tensor_loop
    more_tensor_groups -->|No| setup_defaults["Set up default access:
    - moutput_tensors = group 0 outputs
    - minput_tensors = group 0 inputs"]
    
    setup_defaults --> mark_prepared["Mark I/O as prepared:
    _parepare_io = true"]
    
    mark_prepared --> return_success["Return status code"]
    skip_prepare --> return_success
    
    return_success --> finish([End])
    return_error --> finish
```

### Key Components

1. **Context Creation**:
   - Creates execution context for the hardware accelerator
   - Links the context to the model handle

2. **Memory Organization**:
   - Determines the number of I/O groups in the model
   - Allocates appropriately sized containers for all resources

3. **I/O Group Handling**:
   - Gets detailed information about each I/O group
   - Allocates memory for all inputs and outputs
   - Uses DEFAULT strategy for inputs and CACHED strategy for outputs

4. **Tensor Abstraction**:
   - Creates higher-level tensor abstractions with metadata
   - Stores shape information and memory pointers
   - Organizes tensors in convenient access structures

This function completes the initialization started by the `init()` function, preparing all resources needed for model execution on the AX650 hardware.

## `prepare_io()` Function Analysis

This function allocates memory for input and output tensors needed by the AX650 NPU (Neural Processing Unit), setting up the data transfer pathway between the host system and the accelerator.

```mermaid
flowchart TD
    start([Start]) --> init["Initialize io_data:
    - Clear memory structure
    - Allocate input buffer array
    - Set input count"]
    
    init --> input_loop["Process all input tensors
    (Loop through info->nInputSize)"]
    
    input_loop --> alloc_strategy{"Check allocation
    strategy.first"}
    
    alloc_strategy -->|Cached| alloc_cached["Allocate cached memory:
    AX_SYS_MemAllocCached()"]
    
    alloc_strategy -->|Default| alloc_regular["Allocate regular memory:
    AX_SYS_MemAlloc()"]
    
    alloc_cached --> check_input_alloc{"Allocation
    successful?"}
    alloc_regular --> check_input_alloc
    
    check_input_alloc -->|No| cleanup_inputs["Free allocated input buffers
    free_io_index()"]
    check_input_alloc -->|Yes| zero_input["Initialize buffer with zeros
    memset()"]
    
    cleanup_inputs --> return_error["Return error code"]
    
    zero_input --> more_inputs{"More inputs?"}
    more_inputs -->|Yes| input_loop
    more_inputs -->|No| init_outputs["Initialize output structure:
    - Allocate output buffer array
    - Set output count"]
    
    init_outputs --> output_loop["Process all output tensors
    (Loop through info->nOutputSize)"]
    
    output_loop --> out_strategy{"Check allocation
    strategy.second"}
    
    out_strategy -->|Cached| out_cached["Allocate cached memory:
    AX_SYS_MemAllocCached()"]
    
    out_strategy -->|Default| out_regular["Allocate regular memory:
    AX_SYS_MemAlloc()"]
    
    out_cached --> check_output_alloc{"Allocation
    successful?"}
    out_regular --> check_output_alloc
    
    check_output_alloc -->|No| cleanup_all["Free all inputs and
    processed outputs"]
    check_output_alloc -->|Yes| zero_output["Initialize buffer with zeros
    memset()"]
    
    cleanup_all --> return_error
    
    zero_output --> more_outputs{"More outputs?"}
    more_outputs -->|Yes| output_loop
    more_outputs -->|No| return_success["Return success (0)"]
    
    return_success --> finish([End])
    return_error --> finish
```

### Key Features

1. **Hardware-Accelerated Memory Allocation**
   - Allocates memory that's optimized for NPU access
   - Uses physical memory addresses for hardware DMA transfers
   - Maintains virtual addresses for CPU access

2. **Memory Strategy Support**
   - **Cached Memory**: Better for CPU-side processing, requires synchronization
   - **Uncached Memory**: Direct access, avoids cache coherency issues
   - Separate strategies for inputs and outputs

3. **Memory Alignment**
   - Ensures proper alignment (128-byte boundaries) for optimal DMA performance
   - Critical for hardware acceleration efficiency

4. **Error Handling**
   - Properly cleans up resources on allocation failures
   - Returns meaningful error codes when allocation fails

This function is a critical bridge between software and hardware acceleration, setting up the memory structures that allow efficient data transfer between the CPU and the AX650 NPU.