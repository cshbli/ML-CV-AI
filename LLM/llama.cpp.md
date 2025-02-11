# llama.cpp

## Llama.cpp Archtiecture

Llama.cpp’s backbone is the original Llama models, which is also based on the transformer architecture.

<img src="llama_architecture.avif">

The main difference between the LLaMa architecture and the transformers’:

- <b>Pre-normalization (GPT3):</b> used to improve the training stability by normalizing the input of each transformer sub-layer using the RMSNorm approach instead of normalizing the output.
- <b>SwigGLU activation function (PaLM):</b> the original non-linearity ReLU activation function is replaced by the SwiGLU activation function, which leads to performance improvements.
- <b>Rotary embeddings (GPTNeao):</b> the rotary positional embeddings (RoPE) was added at each layer of the network after removing the absolute positional embeddings.

## Computation Acceleration

|Backend|Supported Platforms|Notes|
|---|---|---|
|CPU|All|mutli-threading and SIMD(AVX, AVX2, AVX512)|
|CUDA|NVIDIA GPUs|Compute Unified Device Architecture|
|HIP|AMD GPUs|Heterogeneous-Interface Parallel Programming|
|Metal|Apple GPUs|
|CANN|Huawei Ascend AI processors|Compute Architecture for Neural Networks|
|MUSA|Moore Threads MTT GPU|Multiverse Unified System Architecture|
|Vulkan|Cross-platform (NVIDIA, AMD, Apple, Intel) |Low-level API for rendering and compute|
| |Cross-platform (Windows, Linux, MacOS, Android) |On platform like Android, it is the de facto standard for GPU computing|
|Kompute|High-leve framework for <b>Vulkan</b>-based GPU computing|
|OpenCL|Cross-platform (NVIDIA, AMD, Intel) |Enables acceleration on devices that don't support CUDA or Metal|
|SYCL|C++ for Heterogeneous computing|high-level programming model builds on the foundation of <b>OpenCL</b>|
|BLAS|Cross-platform math libraries for matrix operations|Basic Linear Algebra Subprograms|
| | Intel MKL: Intel Math Kernel Library| using AVX/AVX2/AVX512 instructions for SIMD|
| | OpenBLAS: Optimized BLAS library| Open-source, optimized for various CPUs (Intel, AMD, ARM)|
| | cuBLAS: NVIDIA CUDA BLAS library| NVIDIA GPUs|
| | rocBLAS: AMD ROC BLAS library| AMD GPUs using the HIP platform|
|BLIS|BLAS-like library for high-performance dense linear algebra|
|RPC|Distributed or remote computing|remote procedure call|

## Build a project outside of the source tree
- Please see this [example](https://github.com/ggerganov/llama.cpp/tree/master/examples/simple-cmake-pkg) 

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