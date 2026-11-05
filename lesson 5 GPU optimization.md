
# Lesson 5.1 — GPU Performance Bottlenecks

## 1. Two Fundamental Bottlenecks

### Compute-bound

- Limited by GPU computational capacity.
- GPU spends most of its time performing arithmetic.
- Examples: GEMM and neural network operations.
- Optimization: improve computational efficiency and use Tensor Cores.

### Memory-bound

- Limited by data transfer speed.
- GPU spends relatively more time waiting for data.
- Examples: vector addition, tensor copying, reductions.
- Optimization: reduce memory traffic and improve data reuse.

## 2. Arithmetic Intensity

Arithmetic intensity measures computation per byte transferred.

$$
AI=\frac{\text{FLOPs}}{\text{Bytes transferred from memory}}
$$

Examples:

- Tensor copy: almost zero.
- Vector addition: low.
- Matrix multiplication: high.

Higher arithmetic intensity generally improves the utilization
of computational resources.

## 3. Roofline Model

The Roofline Model estimates maximum achievable performance.

$$
Performance \leq \min(P_{\text{peak}}, BW\times AI)
$$

Where:

- P_peak: peak computational throughput.
- BW: memory bandwidth.
- AI: arithmetic intensity.

Example:

- Peak compute = 100 TFLOPS.
- Memory bandwidth = 2 TB/s.
- Arithmetic intensity = 10 FLOPs/byte.

Memory-limited performance:

$$
2\times10=20\text{ TFLOPS}
$$

Therefore:

$$
Performance\leq\min(100,20)=20\text{ TFLOPS}
$$

The workload is memory-bound.

## 4. Memory Optimization Techniques

| Technique | Purpose |
|---|---|
| Shared memory | Reuse data on-chip |
| Kernel fusion | Avoid intermediate memory traffic |
| Tiling | Improve data reuse |

## 5. Connection to FlashAttention

Traditional attention:

$$
S=QK^T
$$

$$
P=\text{softmax}(S)
$$

$$
O=PV
$$

Traditional implementations may write large intermediate
matrices to global memory and read them back.

FlashAttention uses:

- Tiling.
- Online softmax.
- On-chip memory reuse.

Its main innovation is reducing data movement, not eliminating
the underlying attention computation.

## 6. Key Takeaways

- Compute-bound: arithmetic capacity is the bottleneck.
- Memory-bound: data movement is the bottleneck.
- Arithmetic intensity: computation per byte transferred.
- Roofline: estimates the performance ceiling.
- Shared memory, tiling, and fusion reduce unnecessary
  global-memory traffic.



# Lesson 5.2 — GEMM Optimization

## 1. What is GEMM?

GEMM = General Matrix Multiplication.

Given:

$$
A_{M \times K} \times B_{K \times N}
= C_{M \times N}
$$

Each output element:

$$
C_{ij} = \sum_{k=0}^{K-1} A_{ik}B_{kj}
$$

GEMM is widely used in neural networks and transformers.

Examples:
- Linear layers: `Y = XW^T + b`
- Attention scores: `QK^T`
- Attention output: `PV`

## 2. Naive GEMM

A naive CUDA kernel assigns one thread to each
output element.

Problem:
- Different threads may load the same input values.
- Repeated global-memory accesses create memory traffic.
- Data reuse is limited.

## 3. Tiled GEMM

Divide matrices into smaller tiles.

For example, use 2x2 tiles for 6x6 matrices.

Process:
1. Load an A tile from global memory.
2. Load a B tile from global memory.
3. Store both tiles in shared memory.
4. Synchronize threads.
5. Calculate partial output values.
6. Repeat for the next K tile.
7. Write the output tile to global memory.

Main advantage:

Load data once and reuse it for multiple calculations.

## 4. GPU Memory Hierarchy

| Memory | Characteristic |
|---|---|
| Registers | Fastest, thread-private |
| Shared memory | Fast, block-shared |
| Global memory | Large, higher latency |

Tiling uses shared memory to reduce repeated
global-memory accesses.

## 5. Tensor Cores

Tensor Cores are specialized hardware units for
matrix multiply-accumulate operations.

Conceptually:

$$
D = A \times B + C
$$

They accelerate supported data types and matrix shapes,
including FP16 and BF16 on compatible hardware.

## 6. GEMM Implementations

| Method | Characteristic |
|---|---|
| Naive GEMM | Limited data reuse |
| Tiled GEMM | Shared-memory data reuse |
| Tensor Core GEMM | Specialized matrix hardware |

High-performance GEMM combines:
- Tiling
- Shared memory
- Registers
- Efficient thread layouts
- Tensor Cores
- Optimized data transfers

PyTorch commonly uses optimized libraries such as
cuBLAS and cuBLASLt for GPU matrix multiplication.

## 7. Key Takeaway

GEMM optimization aims to reduce global-memory
transfers while increasing computation per loaded value.

The principle:

Load once, reuse many times.


# Lesson 5.3 — Attention Optimization and FlashAttention

## 1. Transformer Attention

A Transformer uses three matrices:

- Q (Query): what information am I looking for?
- K (Key): what information does each token contain?
- V (Value): what information should I retrieve?

Attention consists of three main calculations.

### Step 1: Calculate Attention Scores

$$
S = \frac{QK^T}{\sqrt{d_k}}
$$

This is a GEMM operation.

If Q has shape 4 x 2 and K has shape 4 x 2:

$$
(4 \times 2)(2 \times 4) = (4 \times 4)
$$

The result is a 4 x 4 attention score matrix.

### Step 2: Apply Softmax

$$
P_{ij} =
\frac{\exp(S_{ij})}
{\sum_{k=1}^{N}\exp(S_{ik})}
$$

Softmax converts each row into a probability
distribution.

For example:

$$
[1,2,3] \rightarrow [0.09,0.24,0.67]
$$

Each probability represents how much attention
one token pays to another token.

### Step 3: Calculate the Weighted Values

$$
O = PV
$$

If P is 4 x 4 and V is 4 x 2:

$$
(4 \times 4)(4 \times 2) = (4 \times 2)
$$

The output contains the new representation
for each token.

## 2. The Performance Problem

The attention score matrix has dimensions N x N,
where N is the sequence length.

For a sequence length of 8,192:

$$
8192^2 = 67,108,864
$$

There are over 67 million elements.

For FP16, one such matrix requires approximately
128 MiB of storage.

Standard attention may materialize both the score
matrix S and probability matrix P.

Writing and reading these intermediate matrices
creates substantial global-memory traffic.

Remember the GPU memory hierarchy:

- Registers: very fast, but limited.
- Shared memory: fast, but limited.
- Global memory: large, but slower.

The problem is not only computation.
Moving large intermediate matrices also costs time.

## 3. FlashAttention: The Basic Idea

FlashAttention applies the same tiling principle
used in optimized GEMM.

Instead of calculating the entire attention matrix
at once, it processes small blocks (tiles).

Simplified process:

1. Load a Q tile into fast memory.
2. Load a K tile.
3. Calculate a small score tile.
4. Perform softmax-related calculations.
5. Load the corresponding V tile.
6. Update the output.
7. Move to the next K and V tiles.

The complete attention score matrix does not need
to be stored in global memory.

The key idea:

Process small tiles and combine their results.

## 4. Online Softmax

A challenge appears when calculating softmax tile
by tile.

Recall:

$$
P_{ij} =
\frac{\exp(S_{ij})}
{\sum_{k=1}^{N}\exp(S_{ik})}
$$

The denominator requires all scores in the row.

FlashAttention solves this using online softmax.

It maintains three running quantities:

- m: maximum score seen so far.
- l: running normalization sum.
- o: running weighted output accumulator.

### Example

Suppose one attention row contains:

$$
S = [1,2,3,4]
$$

Process it in two tiles:

- Tile 1: [1,2]
- Tile 2: [3,4]

### First Tile: [1,2]

Calculate the maximum:

$$
m_1 = 2
$$

Calculate the normalization sum:

$$
l_1 = e^{1-2} + e^{2-2}
$$

$$
l_1 \approx 1.368
$$

### Second Tile: [3,4]

The new maximum becomes:

$$
m_2 = 4
$$

Adjust the previous normalization sum:

$$
l_2 =
e^{m_1-m_2}l_1
+
\sum_{j \in tile_2}e^{S_j-m_2}
$$

Substitute the values:

$$
l_2 =
e^{-2}(1.368)+e^{-1}+1
$$

$$
l_2 \approx 1.503
$$

The normalization sum now represents all four scores.

The output accumulator is adjusted in the same way.

Finally:

$$
O = \frac{o}{l}
$$

This produces the same mathematical result as
ordinary softmax attention, subject to floating-point
rounding differences.

The key insight:

Online softmax allows attention to be calculated
incrementally without storing the complete row.

## 5. Standard Attention vs FlashAttention

| Feature | Standard Attention | FlashAttention |
|---|---|---|
| Score matrix | Often materialized | Processed in tiles |
| Probability matrix | Often materialized | Not stored in full |
| Softmax | Whole row | Online calculation |
| Memory traffic | Higher | Reduced |
| Arithmetic complexity | O(N^2 d) | O(N^2 d) |

FlashAttention does not eliminate quadratic
attention computation.

It improves how data moves through the GPU.

## 6. Why FlashAttention Matters

As sequence length increases:

- The attention matrix grows quadratically.
- Intermediate memory requirements grow.
- Global-memory traffic becomes expensive.

FlashAttention reduces memory overhead.

Potential benefits:

- Longer sequences within the same GPU memory.
- Larger training batches in suitable workloads.
- Faster attention computation.

Actual speedups depend on GPU architecture,
sequence length, data type, and implementation.

## 7. Simplified FlashAttention Pseudocode

This is conceptual pseudocode, not a complete
CUDA implementation.

```python
for each query_tile in Q:

    initialize running_max
    initialize running_sum
    initialize output_accumulator

    for each key_value_tile in K, V:

        scores = query_tile @ key_tile.T

        update running_max
        update running_sum
        update output_accumulator

    output_tile = output_accumulator / running_sum
```

Each query tile is processed against multiple
key/value tiles.

The output is finalized after all relevant
key/value tiles have been processed.

## 8. Key Takeaways

1. Attention uses two major GEMM operations:
   - Q multiplied by K transpose.
   - P multiplied by V.

2. Standard attention can create large N x N
   intermediate matrices.

3. FlashAttention uses tiling to process smaller
   portions of the attention matrix.

4. Online softmax maintains running statistics
   instead of requiring the entire row at once.

5. FlashAttention reduces global-memory traffic
   without eliminating quadratic attention arithmetic.

The most important principle:

**FlashAttention optimizes data movement, not the
fundamental mathematical complexity of attention.**


# Lesson 5.4 — CUDA Kernel Fusion

## 1. What Is a CUDA Kernel?

A CUDA kernel is a function executed by GPU threads.

For example:

$$
C = A + B
$$

A simple CUDA kernel assigns one thread to calculate
one output element.

## 2. The Problem: Too Many Small Kernels

Consider this calculation:

$$
Y = (A+B)\times C
$$

Without fusion, we use two kernels.

### Kernel 1: Addition

$$
T = A+B
$$

The intermediate result T is written to global memory.

### Kernel 2: Multiplication

$$
Y = T\times C
$$

The GPU must read T from global memory again.

This creates two problems:

- Extra global-memory transfers.
- Additional kernel launch overhead.

## 3. The Solution: Kernel Fusion

Combine both operations into one kernel.

$$
Y = (A+B)\times C
$$

Example CUDA code:

```cpp
__global__ void fused(
    float* A,
    float* B,
    float* C,
    float* Y,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float temp = A[i] + B[i];
        Y[i] = temp * C[i];
    }
}
```

The intermediate variable `temp` can remain in
a register instead of being written to global memory.

## 4. Memory Transfer Comparison

Assume:
- Each array contains N elements.
- Each element is a 32-bit float (4 bytes).

### Without Fusion

| Operation | Data transferred |
|---|---:|
| Read A | N |
| Read B | N |
| Write T | N |
| Read T | N |
| Read C | N |
| Write Y | N |
| Total | 6N |

Total memory traffic:

$$
6N \times 4 = 24N \text{ bytes}
$$

### With Fusion

| Operation | Data transferred |
|---|---:|
| Read A | N |
| Read B | N |
| Read C | N |
| Write Y | N |
| Total | 4N |

Total memory traffic:

$$
4N \times 4 = 16N \text{ bytes}
$$

Reduction:

$$
\frac{24N-16N}{24N}\times100\%
=33.3\%
$$

Fusion reduces the simplified global-memory traffic
by approximately 33.3%.

It also reduces kernel launches from two to one.

## 5. Where Does the Intermediate Result Go?

In the fused kernel:

```cpp
float temp = A[i] + B[i];
Y[i] = temp * C[i];
```

The temporary variable can normally be stored
in a register.

GPU memory hierarchy:

1. Registers: private to each thread.
2. Shared memory: shared within a block.
3. Global memory: large, but higher latency.

Avoiding unnecessary global-memory writes
is a major benefit of fusion.

## 6. Kernel Fusion in PyTorch

PyTorch code:

```python
y = (a + b) * c
```

These operations may execute as separate GPU kernels.

Using `torch.compile` can allow the compiler
to fuse compatible operations.

Example:

```python
import torch

def calculation(a, b, c):
    return (a + b) * c

compiled_calculation = torch.compile(calculation)

a = torch.randn(1_000_000, device="cuda")
b = torch.randn(1_000_000, device="cuda")
c = torch.randn(1_000_000, device="cuda")

y = compiled_calculation(a, b, c)
```

Fusion is one optimization that `torch.compile`
may apply. It is not guaranteed for every operation.

## 7. Where Is Kernel Fusion Useful?

Common examples:

- Element-wise operations.
- ReLU and GELU activation functions.
- Bias addition followed by activation.
- Normalization operations.
- Combinations of reductions and pointwise operations.

For example:

$$
Y = \text{ReLU}(XW^T+b)
$$

An optimized implementation may combine compatible
operations to reduce intermediate memory traffic.

## 8. Is Kernel Fusion Always Better?

No. There are trade-offs.

### Register Pressure

A larger fused kernel may require more registers.

This can reduce the number of resident blocks
on an SM and lower occupancy.

### Reduced Parallelism

Combining operations can sometimes reduce
opportunities for independent parallel execution.

### Kernel Complexity

Large fused kernels can be harder to optimize
and maintain.

### Different Computational Requirements

Matrix multiplication may benefit from a highly
optimized library kernel. Fusing it with other
operations is not necessarily faster.

The correct rule:

Fusion is useful when saved memory traffic and
kernel launch overhead outweigh its costs.

## 9. Compare Three Optimization Techniques

| Technique | Main objective |
|---|---|
| GEMM tiling | Reuse loaded data |
| FlashAttention | Avoid large intermediate matrices |
| Kernel fusion | Avoid intermediate writes and extra launches |

All three techniques address an important GPU problem:
moving data between memory and computation units.

## 10. Key Takeaways

1. Kernel fusion combines multiple operations
   into one GPU kernel.
2. It reduces intermediate global-memory traffic.
3. It reduces kernel launch overhead.
4. Intermediate values can remain in registers.
5. Excessive fusion can increase register pressure
   and reduce occupancy.
6. Fusion is not always faster; measure performance.

The most important principle:

**Combine operations when the saved memory traffic
and launch overhead outweigh the costs of fusion.**



# Lesson 5.5 — Triton and Custom GPU Kernels

## 1. What Is Triton?

Triton is an open-source GPU programming language and compiler
originally developed by OpenAI.

It allows programmers to write custom GPU kernels using a
Python-like syntax.

Compared with CUDA C++, Triton provides a higher-level
programming abstraction.

| Technology | Programming approach |
|---|---|
| CUDA C++ | Explicit threads and blocks |
| Triton | Program instances and data tiles |
| PyTorch | High-level tensor operations |

Triton makes many low-level GPU programming tasks easier,
but understanding GPU architecture is still important.

## 2. CUDA C++ vs Triton

### CUDA C++

A typical CUDA kernel assigns work to individual threads.

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < N) {
    C[i] = A[i] + B[i];
}
```

Each thread calculates one element.

### Triton

Triton describes operations on a block of elements.

```python
pid = tl.program_id(0)

offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

mask = offsets < N

a = tl.load(A + offsets, mask=mask)
b = tl.load(B + offsets, mask=mask)

c = a + b

tl.store(C + offsets, c, mask=mask)
```

Important concepts:

- `tl.program_id(0)`: identifies the current program instance.
- `tl.arange()`: generates element indices.
- `tl.load()`: loads data from GPU memory.
- `tl.store()`: writes data to GPU memory.
- `mask`: prevents invalid memory accesses.

A Triton program instance processes a tile of data.

The compiler maps its operations onto GPU threads and hardware.

## 3. Program Instances and Data Tiles

Suppose:

```python
BLOCK_SIZE = 256
N = 1000
```

The number of program instances is:

```python
triton.cdiv(N, BLOCK_SIZE)
```

Therefore:

\[
\text{Programs} = \lceil 1000/256 \rceil = 4
\]

The work is divided as follows:

- Program 0: elements 0–255
- Program 1: elements 256–511
- Program 2: elements 512–767
- Program 3: elements 768–999

The final program has 24 unused positions.

The mask handles these positions safely.

```python
mask = offsets < N
```

Important distinction:

A Triton program instance is a logical unit of work.
It is not necessarily one GPU thread or one physical SM.

The compiler determines how its work is distributed across
GPU threads and warps.

## 4. Memory Operations

Triton provides operations such as:

```python
a = tl.load(A + offsets)
b = tl.load(B + offsets)

c = a + b

tl.store(C + offsets, c)
```

The compiler can organize intermediate values in registers
and optimize memory operations.

Remember the GPU memory hierarchy:

1. Registers: fastest, private to a thread.
2. Shared memory: on-chip memory shared within a block.
3. Global memory: large GPU memory with higher latency.

Understanding memory movement is essential for optimization.

## 5. Kernel Fusion with Triton

Consider:

\[
Y = (A+B)\times C
\]

With separate operations, intermediate results may be written
to and read from global memory.

With Triton, we can implement the entire calculation
in one kernel.

```python
a = tl.load(A + offsets, mask=mask)
b = tl.load(B + offsets, mask=mask)
c = tl.load(C + offsets, mask=mask)

result = (a + b) * c

tl.store(Y + offsets, result, mask=mask)
```

The intermediate value can remain in registers.

Benefits:

- Fewer kernel launches.
- Less intermediate global-memory traffic.
- Potentially better performance.

This is an application of kernel fusion from Lesson 5.4.

## 6. Complete Vector Addition Example

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(A, B, C, N: tl.constexpr,
               BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < N

    a = tl.load(A + offsets, mask=mask)
    b = tl.load(B + offsets, mask=mask)

    c = a + b

    tl.store(C + offsets, c, mask=mask)


def vector_add(a, b):

    n = a.numel()
    output = torch.empty_like(a)

    grid = lambda meta: (
        triton.cdiv(n, meta["BLOCK_SIZE"]),
    )

    add_kernel[grid](
        a, b, output,
        n,
        BLOCK_SIZE=256
    )

    return output
```

The kernel launch syntax is:

```python
add_kernel[grid](...)
```

The grid determines the number of program instances.

## 7. CUDA C++ vs Triton

| Feature | CUDA C++ | Triton |
|---|---|---|
| Programming level | Lower-level | Higher-level |
| Work abstraction | Threads and blocks | Program instances |
| Memory control | More explicit | More compiler-managed |
| Development | More detailed | Often faster |
| Hardware control | More direct | Less direct |

Triton does not automatically outperform CUDA C++.

CUDA C++ provides more direct hardware control, while Triton
simplifies many custom GPU kernel implementations.

## 8. When to Use Triton

| Workload | Good starting point |
|---|---|
| Standard tensor operations | PyTorch |
| Custom fused operations | Triton |
| Specialized low-level kernels | CUDA C++ |
| Standard matrix multiplication | cuBLAS / PyTorch |
| Advanced GEMM development | CUTLASS |

Triton is particularly useful for AI workloads involving
combinations of tensor operations.

It can also be used to experiment with custom attention
and other GPU algorithms.

## 9. How Triton Fits into the AI Stack

A typical software stack can look like this:

```text
        PyTorch
  Model and tensor operations
           |
           v
         Triton
    Custom GPU kernels
           |
           v
      CUDA Runtime
           |
           v
       GPU Hardware
```

PyTorch, Triton, and CUDA can work together.

Triton is not a replacement for CUDA or PyTorch.
It provides another way to program the GPU.

## 10. Running Triton

Triton generally requires a compatible NVIDIA GPU and
software environment.

Google Colab can provide remote GPU access when an
appropriate GPU runtime is available.

Check GPU availability with:

```python
import torch

print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

## 11. Key Takeaways

1. Triton provides a Python-like way to write GPU kernels.
2. Triton programs operate on blocks or tiles of data.
3. The compiler maps logical operations onto GPU threads.
4. `tl.load()` and `tl.store()` perform memory operations.
5. Masks protect against out-of-bounds memory accesses.
6. Triton can implement kernel fusion.
7. Triton simplifies kernel development but does not
   guarantee better performance than CUDA C++.

## 12. Connection to Previous Lessons

| Technique | Main optimization |
|---|---|
| GEMM tiling | Reuse loaded data |
| FlashAttention | Avoid large intermediate matrices |
| Kernel fusion | Reduce memory traffic and launches |
| Triton | Simplify custom GPU kernel development |

The central principle remains:

**High-performance GPU programming requires balancing
computation, memory movement, and parallel execution.**



# CUDA Course — Complete Summary

## 1. CUDA Fundamentals

### 1.1 GPU Execution Model

CUDA organizes parallel work into three levels:

- **Thread:** Executes one instance of the kernel.
- **Block:** A group of threads that can cooperate.
- **Grid:** A collection of blocks launched by a kernel.

Important built-in variables:

```cpp
threadIdx.x  // Thread index inside its block
blockIdx.x   // Block index inside the grid
blockDim.x   // Number of threads in a block
gridDim.x    // Number of blocks in the grid
```

Global thread index:

```cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```

A warp normally contains 32 threads.

The CPU (host) manages the program, while the GPU
(device) executes CUDA kernels.

### 1.2 GPU Memory Hierarchy

| Memory | Scope | Main characteristic |
|---|---|---|
| Registers | Thread | Fast, private |
| Shared memory | Block | Fast, shared |
| Global memory | Device | Large, higher latency |
| Local memory | Thread | Often backed by device memory |
| Constant memory | Device | Read-only, cached |

Important principles:

- Keep frequently used values in registers.
- Use shared memory to reuse data across threads.
- Reduce unnecessary global-memory accesses.
- Use coalesced memory access whenever possible.

### 1.3 Occupancy and Latency Hiding

Occupancy:

```text
Occupancy = Active warps / Maximum supported warps
```

The GPU switches between ready warps to hide memory
and instruction latency.

Occupancy is limited by:

- Registers per thread.
- Shared memory per block.
- Threads per block.
- Maximum resident blocks and warps.

Higher occupancy does not always mean higher performance.
Data reuse can justify lower occupancy.

### 1.4 Synchronization and Communication

Important CUDA mechanisms:

- `__syncthreads()`: Synchronizes threads in a block.
- `__syncwarp()`: Synchronizes threads in a warp.
- `__shfl_sync()`: Exchanges register values within a warp.
- `atomicAdd()`: Performs an atomic addition.
- `cudaDeviceSynchronize()`: Makes the CPU wait for GPU work.

Remember:

- `__syncthreads()` cannot synchronize different blocks.
- Warp shuffle is useful for warp-level reductions.
- Shared memory plus barriers supports communication
  between warps in the same block.

### 1.5 Parallel Reduction

Reduction combines many values into fewer values.

Example:

```text
Input:  [1, 2, 3, 4, 5, 6, 7, 8]
Output: 36
```

A tree reduction combines values in multiple stages.

Useful techniques:

- Accumulate several values in registers.
- Use warp shuffle for warp-level reduction.
- Use shared memory for block-level communication.
- Use multiple kernel launches for large global reductions.

---

## 2. Practical CUDA Programming

### 2.1 Host and Device Programming

Typical CUDA workflow:

1. Allocate host memory.
2. Allocate device memory.
3. Copy input data to the GPU.
4. Launch a CUDA kernel.
5. Copy results back to the CPU.
6. Release allocated memory.

Important APIs:

```cpp
cudaMalloc()
cudaMemcpy()
cudaFree()
```

Memory transfer directions:

- HostToDevice
- DeviceToHost
- DeviceToDevice

### 2.2 Asynchronous Execution

CUDA kernel launches are generally asynchronous
from the CPU's perspective.

Important concepts:

- A stream is an ordered sequence of GPU operations.
- Different streams may execute concurrently.
- Async memory transfers can overlap with computation.
- Pinned host memory supports reliable asynchronous
  host-device transfers.

### 2.3 CUDA Events and Timing

CUDA events measure GPU execution time.

```cpp
cudaEventRecord(start);

// GPU operations

cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);
```

Do not rely on a CPU timer around an asynchronous
kernel launch without synchronization.

### 2.4 Error Handling and Debugging

Important APIs:

```cpp
cudaGetLastError()
cudaPeekAtLastError()
cudaGetErrorString()
```

Useful debugging tools:

- Compute Sanitizer memcheck: Invalid memory accesses.
- Racecheck: Shared-memory race conditions.
- Initcheck: Uninitialized memory reads.
- Synccheck: Synchronization errors.

---

## 3. GPU Performance Optimization

### 3.1 Occupancy Optimization

Occupancy helps hide latency by keeping multiple
warps ready to execute.

However, maximizing occupancy is not the objective.

The real objective is maximizing useful GPU work.

### 3.2 Memory Coalescing

When neighboring threads access neighboring addresses,
the GPU can use memory transactions efficiently.

Good pattern:

```cpp
output[tid] = input[tid];
```

Potentially inefficient pattern:

```cpp
output[tid] = input[tid * 32];
```

Memory coalescing is especially important for
memory-bandwidth-bound workloads.

### 3.3 Shared Memory and Tiling

Shared memory allows threads in a block to reuse data.

Typical workflow:

1. Load a tile from global memory.
2. Store it in shared memory.
3. Synchronize threads.
4. Reuse the tile for computation.
5. Load the next tile.

Tiling is especially useful for matrix multiplication.

### 3.4 Advanced Optimization

Important techniques:

- Instruction-level parallelism (ILP).
- Loop unrolling.
- Warp specialization.
- Asynchronous data movement.
- Double buffering.
- Tensor Core operations.

These techniques are workload-dependent.

### 3.5 Profiling

Two important tools:

- **Nsight Systems:** Shows CPU/GPU execution timelines.
- **Nsight Compute:** Analyzes individual GPU kernels.

A useful optimization workflow:

1. Verify correctness.
2. Measure baseline performance.
3. Profile the workload.
4. Identify the bottleneck.
5. Choose one optimization.
6. Measure again.
7. Keep the change only if it helps.

---

## 4. CUDA and PyTorch Integration

### 4.1 Python, PyTorch, and CUDA

Python usually orchestrates computation.

PyTorch manages tensors and dispatches supported
operations to optimized GPU libraries and kernels.

A typical flow:

```text
Python
  |
PyTorch
  |
CUDA runtime / GPU libraries
  |
GPU hardware
```

### 4.2 GPU Tensor Operations

A tensor can be moved to the GPU:

```python
x = torch.randn(1000)
x = x.to("cuda")
```

Operations involving CUDA tensors can execute
on the GPU when supported.

### 4.3 Autograd

PyTorch automatically calculates gradients.

Typical training sequence:

```python
optimizer.zero_grad()

prediction = model(x)
loss = loss_function(prediction, target)

loss.backward()
optimizer.step()
```

Meaning:

- `zero_grad()`: Clears old gradients.
- Forward pass: Calculates predictions.
- Loss: Measures prediction error.
- `backward()`: Calculates gradients.
- `step()`: Updates model parameters.

### 4.4 GPU Memory and Training Efficiency

Important concepts:

- Parameters.
- Gradients.
- Optimizer states.
- Activations.
- Batch size.
- Gradient accumulation.
- Activation checkpointing.

GPU memory consumption is not limited to model
parameters. Training also stores gradients,
activations, and optimizer states.

### 4.5 Mixed Precision

Floating-point formats:

| Format | Sign | Exponent | Fraction |
|---|---:|---:|---:|
| FP32 | 1 | 8 | 23 |
| FP16 | 1 | 5 | 10 |
| BF16 | 1 | 8 | 7 |

Key differences:

- FP32: High precision and wide range.
- FP16: More fraction precision than BF16,
  but a narrower numerical range.
- BF16: Similar numerical range to FP32,
  but less fraction precision.

Mixed precision uses lower-precision operations
where appropriate to improve performance and reduce
memory consumption.

FP16 training may need gradient scaling to prevent
small gradients from underflowing.

BF16 generally does not need gradient scaling.

### 4.6 Complete PyTorch Training Pipeline

A typical training pipeline:

```text
Data loading
    |
CPU preprocessing
    |
CPU-to-GPU transfer
    |
Forward pass
    |
Loss calculation
    |
Backward pass
    |
Optimizer update
```

Potential bottlenecks:

- Slow data loading.
- CPU-to-GPU transfers.
- GPU computation.
- GPU memory capacity.
- Kernel launch overhead.

CUDA knowledge helps identify and optimize these
bottlenecks.

---

## 5. Advanced GPU Optimization for AI

### 5.1 GPU Performance Bottlenecks

Two major bottleneck categories:

**Compute-bound**

Performance is limited by computational throughput.

Examples:
- Large matrix multiplication.
- Neural network projection layers.

**Memory-bound**

Performance is limited by data movement.

Examples:
- Tensor copying.
- Vector addition.
- Some reduction operations.

Arithmetic intensity:

```text
Arithmetic Intensity = FLOPs / Bytes transferred
```

Roofline model:

```text
Performance <= min(
    Peak Compute Performance,
    Memory Bandwidth * Arithmetic Intensity
)
```

### 5.2 GEMM Optimization

GEMM means General Matrix Multiplication.

```text
C = A * B
```

For matrix dimensions:

```text
A: M x K
B: K x N
C: M x N
```

The main optimization technique is tiling.

Instead of repeatedly loading individual values
from global memory, the GPU loads tiles into
shared memory and reuses them.

Tensor Cores accelerate supported matrix operations.

PyTorch commonly uses optimized libraries such as
cuBLAS for standard matrix multiplication.

### 5.3 Attention and FlashAttention

Transformer attention:

```text
S = Q * K^T / sqrt(d_k)
P = softmax(S)
O = P * V
```

Standard attention may materialize large intermediate
score and probability matrices.

FlashAttention uses:

- Tiling.
- Online softmax.
- Register and shared-memory reuse.
- Reduced global-memory traffic.

Its main achievement is reducing memory movement.

It does not eliminate the quadratic attention
computation.

### 5.4 CUDA Kernel Fusion

Kernel fusion combines multiple operations
into one GPU kernel.

Example:

```text
Y = (A + B) * C
```

Without fusion:

1. Calculate A + B.
2. Write intermediate result to memory.
3. Read intermediate result.
4. Multiply by C.

With fusion:

1. Load A, B, and C.
2. Calculate the result in one kernel.
3. Write Y.

Benefits:

- Fewer kernel launches.
- Less intermediate memory traffic.
- Potentially better performance.

However, excessive fusion can increase register
pressure and reduce occupancy.

### 5.5 Triton

Triton is a Python-like language for writing
custom GPU kernels.

Comparison:

| Technology | Main purpose |
|---|---|
| PyTorch | High-level tensor operations |
| Triton | Custom GPU kernels |
| CUDA C++ | Low-level GPU programming |
| cuBLAS | Optimized matrix operations |
| CUTLASS | Custom high-performance GEMM |

Triton uses program instances and tiles.

The compiler maps these logical operations to
GPU threads and hardware resources.

Triton makes custom kernels easier to develop,
but does not guarantee better performance than
optimized CUDA libraries.

### 5.6 Putting Everything Together

A complete optimization workflow:

1. Establish a correct baseline.
2. Measure execution time and memory consumption.
3. Profile the application.
4. Identify the dominant bottleneck.
5. Select the appropriate optimization.
6. Implement the change.
7. Verify numerical correctness.
8. Measure end-to-end performance.
9. Keep the optimization only if it provides
   a meaningful improvement.

Useful technique-to-problem mapping:

| Problem | Possible technique |
|---|---|
| Too many global-memory transactions | Coalescing |
| Repeated data loading | Shared-memory tiling |
| Poor latency hiding | Occupancy tuning |
| Warp communication overhead | Warp shuffle |
| Transfer and compute serialization | Streams |
| Slow matrix multiplication | GEMM tiling, Tensor Cores |
| Large attention intermediates | FlashAttention |
| Excessive intermediate tensors | Kernel fusion |
| High memory consumption | Mixed precision |
| Need for custom GPU operations | Triton or CUDA C++ |

---

## 6. The Most Important Principles

### Principle 1: Data Movement Matters

GPU performance is not only about how many
calculations the GPU can perform.

Moving data between memory levels can be expensive.

### Principle 2: Reuse Data

Load data once and reuse it whenever possible.

Examples:
- Shared-memory tiling.
- Register accumulation.
- GEMM optimization.
- FlashAttention.

### Principle 3: Reduce Unnecessary Work

Avoid unnecessary:
- Kernel launches.
- Intermediate tensors.
- Memory transfers.
- Synchronization.
- Repeated calculations.

### Principle 4: Profile Before Optimizing

Never assume which part is slow.

Measure first, identify the bottleneck,
and optimize the part that matters.

### Principle 5: Optimize the Whole Application

A kernel that becomes twice as fast does not
necessarily make the whole application twice as fast.

Amdahl's Law:

```text
Overall speedup =
    1 / ((1 - P) + P / S)
```

Where:
- P = Fraction of execution time being optimized.
- S = Speedup of that part.

The overall improvement is limited by the
portion of the workload that remains unchanged.

---

## 7. Recommended Next Step: Practical AI Optimization

The next phase should apply the CUDA knowledge
to a real AI workload.

A good starting project is Transformer inference.

Suggested progression:

1. Establish a PyTorch inference baseline.
2. Profile the model.
3. Identify GEMM and attention costs.
4. Compare FP32, FP16, and BF16 where supported.
5. Compare standard attention with optimized attention.
6. Investigate kernel fusion.
7. Try a custom Triton kernel.
8. Compare latency, throughput, and memory usage.

The objective is not to rewrite everything in CUDA.

The objective is to understand the performance
bottleneck and choose the right optimization.
