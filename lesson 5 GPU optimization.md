
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
