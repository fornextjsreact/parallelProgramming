
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

