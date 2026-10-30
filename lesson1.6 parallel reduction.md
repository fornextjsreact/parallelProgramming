
# Lesson 1.6 — Parallel Reduction

## 1. What Is Parallel Reduction?

Reduction is the process of combining many input values
into fewer output values.

Common examples:

- Sum: add all elements.
- Maximum: find the largest element.
- Minimum: find the smallest element.
- Average: calculate the sum and divide by the count.

For example:

$$
A = [2,4,6,8,10,12,14,16]
$$

The sum is:

$$
Sum(A) = 72
$$

The main idea is to divide the calculation among many
threads and combine their partial results.

## 2. Sequential vs. Parallel Reduction

### Sequential reduction

A CPU can calculate the sum one element at a time:

$$
2+4+6+8+10+12+14+16=72
$$

For $N$ elements, sequential addition requires
$N-1$ additions.

### Parallel reduction

A GPU can calculate independent partial sums simultaneously.

Stage 1:

$$
\begin{aligned}
2+4 &= 6\\
6+8 &= 14\\
10+12 &= 22\\
14+16 &= 30
\end{aligned}
$$

Stage 2:

$$
6+14=20
$$

$$
22+30=52
$$

Stage 3:

$$
20+52=72
$$

For eight elements, the tree reduction requires
three stages:

$$
\log_2(8)=3
$$

In general, the number of stages is approximately:

$$
\log_2(N)
$$

The important advantage is that independent additions
can happen simultaneously.

## 3. How Threads Divide the Work

Suppose we have eight elements and eight CUDA threads.

Each thread initially loads one element:

| Thread | Element index | Value |
|---|---:|---:|
| T0 | 0 | 2 |
| T1 | 1 | 4 |
| T2 | 2 | 6 |
| T3 | 3 | 8 |
| T4 | 4 | 10 |
| T5 | 5 | 12 |
| T6 | 6 | 14 |
| T7 | 7 | 16 |

Each thread stores its value in a register.

The threads then combine their values through
a sequence of reduction stages.

## 4. Shared-Memory Reduction

A simple CUDA implementation uses shared memory
to exchange intermediate results.

```cpp
__global__ void reduceSum(
    float* input,
    float* partialSum
) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;

    // Each thread loads one element.
    sdata[tid] = input[tid];

    __syncthreads();

    // Tree reduction.
    for (int stride = 1;
         stride < 256;
         stride *= 2) {

        if (tid % (2 * stride) == 0) {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    // Thread 0 writes the final result.
    if (tid == 0) {
        partialSum[0] = sdata[0];
    }
}
```

This example assumes:

- Exactly 256 threads per block.
- At least 256 input elements.
- A valid output pointer.

### Execution process

1. Each thread loads one input element.
2. Each thread stores its value in shared memory.
3. `__syncthreads()` ensures all initial writes are complete.
4. Threads combine partial sums in stages.
5. Thread 0 writes the final sum.

### Reduction strides

For a 256-thread block:

| Iteration | Stride | Active threads |
|---|---:|---:|
| 1 | 1 | 128 |
| 2 | 2 | 64 |
| 3 | 4 | 32 |
| 4 | 8 | 16 |
| 5 | 16 | 8 |
| 6 | 32 | 4 |
| 7 | 64 | 2 |
| 8 | 128 | 1 |

After eight iterations, thread 0 holds the final sum.

## 5. Why Use Shared Memory?

During reduction, threads need to exchange intermediate
results.

Different memory types have different roles:

| Memory | Role in reduction |
|---|---|
| Registers | Store each thread's private values |
| Warp shuffle | Exchange values between warp lanes |
| Shared memory | Exchange values across warps in a block |
| Global memory | Store partial results for later blocks |

Shared memory is useful because threads within a block
can access a common, relatively fast memory space.

The goal is to minimize unnecessary global-memory
access and synchronization.

## 6. Reduction of a Large Array

A large array cannot normally be handled by one block.

Instead, divide the array among multiple blocks.

Each block calculates a partial sum:

$$
S_i = \sum_{j \in Block_i} A_j
$$

The final result is:

$$
S = \sum_{i=0}^{B-1} S_i
$$

where $B$ is the number of blocks.

A common implementation uses two kernel launches:

**Kernel 1:**

- Each block processes part of the input array.
- Each block calculates one partial sum.
- Each block writes its result to global memory.

**Kernel 2:**

- Threads combine all partial sums.
- The final result is written to memory.

Why two kernels?

Because `__syncthreads()` only synchronizes threads
within one block. Ordinary CUDA blocks cannot use it
to synchronize the entire grid.

## 7. Important Performance Concepts

Parallel reduction performance depends on several factors:

1. **Parallelism:** Independent additions can execute
   simultaneously.
2. **Memory access:** Reading input efficiently is important.
3. **Shared memory:** Reduces unnecessary global-memory
   communication.
4. **Synchronization:** Barriers have a cost.
5. **Warp-level communication:** Shuffle operations can
   reduce shared-memory usage and synchronization overhead.

A basic shared-memory reduction is useful for learning,
but optimized implementations use more efficient techniques.

## 8. Key Takeaways

1. Reduction combines many values into fewer results.
2. Tree reduction performs independent additions in parallel.
3. The number of reduction stages grows as $\log_2(N)$.
4. Each thread initially loads its own input value.
5. Shared memory allows threads in a block to exchange
   intermediate results.
6. `__syncthreads()` ensures threads reach a synchronization
   point before proceeding.
7. Large arrays are divided among multiple blocks.
8. Each block produces a partial sum.
9. A later stage combines the partial sums.
10. Ordinary `__syncthreads()` cannot synchronize
    different blocks.

## 9. What Comes Next?

### Lesson 1.7 — Optimizing Parallel Reduction

We will improve the basic reduction kernel by studying:

- Warp-level reduction.
- `__shfl_down_sync()`.
- Avoiding unnecessary synchronization.
- Memory coalescing.
- Reduction performance and GPU utilization.

The goal is to understand how a simple reduction algorithm
becomes an efficient GPU implementation.
