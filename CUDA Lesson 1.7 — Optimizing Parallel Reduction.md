
# Lesson 1.7 — Optimizing Parallel Reduction

## 1. Why Optimize Parallel Reduction?

In Lesson 1.6, we learned how to calculate the sum of
an array using parallel reduction.

A basic implementation:

1. Each thread loads one element.
2. Threads combine values using shared memory.
3. Each block produces one partial sum.
4. A later kernel combines the partial sums.

However, this approach can be improved.

The main optimization goals are:

- Reduce unnecessary memory operations.
- Reduce synchronization overhead.
- Improve memory access efficiency.
- Make better use of registers and GPU resources.

## 2. Multiple Elements per Thread

Instead of processing one element per thread, we can
let each thread process multiple elements.

For example, suppose we have 1,024 elements and
256 threads.

Each thread processes four elements:

$$
1024/256=4
$$

Thread 0 processes:

$$
A[0]+A[256]+A[512]+A[768]
$$

Thread 1 processes:

$$
A[1]+A[257]+A[513]+A[769]
$$

Each thread calculates a local sum in its registers.

Example:

```cpp
float sum = 0.0f;

for (int i = tid; i < N; i += blockDim.x) {
    sum += input[i];
}
```

After this loop, each thread has a local sum.

These local sums are then combined using parallel
reduction.

### Advantages

- More work is performed by each thread.
- Intermediate results can stay in registers.
- Fewer partial sums may be needed.

### Disadvantages

- Register usage may increase.
- Occupancy may decrease.
- Excessive work per thread can reduce parallelism.

The goal is to balance work per thread and GPU utilization.

## 3. Coalesced Global-Memory Access

Memory access patterns strongly affect GPU performance.

Suppose 32 threads in a warp read consecutive elements:

```cpp
value = input[globalId];
```

Their accesses are:

| Lane | Memory address |
|---:|---|
| 0 | `input[0]` |
| 1 | `input[1]` |
| 2 | `input[2]` |
| ... | ... |
| 31 | `input[31]` |

These accesses are adjacent.

The GPU can combine them into a small number of
efficient memory transactions.

This is called **coalesced memory access**.

By contrast:

```cpp
value = input[globalId * 32];
```

causes threads to access widely separated addresses.

This may require more memory transactions.

### Key principle

Organize memory access so neighboring warp lanes read
neighboring memory locations.

Efficient memory access is often more important than
reducing the number of arithmetic operations.

## 4. Optimized Reduction Structure

An efficient reduction kernel typically follows this process:

1. Each thread loads multiple input elements.
2. Each thread calculates a local sum in registers.
3. Warp shuffle combines values within each warp.
4. One lane from each warp writes its partial sum
   to shared memory.
5. `__syncthreads()` ensures all warp sums are available.
6. One warp combines the partial sums.
7. One thread writes the block's final partial sum
   to global memory.

This is called hierarchical reduction.

## 5. Different Memory and Communication Roles

| Mechanism | Purpose |
|---|---|
| Registers | Store thread-local values and sums |
| Warp shuffle | Exchange values within a warp |
| Shared memory | Exchange results between warps |
| Global memory | Store input data and block partial sums |

The design principle is:

Keep intermediate calculations close to the threads
performing them.

Avoid unnecessary movement to global memory.

## 6. Memory Bandwidth as a Bottleneck

Summation performs very little arithmetic per input element.

For each `float` element, we typically perform:

- One global-memory read.
- One floating-point addition.

A `float` occupies 4 bytes.

For one million elements:

$$
1,000,000 \times 4 = 4,000,000\ bytes
$$

Approximately 4 MB must be read.

If memory bandwidth is 500 GB/s, the idealized
minimum transfer time is:

$$
T = \frac{4\ MB}{500\ GB/s}
$$

$$
T \approx 8\ \mu s
$$

This is only a theoretical estimate.

Actual execution also includes:

- Kernel launch overhead.
- Synchronization.
- Cache behavior.
- Memory transaction efficiency.
- Reduction operations.

For simple reductions, memory bandwidth can be the
main performance bottleneck.

## 7. Three Important Optimization Trade-offs

### A. More Elements per Thread

Advantages:
- More local work.
- Fewer partial results.
- Better use of register accumulation.

Disadvantages:
- Higher register usage.
- Potentially lower occupancy.
- Reduced parallelism if taken too far.

### B. Larger Block Size

Advantages:
- More threads process data in one block.

Disadvantages:
- Higher resource requirements per block.
- Fewer blocks may fit on an SM.
- Occupancy may decrease.

### C. Synchronization

Synchronization ensures correctness but has overhead.

Warp shuffle can reduce the need for block-wide
barriers during warp-level reduction.

However, shared memory and synchronization are still
needed to combine results across warps.

## 8. Performance Measurement

We should compare different implementations:

1. Basic shared-memory reduction.
2. Warp-shuffle reduction.
3. Multiple elements per thread.
4. Combined optimized reduction.

Measure execution time and verify numerical correctness.

Do not assume an optimization is faster just because
it reduces instructions or synchronization.

Floating-point addition is not perfectly associative:

$$
(a+b)+c \neq a+(b+c)
$$

Different reduction orders can produce slightly
different floating-point results due to rounding.

This is normal if the numerical error is within
the required tolerance.

## 9. Key Takeaways

1. Each thread can process multiple input elements.
2. Registers are useful for local accumulation.
3. Coalesced access improves global-memory efficiency.
4. Warp shuffle combines values within a warp.
5. Shared memory combines results between warps.
6. Each block produces one partial sum.
7. A later kernel combines block partial sums.
8. Excessive register usage can reduce occupancy.
9. Memory bandwidth can be the main bottleneck.
10. Always measure performance instead of assuming
    an optimization is faster.

## 10. What Comes Next?

### Lesson 1.8 — GPU Performance Measurement

We will study:

- CUDA event timing.
- Kernel execution time.
- Memory bandwidth.
- Compute-bound vs. memory-bound workloads.
- Occupancy and performance analysis.
- How to identify the actual bottleneck.
