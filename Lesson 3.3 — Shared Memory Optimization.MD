
# Lesson 3.3 — Shared Memory Optimization

## 1. What Is Shared Memory?

CUDA provides several levels of memory:

| Memory | Scope | Main Purpose |
|---|---|---|
| Registers | Individual thread | Thread-local variables |
| Shared memory | Thread block | Data sharing and reuse |
| Global memory | Entire GPU | Large data storage |

Shared memory is:
- Located on-chip, within an SM.
- Accessible by all threads in the same block.
- Explicitly managed by the programmer.
- Much faster to access than global memory in typical use.
- Limited in capacity.

The main idea is to load data from global memory once and
reuse it through shared memory.

## 2. Why Do We Need Shared Memory?

Consider matrix multiplication:

$$
C=A\times B
$$

Each output element is:

$$
C_{ij}=\sum_{k=0}^{K-1}A_{ik}B_{kj}
$$

Different threads may need the same elements of A and B.

Without shared memory, threads may repeatedly fetch the same
data from global memory.

Shared memory allows threads in a block to reuse previously
loaded data, reducing global-memory traffic.

This is particularly useful in:
- Matrix multiplication.
- Matrix transpose.
- Image processing.
- Deep-learning operations.
- Attention mechanisms.

## 3. Tiling

Tiling divides a large matrix into smaller submatrices.

For each tile:

1. Threads load a tile from matrix A into shared memory.
2. Threads load a tile from matrix B into shared memory.
3. Threads synchronize.
4. Threads calculate partial results using shared memory.
5. Threads synchronize before loading the next tile.

The process repeats until the output tile is complete.

The key benefit is data reuse: multiple threads can use the
same data loaded into shared memory.

## 4. Tiled Matrix Multiplication

Example shared-memory declarations:

```cpp
#define TILE 16

__shared__ float As[TILE][TILE];
__shared__ float Bs[TILE][TILE];
```

For TILE = 16:

- As contains 256 floats.
- Bs contains 256 floats.
- Total shared memory is 512 floats, or 2 KB.

Each thread loads one element from A and one element from B.

The block collectively loads two tiles and reuses their
elements to calculate output values.

The main calculation is:

```cpp
for (int k = 0; k < TILE; k++) {
    sum += As[threadIdx.y][k] *
           Bs[k][threadIdx.x];
}
```

Each thread calculates one output element by accumulating
the products of one row of A and one column of B.

## 5. Synchronization

Two synchronization points are important in tiled
matrix multiplication.

### First synchronization

```cpp
__syncthreads();
```

This ensures all threads have finished loading the current
tiles before any thread starts consuming them.

Without it, a thread might read data that another thread
has not finished writing.

### Second synchronization

```cpp
__syncthreads();
```

This ensures all threads have finished using the current
tiles before shared memory is overwritten by the next tiles.

Remember:

- First barrier: data is ready before reading.
- Second barrier: data is no longer in use before overwriting.

`__syncthreads()` synchronizes threads within one block.
It does not synchronize different blocks.

## 6. Shared Memory Bank Conflicts

Shared memory is divided into smaller units called banks.

A simplified model of a GPU with 32 banks:

- Consecutive 32-bit words map to consecutive banks.
- Different lanes accessing different addresses in the
  same bank can cause serialization.
- Multiple lanes reading the same address can often use
  a broadcast mechanism.

Example:

```cpp
__shared__ float tile[32][32];
```

Consider:

```cpp
float value = tile[threadIdx.x][0];
```

For 32 threads, addresses are separated by 32 floats.

In the simplified 32-bank model, these addresses can all
map to the same bank, causing a bank conflict.

### Padding

A common solution is to change the array dimensions:

```cpp
__shared__ float tile[32][33];
```

The row stride changes from 32 to 33 words.

This changes the bank mapping, allowing the column accesses
to be distributed across different banks.

Padding is commonly used in optimized matrix transpose
operations.

Actual bank behavior depends on GPU architecture, access
width, and data type.

## 7. Shared Memory and Occupancy

Shared memory is a limited resource on each SM.

Suppose an SM has 64 KB available for shared memory.

| Shared Memory per Block | Theoretical Blocks |
|---|---:|
| 8 KB | 8 |
| 16 KB | 4 |
| 32 KB | 2 |

These are simplified calculations that ignore other hardware
limits.

More shared memory per block can reduce the number of
resident blocks on an SM.

This may reduce occupancy.

Therefore, shared-memory optimization involves a trade-off:

- More shared memory can reduce global-memory traffic.
- Excessive shared-memory usage can reduce occupancy.

The best configuration depends on the workload and GPU.

## 8. Connection to FlashAttention

Standard attention calculates:

$$
S=QK^T
$$

$$
P=\operatorname{softmax}(S)
$$

$$
O=PV
$$

The intermediate matrices can be very large.

FlashAttention uses tiling to avoid repeatedly moving large
intermediate matrices between global memory and on-chip
memory.

It loads small Q, K, and V tiles, performs calculations,
and maintains intermediate results on-chip.

Shared memory is one of the important resources used in
this process, alongside registers and other on-chip
mechanisms.

## 9. Key Takeaways

1. Shared memory is on-chip memory shared by threads in
   the same block.
2. Its main purpose is to enable data reuse and reduce
   repeated global-memory accesses.
3. Tiling divides large computations into smaller pieces
   that fit into shared memory.
4. `__syncthreads()` is necessary when threads cooperate
   through shared memory.
5. Bank conflicts can serialize shared-memory accesses.
6. Padding can help avoid common bank conflicts.
7. Excessive shared-memory usage can reduce occupancy.
8. Shared memory is an important optimization technique
   in matrix multiplication and FlashAttention.

## 10. The Bigger Picture

GPU optimization is not only about performing arithmetic
faster.

It is also about minimizing unnecessary data movement
between memory levels.

The goal is to keep frequently reused data close to the
computing units while maintaining enough parallelism to
keep the GPU busy.
