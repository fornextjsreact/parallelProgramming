
# Lesson 3.2 — Memory Coalescing

## 1. What Is Memory Coalescing?

A GPU Warp typically contains 32 threads.

When threads access Global Memory, the GPU organizes
their memory requests into memory transactions.

If neighboring threads access neighboring memory
addresses, the GPU can combine their requests into
fewer transactions.

This is called **Memory Coalescing**.

Example:

```cpp
float x = input[threadIdx.x];
```

For a Warp of 32 threads:

| Thread | Memory Address |
|---|---|
| 0 | input[0] |
| 1 | input[1] |
| 2 | input[2] |
| 3 | input[3] |
| ... | ... |
| 31 | input[31] |

Each float occupies 4 bytes.

Total requested data:

$$
32 \times 4 = 128\ bytes
$$

The addresses are consecutive, allowing the GPU to
serve the requests efficiently.

## 2. Why Does Coalescing Matter?

GPU memory access is organized into memory transactions.

A memory transaction transfers a region of memory,
rather than treating every thread as an independent
full memory transfer.

When addresses are consecutive and properly aligned,
fewer transactions may be needed.

When addresses are scattered, more transactions may
be required.

The key principle:

**Memory Coalescing reduces unnecessary memory traffic.**

Actual transaction counts depend on:

- GPU architecture
- Access width
- Address alignment
- Cache behavior
- Memory transaction granularity

## 3. Coalesced vs. Strided Access

### Coalesced Access

```cpp
__global__ void coalesced(
    float* input,
    float* output
) {
    int tid =
        blockIdx.x * blockDim.x + threadIdx.x;

    output[tid] = input[tid];
}
```

Neighboring threads access consecutive elements:

```text
Thread 0  -> input[0]
Thread 1  -> input[1]
Thread 2  -> input[2]
...
Thread 31 -> input[31]
```

This is an efficient access pattern.

### Strided Access

```cpp
__global__ void strided(
    float* input,
    float* output
) {
    int tid =
        blockIdx.x * blockDim.x + threadIdx.x;

    output[tid] = input[tid * 32];
}
```

Now the addresses are:

```text
Thread 0  -> input[0]
Thread 1  -> input[32]
Thread 2  -> input[64]
Thread 3  -> input[96]
...
Thread 31 -> input[992]
```

Neighboring threads access memory locations
separated by 32 floats.

Each thread requests only 4 bytes, but the address
span across the Warp is:

$$
32 \times 32 \times 4 = 4096\ bytes
$$

This can generate many unnecessary memory transactions.

Note: 4096 bytes is the address span, not the actual
number of bytes transferred.

## 4. Matrix Transpose and Coalescing

Consider a matrix:

$$
A =
\begin{bmatrix}
1&2&3&4\\
5&6&7&8\\
9&10&11&12\\
13&14&15&16
\end{bmatrix}
$$

Its transpose is:

$$
B = A^T =
\begin{bmatrix}
1&5&9&13\\
2&6&10&14\\
3&7&11&15\\
4&8&12&16
\end{bmatrix}
$$

Assume Row-major storage.

Elements in each row are stored consecutively:

```text
A[0][0], A[0][1], A[0][2], A[0][3],
A[1][0], A[1][1], A[1][2], A[1][3], ...
```

A simple transpose operation is:

```cpp
B[col][row] = A[row][col];
```

For neighboring threads:

- Reading A can be coalesced.
- Writing B may be uncoalesced.

This is because neighboring threads write elements
separated by an entire row in B.

### Shared Memory Solution

Use Shared Memory as an intermediate storage area.

1. Read a tile from A using coalesced Global Memory access.
2. Store the tile in Shared Memory.
3. Rearrange data inside Shared Memory.
4. Read the rearranged tile.
5. Write B using coalesced Global Memory access.

This allows both Global Memory reads and writes
to be efficient.

This technique is called **Tiled Matrix Transpose**.

## 5. Relationship Between Coalescing and Tiling

Tiling and Coalescing solve different problems.

| Technique | Main Purpose |
|---|---|
| Tiling | Divide large data into manageable blocks |
| Coalescing | Reduce unnecessary Global Memory transactions |
| Shared Memory | Exchange and reuse data efficiently |

They are often used together.

For example, FlashAttention uses Tiling to process
small attention blocks and carefully organizes memory
accesses to improve data-transfer efficiency.

## 6. Important Concepts

| Concept | Meaning |
|---|---|
| Memory Transaction | A hardware memory-access operation |
| Coalesced Access | Neighboring lanes access nearby addresses |
| Uncoalesced Access | Requests are spread across memory regions |
| Strided Access | Neighboring lanes access addresses separated by a stride |
| Row-major | Matrix elements are stored row by row |
| Tiling | Divide data into smaller blocks |
| Shared Memory | Fast on-chip storage for reuse and communication |

## 7. Key Takeaways

1. A Warp typically contains 32 threads.
2. Neighboring threads should access neighboring addresses.
3. Coalesced access can reduce memory transactions.
4. Strided access may waste memory bandwidth.
5. Matrix Transpose can have uncoalesced writes.
6. Shared Memory can rearrange data to improve access patterns.
7. Tiling and Coalescing are complementary optimization techniques.

## 8. Connection to the Next Lesson

Next: **Lesson 3.3 — Shared Memory Optimization**

We will study:

- Shared Memory data reuse
- Shared Memory bank organization
- Bank Conflicts
- Padding techniques
- Tiled matrix operations

