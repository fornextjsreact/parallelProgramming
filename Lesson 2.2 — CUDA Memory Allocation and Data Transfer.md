
# Lesson 2.2 — CUDA Memory Allocation and Data Transfer

## 1. CPU Memory vs GPU Memory

A typical computer with a discrete NVIDIA GPU has
two separate memory systems:

- **System RAM:** Used by the CPU.
- **GPU VRAM:** Used by the GPU.

They are physically separate memory spaces.

Example:

```cpp
float h_A[1000];  // CPU memory
float* d_A;       // GPU memory pointer
```

Naming convention:

- `h_` means host memory.
- `d_` means device memory.

The GPU cannot normally use an ordinary CPU pointer
as a pointer to its own device memory.

Therefore, CUDA requires explicit memory allocation
and data transfer.

## 2. What Does cudaMalloc() Do?

```cpp
float* d_A;

cudaMalloc(&d_A, 1000 * sizeof(float));
```

`cudaMalloc()` requests an allocation in GPU device
memory.

After successful allocation:

- `d_A` contains a device memory address.
- The allocated memory can be accessed by GPU kernels.
- The pointer variable `d_A` itself is normally stored
  in CPU memory.

An important distinction:

The pointer variable is in CPU memory, but the address
it holds refers to GPU memory.

## 3. What Does cudaMemcpy() Do?

Example:

```cpp
float h_A[4] = {1, 2, 3, 4};
float* d_A;

cudaMalloc(&d_A, 4 * sizeof(float));

cudaMemcpy(
    d_A,
    h_A,
    4 * sizeof(float),
    cudaMemcpyHostToDevice
);
```

This copies data from CPU memory to GPU memory.

After copying:

- `h_A` still contains its original values.
- `d_A` contains an independent copy.
- Changing `h_A` does not automatically change `d_A`.

**Important:** `cudaMemcpy()` copies data. It does not
move the original array.

## 4. Four Memory Transfer Directions

| Direction | CUDA parameter | Meaning |
|---|---|---|
| CPU → GPU | `cudaMemcpyHostToDevice` | Upload input |
| GPU → CPU | `cudaMemcpyDeviceToHost` | Download result |
| GPU → GPU | `cudaMemcpyDeviceToDevice` | Copy device data |
| Default | `cudaMemcpyDefault` | Infer direction where supported |

Example:

```cpp
cudaMemcpy(
    d_B,
    d_A,
    bytes,
    cudaMemcpyDeviceToDevice
);
```

This copies data directly between GPU memory
allocations.

## 5. How cudaMemcpy() Works

General syntax:

```cpp
cudaMemcpy(
    destination,
    source,
    numberOfBytes,
    transferDirection
);
```

The four arguments specify:

1. Destination pointer.
2. Source pointer.
3. Number of bytes to transfer.
4. Transfer direction.

For example:

```cpp
cudaMemcpy(
    d_A,
    h_A,
    bytes,
    cudaMemcpyHostToDevice
);
```

The source is `h_A`, and the destination is `d_A`.

## 6. Why Data Transfer Can Become a Bottleneck

A discrete GPU typically communicates with CPU
memory through PCIe.

Although GPUs can perform computations extremely
quickly, transferring large amounts of data between
CPU and GPU takes time.

Transfer time can be approximated by:

$$
T_{\text{transfer}} =
\frac{\text{Data size}}{\text{Bandwidth}}
$$

Example:

- Input size: 100 MB.
- Effective transfer bandwidth: 25 GB/s.
- GPU computation time: 1 ms.

Approximate upload time:

$$
T_{\text{upload}} =
\frac{100\text{ MB}}{25\text{ GB/s}}
\approx 4\text{ ms}
$$

If we also transfer 100 MB back:

$$
T_{\text{total}} \approx 4 + 1 + 4
$$

$$
T_{\text{total}} \approx 9\text{ ms}
$$

Only 1 ms is actual computation.

This demonstrates why data transfer can dominate
the total execution time.

These are illustrative values, not benchmark results.

## 7. Keep Data on the GPU as Long as Possible

Suppose we need to perform ten operations on
the same dataset.

### Inefficient approach

```text
CPU → GPU
Calculate 1
GPU → CPU

CPU → GPU
Calculate 2
GPU → CPU

...
```

This repeatedly transfers the same data.

### Better approach

```text
CPU → GPU

Calculate 1
Calculate 2
Calculate 3
...
Calculate 10

GPU → CPU
```

The second approach reduces unnecessary data transfers.

**Optimization principle:**

Keep data on the GPU as long as possible.

This is especially important in machine learning,
where model parameters and intermediate tensors
are reused across many GPU operations.

## 8. Synchronous vs Asynchronous Transfer

### Synchronous transfer

```cpp
cudaMemcpy(
    d_A,
    h_A,
    bytes,
    cudaMemcpyHostToDevice
);
```

The CPU waits for the copy operation to complete
before continuing past the call.

### Asynchronous transfer

```cpp
cudaMemcpyAsync(
    d_A,
    h_A,
    bytes,
    cudaMemcpyHostToDevice,
    stream
);
```

This allows the CPU to continue without waiting
for the transfer to finish.

Important points:

- Asynchronous transfer does not automatically mean
  faster transfer.
- It allows potential overlap between data transfer
  and GPU computation.
- Reliable asynchronous host transfers and overlap
  generally require pinned (page-locked) host memory.
- CUDA streams help organize asynchronous operations.

Streams will be covered in Lesson 2.3.

## 9. Releasing GPU Memory

Memory allocated using `cudaMalloc()` should
eventually be released.

```cpp
cudaFree(d_A);
```

This returns the allocated device memory to
the CUDA allocator.

Releasing unused allocations helps prevent
unnecessary GPU memory consumption.

## 10. Key Takeaways

1. CPU RAM and GPU VRAM are separate memory spaces
   on a typical discrete GPU system.
2. `cudaMalloc()` allocates memory on the GPU.
3. `d_A` is a CPU-side pointer variable containing
   a device memory address.
4. `cudaMemcpy()` copies data between memory spaces.
5. Copying data does not remove the original data.
6. PCIe transfer overhead can dominate computation.
7. Keeping data on the GPU reduces unnecessary transfers.
8. `cudaMemcpy()` is synchronous from the host's
   perspective.
9. `cudaMemcpyAsync()` allows asynchronous execution.
10. `cudaFree()` releases GPU memory.

## 11. Important Mental Model

Think of GPU programming as two separate activities:

**Data movement:**
- CPU memory → GPU memory.
- GPU memory → CPU memory.

**Computation:**
- GPU threads operate on data already in GPU memory.

Good CUDA performance requires optimizing both.

A fast GPU kernel is not enough if the application
spends most of its time moving data.

The goal is to minimize unnecessary data movement
and maximize useful computation on the GPU.

## 12. Preview of the Next Lesson

Lesson 2.3 — CUDA Streams and Asynchronous Execution

Topics:

- What is a CUDA stream?
- Why CUDA operations are asynchronous.
- How to overlap data transfer and computation.
- What pinned memory is.
- How streams improve GPU utilization.
