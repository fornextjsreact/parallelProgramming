
# Lesson 2.3 — CUDA Streams and Asynchronous Execution

## 1. Why Do We Need CUDA Streams?

A typical CUDA program performs three operations:

1. Copy input data from CPU to GPU.
2. Execute a CUDA kernel.
3. Copy results from GPU to CPU.

A sequential execution pattern looks like this:

```text
Copy input → Execute kernel → Copy output
```

Each operation waits for the preceding operation to finish.

This can leave GPU resources idle. For example, while data is
being transferred, the GPU's compute resources may not be doing
useful work.

CUDA streams allow us to organize operations so that independent
operations can potentially overlap.

## 2. What Is a CUDA Stream?

A CUDA stream is an ordered sequence of operations submitted
to the GPU.

For example:

```cpp
cudaMemcpyAsync(
    d_A, h_A, bytes,
    cudaMemcpyHostToDevice, stream1
);

vectorAdd<<<blocks, threads, 0, stream1>>>(
    d_A, d_B, d_C, N
);

cudaMemcpyAsync(
    h_C, d_C, bytes,
    cudaMemcpyDeviceToHost, stream1
);
```

These operations belong to `stream1`.

CUDA preserves their order:

1. Copy input data.
2. Execute the kernel.
3. Copy output data.

The kernel cannot execute before the preceding copy in the
same stream has completed.

### Same Stream vs. Different Streams

- Operations in the same stream execute in submission order.
- Operations in different streams may execute concurrently.
- Different streams do not guarantee simultaneous execution.

Actual overlap depends on hardware capability, resource
availability, memory type, and dependencies.

## 3. Creating and Managing Streams

### Create a stream

```cpp
cudaStream_t stream1;

cudaStreamCreate(&stream1);
```

`cudaStream_t` is the CUDA type used to represent a stream.

### Launch a kernel in a stream

Normal kernel launch:

```cpp
vectorAdd<<<blocks, threads>>>(A, B, C, N);
```

Kernel launch with an explicit stream:

```cpp
vectorAdd<<<blocks, threads, 0, stream1>>>(
    A, B, C, N
);
```

The launch configuration is:

```cpp
<<<numberOfBlocks, threadsPerBlock,
   sharedMemoryBytes, stream>>>
```

The third parameter specifies the amount of dynamically
allocated shared memory. Here, it is zero.

### Submit an asynchronous memory copy

```cpp
cudaMemcpyAsync(
    d_A,
    h_A,
    bytes,
    cudaMemcpyHostToDevice,
    stream1
);
```

This submits the transfer to the specified stream.

### Wait for a stream

```cpp
cudaStreamSynchronize(stream1);
```

The CPU waits until all previously submitted operations
in that stream have completed.

### Destroy a stream

```cpp
cudaStreamDestroy(stream1);
```

This releases the stream resource.

## 4. Pinned Host Memory

An important detail:

`cudaMemcpyAsync()` does not automatically guarantee
asynchronous transfer or copy/compute overlap.

For reliable asynchronous transfers between host and device,
pinned (page-locked) host memory is generally required.

Ordinary CPU memory allocated with `malloc()` is pageable.
The CUDA runtime may need to stage its contents through
an internal pinned buffer.

Pinned memory remains resident in physical memory, making
it suitable for asynchronous transfers.

Allocate pinned memory with:

```cpp
float* h_A;

cudaMallocHost(&h_A, bytes);
```

Or:

```cpp
cudaHostAlloc(&h_A, bytes, cudaHostAllocDefault);
```

Memory allocation comparison:

| Function | Memory location |
|---|---|
| `malloc()` | Ordinary CPU memory |
| `cudaMallocHost()` | Pinned CPU memory |
| `cudaMalloc()` | GPU memory |

Pinned memory is a limited system resource. Pinning too much
host memory can reduce overall system performance.

## 5. Pipelining and Double Buffering

Suppose a large array is divided into four chunks:

- Chunk A
- Chunk B
- Chunk C
- Chunk D

A sequential implementation performs:

```text
Copy A → Compute A
Copy B → Compute B
Copy C → Compute C
Copy D → Compute D
```

A pipelined implementation can potentially overlap copying
and computation:

```text
Time →

Copy:      A       B       C       D
Compute:           A       B       C       D
```

While the GPU computes chunk A, it may copy chunk B.

This technique is called pipelining.

With multiple streams, we can submit independent work:

```cpp
cudaMemcpyAsync(
    d_A, h_A, bytes,
    cudaMemcpyHostToDevice, stream1
);

processKernel<<<blocks, threads, 0, stream1>>>(d_A);

cudaMemcpyAsync(
    d_B, h_B, bytes,
    cudaMemcpyHostToDevice, stream2
);

processKernel<<<blocks, threads, 0, stream2>>>(d_B);
```

This is a simplified example. Actual overlap requires:

- Independent operations.
- Suitable pinned host memory.
- Hardware support for the required concurrency.
- No conflicting dependencies or resource constraints.

Buffers must not be reused or overwritten until operations
using them have completed.

## 6. Host Asynchrony vs. GPU Concurrency

These are two different concepts.

### Host-side asynchronous behavior

The CPU submits an operation and continues executing
without waiting for that operation to finish.

### GPU-side overlap

A memory transfer and a kernel actually execute
during overlapping time intervals.

Host-side asynchronous behavior does not automatically
mean GPU operations overlap.

For example, `cudaMemcpyAsync()` may allow the CPU to continue,
but the GPU might still execute the copy and kernel sequentially.

## 7. Important CUDA APIs

| API | Purpose |
|---|---|
| `cudaStreamCreate()` | Create a stream |
| `cudaMemcpyAsync()` | Submit asynchronous memory copy |
| `cudaStreamSynchronize()` | Wait for one stream |
| `cudaStreamDestroy()` | Destroy a stream |
| `cudaMallocHost()` | Allocate pinned host memory |
| `cudaHostAlloc()` | Allocate pinned host memory |

## 8. Key Takeaways

1. A stream is an ordered sequence of CUDA operations.
2. Operations in the same stream execute in submission order.
3. Different streams can potentially execute concurrently.
4. Multiple streams do not guarantee simultaneous execution.
5. Pinned host memory is generally needed for reliable
   asynchronous host-device transfers and overlap.
6. `cudaStreamSynchronize()` waits for a specific stream.
7. Pipelining overlaps data transfer and computation to
   improve GPU utilization.
8. Correct buffer management and synchronization are essential.

The main objective of CUDA streams is to reduce idle time
by allowing independent operations to overlap when the
hardware supports it.
