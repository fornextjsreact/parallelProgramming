
# Lesson 1.5 — Thread Synchronization and Communication

## 1. Why Do Threads Need Synchronization?

CUDA threads execute in parallel. Sometimes one thread needs to
read data written by another thread.

Without synchronization, a thread might read data before another
thread has finished writing it.

This can cause a **race condition**.

Example:

```cpp
data[threadIdx.x] = threadIdx.x * 2;

// Other threads may not have finished writing yet.

float value = data[100];
```

If a thread reads `data[100]` before the corresponding write
has completed, the result may be incorrect.

## 2. Block-Level Synchronization: `__syncthreads()`

`__syncthreads()` creates a barrier for all participating threads
within the same thread block.

```cpp
__global__ void example(float* output) {
    __shared__ float data[256];

    int tid = threadIdx.x;

    data[tid] = tid * 2.0f;

    __syncthreads();

    if (tid == 0) {
        output[0] = data[100];
    }
}
```

Execution process:

1. Each thread writes its own value into shared memory.
2. All threads reach `__syncthreads()`.
3. Threads wait until the block reaches the barrier.
4. Threads continue execution after synchronization.
5. Thread 0 reads the value written by thread 100.

Important:

- `__syncthreads()` synchronizes threads within one block.
- It also ensures visibility of prior shared-memory writes
  to threads in that block after the barrier.
- All non-exited threads must reach the barrier consistently.
- It does not synchronize different blocks.

## 3. Warp-Level Synchronization

A warp typically contains 32 threads, called lanes.

CUDA provides warp-level operations:

```cpp
__syncwarp();
```

This synchronizes participating lanes within a warp.

Another useful operation is:

```cpp
__shfl_down_sync(mask, value, 1);
```

It allows a lane to obtain a value from another lane.

For example, with a full participating warp:

- Lane 0 can obtain the value from lane 1.
- Lane 1 can obtain the value from lane 2.
- Lane 2 can obtain the value from lane 3.

This is useful for exchanging values without explicitly
using shared memory.

The mask must correctly identify participating lanes.

## 4. Atomic Operations

Consider many threads incrementing the same counter:

```cpp
counter = counter + 1;
```

This operation involves multiple steps:

1. Read the current counter.
2. Add one.
3. Write the result back.

If two threads perform these steps concurrently, one update
may overwrite the other.

This is called a lost update.

CUDA provides atomic operations:

```cpp
atomicAdd(&counter, 1);
```

Each increment is performed atomically.

For example:

- Initial counter = 10
- Number of successful increments = 1000
- Final counter = 1010

Atomic operations protect individual memory updates.
They do not automatically synchronize an entire block or grid.

## 5. Three Important Mechanisms Compared

| Mechanism | Purpose | Scope |
|---|---|---|
| `__syncthreads()` | Synchronize threads at a barrier | Block |
| `__syncwarp()` | Synchronize participating lanes | Warp |
| `atomicAdd()` | Perform an indivisible memory update | Memory location |

Remember:

- A barrier makes threads wait for one another.
- An atomic operation protects a memory update.
- A warp shuffle exchanges values between lanes.

These mechanisms solve different problems.

## 6. Important CUDA Concepts

### Thread and Block Identification

```cpp
threadIdx.x
```

Identifies a thread's position within its block.

```cpp
blockIdx.x
```

Identifies the block's position within the grid.

```cpp
blockDim.x
```

Specifies the number of threads along the block's X dimension.

A global thread index can be calculated as:

```cpp
int globalId =
    blockIdx.x * blockDim.x + threadIdx.x;
```

### Shared Memory and Synchronization

Shared memory is useful when threads in a block need to
exchange or reuse data.

However, shared memory alone does not guarantee that data
is ready when another thread reads it.

Synchronization is often necessary.

## 7. Key Takeaways

1. CUDA threads execute concurrently.
2. Unsynchronized access can cause race conditions.
3. `__syncthreads()` provides block-level synchronization.
4. `__syncwarp()` synchronizes participating warp lanes.
5. Warp shuffle operations allow direct lane-to-lane exchange.
6. Atomic operations prevent lost updates to shared memory locations.
7. Synchronization and atomic operations have different purposes.
8. Synchronization has a performance cost, so use it when necessary.

## 8. What Comes Next?

Lesson 1.6 — Parallel Reduction

We will use multiple threads to calculate the sum of a large
array, combining:

- Thread and block organization
- Global memory access
- Shared memory
- Warp-level communication
- Synchronization

The objective is to understand how GPUs combine many partial
results efficiently.
