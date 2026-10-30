
# Lesson 2.1 — Writing Your First Complete CUDA Program

## 1. CPU and GPU Execution

A CUDA program uses two execution environments:

- **Host (CPU):** Runs the main program and manages GPU resources.
- **Device (GPU):** Executes CUDA kernels in parallel.

The CPU prepares data, transfers it to the GPU, launches
kernels, and retrieves the results.

### Typical CUDA Workflow

1. Allocate and initialize input data in CPU memory.
2. Allocate memory on the GPU.
3. Copy input data from CPU to GPU.
4. Launch a CUDA kernel.
5. Copy results from GPU to CPU.
6. Release GPU memory.

## 2. Vector Addition Example

Suppose we have two arrays:

$$
A = [1, 2, 3, 4]
$$

$$
B = [10, 20, 30, 40]
$$

We want to calculate:

$$
C[i] = A[i] + B[i]
$$

The expected result is:

$$
C = [11, 22, 33, 44]
$$

Each GPU thread calculates one element independently.

## 3. CUDA Kernel

```cpp
__global__ void vectorAdd(
    float* A,
    float* B,
    float* C,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
```

### Important Concepts

**`__global__`**

Declares a CUDA kernel.

- Called from the CPU.
- Executed on the GPU.

**Global thread index**

$$
i = blockIdx.x \times blockDim.x + threadIdx.x
$$

This calculates the unique global index of each thread.

**Boundary checking**

```cpp
if (i < N)
```

Prevents threads from accessing elements outside
the input arrays.

## 4. CPU Memory vs GPU Memory

CPU and GPU memory are separate memory spaces
in a typical discrete GPU system.

For example:

```cpp
float h_A[N];       // CPU memory
float* d_A;         // GPU memory
```

Naming convention:

- `h_` means host memory.
- `d_` means device memory.

These pointers refer to different memory locations.

## 5. GPU Memory Allocation

```cpp
cudaMalloc(&d_A, bytes);
```

Allocates memory in GPU device memory.

Example:

```cpp
const int N = 4;
size_t bytes = N * sizeof(float);

float* d_A;
cudaMalloc(&d_A, bytes);
```

For four floating-point numbers, the allocation
requires 16 bytes, assuming `float` is 4 bytes.

## 6. Data Transfer

CUDA uses `cudaMemcpy()` to transfer data.

### CPU to GPU

```cpp
cudaMemcpy(
    d_A,
    h_A,
    bytes,
    cudaMemcpyHostToDevice
);
```

### GPU to CPU

```cpp
cudaMemcpy(
    h_C,
    d_C,
    bytes,
    cudaMemcpyDeviceToHost
);
```

### GPU to GPU

```cpp
cudaMemcpy(
    d_B,
    d_A,
    bytes,
    cudaMemcpyDeviceToDevice
);
```

The four arguments are:

1. Destination pointer.
2. Source pointer.
3. Number of bytes.
4. Transfer direction.

## 7. Kernel Launch Syntax

```cpp
vectorAdd<<<1, 4>>>(d_A, d_B, d_C, N);
```

General syntax:

```cpp
kernel<<<numberOfBlocks, threadsPerBlock>>>(arguments);
```

The example launches:

- 1 block.
- 4 threads per block.
- 4 total threads.

For multiple blocks, CUDA calculates thread indices
using both `blockIdx.x` and `threadIdx.x`.

## 8. Thread Execution Example

For:

```cpp
vectorAdd<<<1, 4>>>(d_A, d_B, d_C, N);
```

| Thread | threadIdx.x | Result |
|---|---:|---:|
| Thread 0 | 0 | 1 + 10 = 11 |
| Thread 1 | 1 | 2 + 20 = 22 |
| Thread 2 | 2 | 3 + 30 = 33 |
| Thread 3 | 3 | 4 + 40 = 44 |

All threads execute the same kernel code,
but each thread operates on different data.

This is the foundation of CUDA parallel programming.

## 9. Synchronization and Execution

CUDA kernel launches are normally asynchronous
from the CPU's perspective.

For example:

```cpp
vectorAdd<<<1, 4>>>(d_A, d_B, d_C, N);
```

The CPU can continue executing without waiting
for the GPU kernel to finish.

However, a blocking device-to-host copy:

```cpp
cudaMemcpy(
    h_C,
    d_C,
    bytes,
    cudaMemcpyDeviceToHost
);
```

waits for the preceding GPU work needed to produce
the copied result.

Therefore, the CPU can safely use `h_C` after
the copy completes.

## 10. Releasing GPU Memory

Memory allocated with `cudaMalloc()` should be
released using `cudaFree()`.

```cpp
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
```

This prevents unnecessary GPU memory consumption.

## 11. Compile and Run

Save the program as:

```text
vector_add.cu
```

Compile:

```bash
nvcc vector_add.cu -o vector_add
```

Run on Linux:

```bash
./vector_add
```

Run on Windows:

```powershell
.\vector_add.exe
```

Expected output:

```text
11.000000 22.000000 33.000000 44.000000
```

## 12. Key Takeaways

1. CUDA programs typically use both CPU and GPU.
2. The CPU manages memory and kernel execution.
3. GPU kernels perform parallel computation.
4. `__global__` declares a GPU kernel.
5. `cudaMalloc()` allocates GPU memory.
6. `cudaMemcpy()` transfers data between memory spaces.
7. `<<<blocks, threads>>>` specifies the execution
   configuration.
8. Each thread calculates its own global index.
9. Kernel launches are normally asynchronous.
10. `cudaFree()` releases allocated GPU memory.

## 13. Important Mental Model

Think of CUDA as a system where:

- CPU prepares and manages the work.
- GPU receives the data.
- GPU threads perform calculations in parallel.
- CPU retrieves and uses the results.

The performance of a CUDA application depends not
only on how quickly the GPU calculates, but also
on how efficiently data moves between CPU and GPU.
