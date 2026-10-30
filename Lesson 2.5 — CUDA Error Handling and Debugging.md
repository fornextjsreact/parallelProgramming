
# Lesson 2.5 — CUDA Error Handling and Debugging

## 1. Two Categories of CUDA Errors

CUDA errors can be divided into two broad categories.

### A. Host-side CUDA API Errors

These occur when calling CUDA runtime functions from the CPU.

For example:

```cpp
cudaError_t err = cudaMalloc(&d_A, bytes);
```

Possible causes include:

- Insufficient GPU memory.
- Invalid arguments.
- Unsupported operations.
- Device initialization problems.

Most CUDA runtime functions return a `cudaError_t` value
indicating success or failure.

### B. Kernel Execution Errors

These occur while the GPU executes a kernel.

For example:

```cpp
__global__ void myKernel(float* data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    data[i] = 1.0f;
}
```

If the allocated array contains only 100 elements but
we launch 256 threads, threads with indices 100–255
access memory outside the allocated array.

This is an out-of-bounds memory access.

Because kernel launches are normally asynchronous,
the error may not be reported immediately.

## 2. Checking CUDA API Errors

CUDA runtime functions return error codes.

For example:

```cpp
cudaError_t err = cudaMalloc(&d_A, bytes);

if (err != cudaSuccess) {
    printf("Allocation failed: %s\n",
           cudaGetErrorString(err));
}
```

Important functions:

| Function | Purpose |
|---|---|
| `cudaGetLastError()` | Returns and clears the last recorded CUDA error |
| `cudaPeekAtLastError()` | Returns the last recorded error without clearing it |
| `cudaGetErrorString()` | Converts an error code into readable text |
| `cudaDeviceSynchronize()` | Waits for device work and reports execution errors |

Checking return values is important because otherwise
the program may continue using an invalid pointer or
an incomplete result.

## 3. Detecting Kernel Launch Errors

Consider:

```cpp
myKernel<<<blocks, threads>>>(d_A, d_B, d_C);
```

Immediately after launching the kernel, we can check
for launch-related errors:

```cpp
cudaError_t err = cudaGetLastError();

if (err != cudaSuccess) {
    printf("Kernel launch error: %s\n",
           cudaGetErrorString(err));
}
```

This can detect errors such as an invalid launch
configuration.

However, it does not guarantee that the kernel
executed successfully.

The kernel may still be running on the GPU.

## 4. Detecting Kernel Execution Errors

To detect errors that occur during execution,
we need a synchronization point.

```cpp
myKernel<<<blocks, threads>>>(d_A, d_B, d_C);

cudaError_t err = cudaGetLastError();

if (err != cudaSuccess) {
    printf("Launch error: %s\n",
           cudaGetErrorString(err));
}

err = cudaDeviceSynchronize();

if (err != cudaSuccess) {
    printf("Execution error: %s\n",
           cudaGetErrorString(err));
}
```

The two checks have different purposes:

1. `cudaGetLastError()` checks for recorded launch errors.
2. `cudaDeviceSynchronize()` waits for device execution
   and can report asynchronous execution errors.

Important: CUDA errors can sometimes surface at later
runtime calls or synchronization points. An error reported
at synchronization is not necessarily caused by the
immediately preceding line of code.

## 5. A Reusable CUDA Error-Checking Macro

Writing an error check after every CUDA API call
can become repetitive.

We can define a macro:

```cpp
#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                __FILE__, __LINE__,                           \
                cudaGetErrorString(err));                     \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)
```

Now we can write:

```cpp
CUDA_CHECK(cudaMalloc(&d_A, bytes));

CUDA_CHECK(cudaMemcpy(
    d_A, h_A, bytes, cudaMemcpyHostToDevice
));

myKernel<<<blocks, threads>>>(d_A);

CUDA_CHECK(cudaGetLastError());

CUDA_CHECK(cudaDeviceSynchronize());

CUDA_CHECK(cudaFree(d_A));
```

The macro uses:

- `__FILE__`: current source filename.
- `__LINE__`: current source line number.
- `cudaGetErrorString()`: readable error description.

This makes error handling easier to maintain.

## 6. A Common Kernel Bug: Out-of-Bounds Access

Suppose:

```cpp
int N = 100;
```

But we launch:

```cpp
myKernel<<<1, 256>>>(d_A);
```

The kernel calculates:

```cpp
int i = threadIdx.x;
d_A[i] = 1.0f;
```

The problem:

- Thread 0 accesses `d_A[0]`.
- Thread 99 accesses `d_A[99]`.
- Thread 100 accesses `d_A[100]`, outside the allocation.
- Threads 101–255 also access invalid locations.

The correct kernel includes a boundary check:

```cpp
__global__ void myKernel(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        data[i] = 1.0f;
    }
}
```

The condition `if (i < N)` protects against extra
threads in the final block.

## 7. CUDA Debugging and Profiling Tools

### NVIDIA Compute Sanitizer

Compute Sanitizer detects several classes of GPU errors.

Run:

```bash
compute-sanitizer ./my_program
```

Useful modes:

| Mode | Purpose |
|---|---|
| `memcheck` | Detect invalid memory accesses |
| `racecheck` | Detect shared-memory data hazards |
| `initcheck` | Detect uninitialized global-memory reads |
| `synccheck` | Detect synchronization misuse |

For example:

```bash
compute-sanitizer --tool memcheck ./my_program
```

### Nsight Systems

Nsight Systems helps analyze:

- CPU/GPU execution timelines.
- CUDA streams.
- Memory transfers.
- Synchronization.
- CPU/GPU idle time.

### Nsight Compute

Nsight Compute focuses on individual kernel performance:

- Memory access behavior.
- Occupancy.
- Instruction execution.
- Resource utilization.
- Kernel bottlenecks.

These tools serve different purposes. Compute Sanitizer
focuses on correctness; Nsight tools focus primarily
on performance analysis.

## 8. Key Takeaways

1. CUDA errors include host API errors and kernel execution
   errors.
2. CUDA runtime functions usually return `cudaError_t`.
3. `cudaGetLastError()` retrieves and clears the last
   recorded CUDA error.
4. `cudaPeekAtLastError()` retrieves the error without
   clearing it.
5. Kernel launch errors and execution errors are different.
6. `cudaDeviceSynchronize()` waits for device execution
   and can expose asynchronous execution errors.
7. Boundary checks such as `if (i < N)` prevent
   out-of-bounds accesses.
8. Compute Sanitizer helps detect memory, race, and
   synchronization errors.
9. Nsight Systems and Nsight Compute help analyze
   performance bottlenecks.

The main lesson:

**A successful kernel launch does not necessarily mean
the kernel executed correctly. Always check errors at
appropriate points in your CUDA program.**
