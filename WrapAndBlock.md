# CUDA Learning — Lesson 1.2: Understanding Warps

## 1. Blocks and Warps

A block is a logical group of threads. A warp is the unit NVIDIA GPUs
normally schedule and execute together.

- A warp normally contains 32 threads.
- A block can contain multiple warps.
- A block with 256 threads contains 8 warps.
- CUDA hardware organizes threads into warps automatically.

## 2. How Warps Execute Instructions

Consider this CUDA kernel statement:

    C[i] = A[i] + B[i];

Each thread executes the same instruction on different data.

For example, threads 0–31 in warp 0 calculate:

    C[0]  = A[0]  + B[0]
    C[1]  = A[1]  + B[1]
    ...
    C[31] = A[31] + B[31]

The warp is scheduled as a group under NVIDIA's SIMT execution model.
This does not guarantee that all 32 operations finish in one clock cycle.

## 3. Why Warps Don't Appear Explicitly in Code

CUDA programmers typically write code for individual threads:

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] + B[i];

The hardware automatically groups threads into warps and schedules
their instructions.

Conceptual hierarchy:

    Grid
      └── Blocks
            └── Threads
                  └── Organized into warps for execution

A block is a programming concept; a warp is an execution group.

## 4. When Warps Matter

### A. Warp Divergence

    if (i % 2 == 0) {
        C[i] = A[i] + B[i];
    } else {
        C[i] = A[i] - B[i];
    }

If threads in a warp take different branches, the GPU may execute
both paths with different sets of active threads. This can reduce
execution efficiency.

### B. Memory Access

Neighboring threads accessing neighboring memory addresses often
allow memory requests to be combined. This is called coalesced
memory access.

### C. Warp-Level Operations

CUDA provides operations such as `__shfl_sync()` for exchanging
values between threads in a warp. These can help implement reductions
and other parallel algorithms efficiently.

## 5. Key Takeaways

- A warp normally contains 32 threads.
- Threads in a warp execute under the SIMT model.
- CUDA creates and schedules warps automatically.
- Warp behavior affects branching, memory access, and performance.
- Understanding warps helps when optimizing CUDA kernels.

## Next Lesson

Learn how registers and shared memory limit the number of warps
that can reside on an SM, and how this relates to occupancy.
