# CUDA Basics: Grid, Block, Thread, and Warp

## 1. Execution Hierarchy

- **Grid:** All blocks launched by one kernel.
- **Block:** A group of threads that can cooperate.
- **Thread:** Executes the kernel code for its own data.
- **Warp:** A group of 32 threads normally scheduled together.

Conceptual hierarchy:

    Grid
      └── Blocks
            └── Threads
                  └── Organized into warps for execution

## 2. Example

Suppose we process 1,000 array elements using blocks of 256 threads.

- Threads per block: 256
- Warps per block: 256 / 32 = 8
- Number of blocks: ceil(1000 / 256) = 4
- Total launched threads: 4 × 256 = 1,024
- Threads doing useful work: 1,000

Extra threads must skip work when their indices exceed the array size.

## 3. Global Thread Index

CUDA provides these built-in variables:

- `threadIdx.x`: Thread index within its block.
- `blockIdx.x`: Block index within the grid.
- `blockDim.x`: Number of threads per block.
- `gridDim.x`: Number of blocks in the grid.

Global index formula:

    i = blockIdx.x * blockDim.x + threadIdx.x

Example: block 2, thread 7, with 256 threads per block:

    i = 2 * 256 + 7 = 519

This thread processes array element 519.

## 4. Simple CUDA Kernel

    __global__ void vector_add(
        const float* A,
        const float* B,
        float* C,
        int N
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < N) {
            C[i] = A[i] + B[i];
        }
    }

The boundary check prevents out-of-range memory accesses.

## 5. How the GPU Executes the Work

- The programmer defines the grid and block dimensions.
- The GPU schedules blocks onto available SMs.
- Each block is divided into warps automatically.
- Warps execute instructions under the SIMT model.
- An SM may host multiple blocks and warps, depending on resources.

## 6. Key Distinctions

| Concept | Role |
|---|---|
| Grid | Organizes all blocks in a kernel launch |
| Block | Groups cooperating threads |
| Thread | Executes the kernel for its data |
| Warp | Hardware execution and scheduling group |
| SM | Physical GPU processing unit |
| CUDA core | Arithmetic execution hardware |

**Remember:** Threads are logical workers; CUDA cores are hardware.
Threads are not permanently assigned to individual CUDA cores.
