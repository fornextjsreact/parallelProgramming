# CUDA Lesson 1.3: Registers, Shared Memory, and Occupancy

## 1. Registers

- Very fast storage located within an SM's execution hardware.
- Each thread has its own register state.
- Used for variables and intermediate calculation results.
- Register resources are limited.
- More registers per thread may reduce the number of resident threads.

## 2. Shared Memory

- Fast, explicitly managed memory located on an SM.
- Shared by threads within the same block.
- Useful for reusing data and reducing global memory accesses.
- Requires explicit allocation and coordination when sharing data.
- Limited in capacity per SM.

Example:

    __shared__ float tile[256];

This declares a shared-memory array accessible to threads in the block.

## 3. Global Memory

- Located in GPU device memory (VRAM).
- Accessible by threads across blocks.
- Has much higher latency than registers and typically higher latency
  than shared memory.
- Provides large capacity for arrays, matrices, and tensors.

## 4. Memory Hierarchy

From a thread's perspective:

    Registers
        ↓
    Shared Memory
        ↓
    Global Memory

Registers are private to each thread. Shared memory is shared within
a block. Global memory is accessible across blocks.

Caches also help reduce the cost of accessing global memory.

## 5. Occupancy

Occupancy is the ratio of active warps on an SM to the maximum number
of warps the SM can support.

    Occupancy = Active warps / Maximum supported warps

Example:

    Active warps = 32
    Maximum warps = 64
    Occupancy = 32 / 64 = 50%

Higher occupancy can help hide memory latency, but does not guarantee
better performance.

## 6. What Limits Occupancy?

Two important resource limits are:

- Registers: More registers per thread can reduce resident threads.
- Shared memory: More shared memory per block can reduce resident blocks.

The GPU must have enough resources to keep additional blocks resident.

## 7. Key Takeaways

- Registers store each thread's private working data.
- Shared memory enables fast data sharing within a block.
- Global memory provides large capacity for GPU data.
- Resource usage affects how many warps can reside on an SM.
- Occupancy helps describe resource utilization, but is not a direct
  measure of performance.

## Next Lesson

Learn how memory latency, memory bandwidth, and occupancy affect
CUDA kernel performance.
