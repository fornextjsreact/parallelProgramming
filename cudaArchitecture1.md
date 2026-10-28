Here is a concise reference note you can save and review later.

# CUDA Learning — Step 1.1: GPU Architecture Fundamentals

## 1. GPU Hardware Hierarchy

**GPU → SM → Execution units**

- **GPU (Graphics Processing Unit):** The entire processor that executes parallel workloads.
- **SM (Streaming Multiprocessor):** A major processing unit inside the GPU. It schedules and executes groups of threads.
- **CUDA core:** An arithmetic execution unit that performs operations such as addition and multiplication.
- **Warp scheduler:** Selects ready warps and issues their instructions to the SM's execution pipelines.
- **Registers:** Fast, on-chip storage for each thread's variables and intermediate results.
- **Shared memory:** Fast, on-chip memory shared by threads in the same thread block.

An SM also contains other hardware, such as load/store units and, on modern GPUs, Tensor Cores.

## 2. What Is a Thread?

A CUDA thread is a **logical execution unit**, not a physical core.

Each thread:

- Executes a sequence of instructions.
- Has its own thread index and logical execution state.
- Processes its own data.
- Uses registers for its private variables.

Threads are scheduled onto the GPU's physical execution resources; they are not permanently assigned to individual CUDA cores.

## 3. What Is a Warp?

A **warp normally contains 32 threads**.

CUDA uses the SIMT execution model:

**SIMT = Single Instruction, Multiple Threads**

Threads in a warp generally execute the same instruction together, each operating on its own data.

Example:

```
C[i] = A[i] + B[i]
```

For a warp of 32 threads:

- Thread 0 calculates `A[0] + B[0]`.
- Thread 1 calculates `A[1] + B[1]`.
- ...
- Thread 31 calculates `A[31] + B[31]`.

The hardware can execute these additions in parallel through its execution pipelines.

## 4. Important Distinctions

- **Thread ≠ CUDA core:** A thread is logical; a CUDA core is hardware.
- **Warp ≠ SM:** A warp is a group of threads; an SM executes and schedules warps.
- **64 CUDA cores ≠ maximum 64 threads:** An SM can maintain many more resident threads than it has CUDA cores.
- **32-thread warp ≠ exactly 32 cores used every cycle:** Actual execution depends on the instruction and GPU architecture.

## 5. Mental Model

Imagine a factory:

- GPU = the entire factory.
- SM = a production unit.
- Warp = a team of 32 workers following the same instruction.
- Thread = an individual worker handling its own data.
- CUDA core = an arithmetic machine used to perform calculations.

This analogy is useful, but remember that real GPU scheduling and execution are more complex.

## 6. Knowledge Check

**Question:** An SM has 64 CUDA cores. A warp contains 32 threads. What happens when the warp executes a simple addition instruction?

**Answer:** The 32 threads execute the addition instruction on their respective data. The hardware schedules their work through the SM's execution pipelines. It does not follow that all 64 CUDA cores must be used by every instruction.

## 7. Next Lesson

Study the CUDA programming hierarchy:

**Grid → Blocks → Threads**

Learn how a kernel launch creates blocks, how blocks are assigned to SMs, and how threads within each block cooperate.



Key takeaway: CUDA organizes computation into threads, groups threads into warps, 
and schedules warps on SMs containing physical execution units. 
Understanding this hierarchy is the foundation for learning CUDA programming and optimization.
