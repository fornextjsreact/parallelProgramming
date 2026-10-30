
# Lesson 3.1 — Occupancy and GPU Resource Utilization

## 1. What is an SM?

A GPU consists of multiple Streaming Multiprocessors (SMs).

An SM is a major execution and resource-management unit
inside a GPU.

Each SM contains resources such as:

- Registers
- Shared Memory
- CUDA Cores
- Warp Schedulers
- Other execution units

A CUDA Block is assigned to one SM and executes there.

An SM can accommodate multiple Blocks simultaneously,
depending on available hardware resources.

### GPU execution hierarchy

- GPU: the entire processor
- SM: an execution unit inside the GPU
- Block: a group of CUDA threads
- Warp: typically 32 threads
- Thread: an individual execution unit

## 2. What is Occupancy?

Occupancy measures the proportion of the maximum possible
resident Warps that are currently active on an SM.

Formula:

$$
Occupancy =
\frac{Active\ Warps}{Maximum\ Warps}
$$

Example: assume an SM supports a maximum of 64 Warps.

| Active Warps | Occupancy |
|---:|---:|
| 16 | 25% |
| 32 | 50% |
| 48 | 75% |
| 64 | 100% |

Active Warps are resident and eligible for scheduling.
They do not necessarily execute instructions at every moment.

## 3. Why Does Occupancy Matter?

Global Memory access has much higher latency than
Register access.

When a Warp waits for data from Global Memory,
it may temporarily be unable to execute dependent instructions.

The GPU can schedule another ready Warp instead.

This technique is called **Latency Hiding**.

Having more active Warps gives the scheduler more
opportunities to execute other work while some Warps
are waiting.

However:

- Higher Occupancy does not guarantee higher performance.
- Occupancy is not the same as GPU utilization.
- A resident Warp is not necessarily doing useful work
  at every moment.

## 4. What Limits Occupancy?

The main resource limitations are:

1. Registers
2. Shared Memory
3. Maximum threads per SM
4. Maximum Warps per SM
5. Maximum Blocks per SM

All resource constraints must be satisfied before a
Block can reside on an SM.

### Example: Register limitation

Assume an SM has:

| Resource | Capacity |
|---|---:|
| Registers | 65,536 (32-bit registers) |
| Maximum threads | 2,048 |
| Maximum Warps | 64 |
| Maximum threads per Block | 1,024 |

Suppose a Kernel uses:

- 256 threads per Block
- 64 Registers per thread

Registers required by one Block:

$$
R_{block} = 256 \times 64
$$

$$
R_{block} = 16,384
$$

Maximum Blocks allowed by Registers:

$$
Blocks_{registers}
=
\frac{65,536}{16,384}
=4
$$

Maximum Blocks allowed by thread count:

$$
Blocks_{threads}
=
\frac{2,048}{256}
=8
$$

Therefore:

$$
Blocks = \min(4,8)=4
$$

Assuming no other resource limit is stricter.

Each Block contains:

$$
256/32=8\ Warps
$$

Four Blocks contain:

$$
4 \times 8=32\ Warps
$$

Occupancy:

$$
Occupancy = \frac{32}{64}=50\%
$$

## 5. Register Pressure and Occupancy

Registers are fast, on-chip storage used by threads.

They hold:

- Local variables
- Intermediate calculation results
- Frequently reused values
- Addresses and other temporary data

Consider:

```cpp
__global__ void myKernel(
    float* input,
    float* output
) {
    int tid = threadIdx.x;

    float a = input[tid];
    float b = input[tid + 256];
    float c = input[tid + 512];
    float d = input[tid + 768];

    float result = a + b + c + d;

    output[tid] = result;
}
```

The compiler may need multiple Registers to
hold these intermediate values.

More Registers per thread can reduce the number of
Blocks that fit simultaneously on an SM.

Illustrative example:

| Registers / Thread | Registers / Block | Blocks by Register Limit | Occupancy |
|---:|---:|---:|---:|
| 32 | 8,192 | 8 | 100% |
| 64 | 16,384 | 4 | 50% |
| 128 | 32,768 | 2 | 25% |

Assumptions:

- 256 threads per Block
- 65,536 Registers per SM
- 64 maximum Warps per SM
- No stricter architectural or resource limits
- Register allocation granularity is ignored

The key trade-off:

- More Registers can improve data reuse.
- More Registers can reduce Global Memory accesses.
- More Registers can also reduce Occupancy.

Excessive Register demand may cause spilling to
local memory, which can introduce additional memory traffic.

## 6. Is 100% Occupancy Always Better?

No.

Consider two Kernels:

- Kernel A: 100% Occupancy, but frequent Global Memory access.
- Kernel B: 50% Occupancy, but better data reuse and fewer
  Global Memory accesses.

Kernel B may be faster.

Performance depends on several factors:

- Memory Latency
- Memory Bandwidth
- Register Usage
- Shared Memory Usage
- Instruction-Level Parallelism
- Warp Scheduling
- Arithmetic Intensity

For GEMM and FlashAttention, using Registers and
Shared Memory for data reuse may reduce Occupancy.

However, the reduction in memory traffic can outweigh
the cost of lower Occupancy.

**The goal is not to maximize Occupancy.
The goal is to maximize actual Kernel performance.**

## 7. Important Concepts

| Concept | Meaning |
|---|---|
| SM | Main GPU execution and resource unit |
| Warp | Usually 32 threads |
| Active Warp | Resident Warp eligible for scheduling |
| Occupancy | Active Warps / Maximum Warps |
| Latency Hiding | Scheduling other Warps during waiting |
| Register Pressure | Resource pressure caused by Register usage |
| Resource Limitation | Limits on resident Blocks and Warps |

## 8. Key Takeaways

1. A GPU contains multiple SMs.
2. Each SM has limited Registers and Shared Memory.
3. Occupancy measures the proportion of resident Warps.
4. More active Warps can help hide memory latency.
5. Register and Shared Memory usage affect Occupancy.
6. Higher Occupancy does not necessarily mean higher performance.
7. Data reuse and reduced memory traffic can be more valuable
   than maximizing Occupancy.

## 9. Connection to the Next Lesson

Next: **Lesson 3.2 — Memory Coalescing**

We will study how neighboring Warp lanes access memory
and why consecutive memory addresses can significantly
improve GPU memory bandwidth.
