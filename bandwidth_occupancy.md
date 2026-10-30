
# CUDA Course — Lesson 1.4
## Memory Bandwidth, Latency, and Occupancy

### 1. Latency vs. Bandwidth

Memory latency and memory bandwidth describe two different
properties.

- **Latency** is the time between requesting data and receiving it.
- **Bandwidth** is the amount of data transferred per unit of time.

A useful analogy is a truck transporting materials from a warehouse:

- Latency is how long the truck takes to arrive.
- Bandwidth is how much material the truck fleet can transport
  per second.

A system can have high bandwidth and still have high latency.

### 2. Why GPUs Use Many Threads

A warp may request data from global memory and then have to wait
for that data to become available.

Rather than leaving execution units idle, the GPU scheduler can
select another ready warp and execute it.

This technique is called **latency hiding**.

The GPU does not necessarily make an individual memory operation
finish sooner. Instead, it performs other work while the first
warp waits.

This is one reason GPUs support many concurrent threads and warps.

### 3. What Is Occupancy?

**Occupancy** is the ratio of active resident warps on an SM to
the maximum number of warps that the SM can support.

The formula is:

$$
Occupancy =
\frac{Active\ Resident\ Warps}
     {Maximum\ Resident\ Warps}
$$

For example, suppose an SM supports a maximum of 64 resident warps:

| Active Warps | Occupancy |
|---:|---:|
| 16 | 25% |
| 32 | 50% |
| 48 | 75% |
| 64 | 100% |

Occupancy describes resident warps, not how many warps are
executing instructions at exactly the same instant.

### 4. What Limits Occupancy?

Several hardware resources and architectural limits determine
how many warps can reside on an SM.

**1. Registers per thread**

- Each thread uses registers.
- An SM has a finite register file.
- High register usage can reduce the number of resident warps.

**2. Shared memory per block**

- Each block consumes part of the SM's shared memory.
- Larger shared-memory requirements can reduce the number
  of simultaneously resident blocks.

**3. Block and warp limits**

- The GPU architecture imposes maximum numbers of resident
  blocks and warps per SM.

This creates a trade-off:

- More registers per thread can help an individual thread
  execute efficiently.
- Fewer registers per thread can sometimes allow more warps
  to reside on an SM.

However, reducing register use too much can cause register
spilling into local memory, increasing memory traffic.

### 5. Is Higher Occupancy Always Faster?

No. **100% occupancy does not guarantee better performance
than 50% occupancy.**

Higher occupancy can help by giving the scheduler more ready
warps to run while other warps wait for memory or dependencies.

However, occupancy is only one factor in performance.

Other factors include:

- Memory latency.
- Memory bandwidth.
- Instruction-level parallelism.
- Register usage.
- Shared-memory usage.
- Memory access patterns.
- Cache behavior.
- Amount of useful work per thread.

A kernel may already have enough active warps to hide the
latency it encounters.

Increasing occupancy further may not improve its speed.

Conversely, very low occupancy can leave the SM without
enough independent work while some warps are stalled.

**Practical principle:** Aim for enough occupancy to keep the
GPU busy, but do not optimize occupancy in isolation.
Measure actual kernel performance.

### 6. Key Concepts to Remember

| Concept | Meaning |
|---|---|
| Memory latency | Time required for a memory request |
| Memory bandwidth | Amount of data transferred per second |
| Latency hiding | Running other ready warps while one warp waits |
| Occupancy | Active resident warps / maximum resident warps |
| Register pressure | Competition for the SM's finite registers |

The central idea of this lesson:

> A GPU improves utilization by switching to other ready warps
> when some warps are waiting. Occupancy provides opportunities
> for latency hiding, but maximum occupancy is not necessarily
> maximum performance.

### 7. Review Questions

**Q1.** A warp requests data from global memory and must wait.
How can the GPU maintain utilization?

**Answer:** The scheduler can select another ready warp.

**Q2.** An SM supports 64 resident warps. A kernel has
32 resident warps. What is its occupancy?

**Answer:** 

$$
Occupancy = \frac{32}{64} = 50\%
$$

**Q3.** A kernel uses too many registers per thread.
What may happen?

**Answer:** Fewer warps may reside on an SM because the register
file is a limited resource.

**Q4.** Is 100% occupancy always faster than 50% occupancy?

**Answer:** No. Higher occupancy can help hide latency, but the
best occupancy depends on the kernel's resource usage and
workload. Performance should be measured rather than inferred
from occupancy alone.

