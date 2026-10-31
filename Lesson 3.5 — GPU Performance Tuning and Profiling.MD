
# Lesson 3.5 — GPU Performance Tuning and Profiling

## 1. Why Do We Need Profiling?

Profiling helps identify where a CUDA application spends
its execution time and which resources limit performance.

A kernel that looks more sophisticated is not necessarily
faster. Optimization should be based on measurements.

Three important performance measurements:

| Measurement | Meaning |
|---|---|
| Kernel execution time | Time spent executing a GPU kernel |
| Data-transfer time | Time spent transferring data between CPU and GPU |
| End-to-end time | Total application execution time |

Example:

- CPU-to-GPU transfer: 4 ms
- Kernel execution: 1 ms
- GPU-to-CPU transfer: 4 ms

Total:

$$
T_{\text{total}}=4+1+4=9\text{ ms}
$$

If kernel execution is reduced from 1 ms to 0.5 ms:

$$
T_{\text{total}}=4+0.5+4=8.5\text{ ms}
$$

The kernel becomes twice as fast, but the entire
application improves by only about 5.6%.

This demonstrates why end-to-end performance matters.

## 2. Amdahl's Law

Amdahl's Law estimates the maximum overall speedup
when only part of a program is optimized.

$$
Speedup=
\frac{1}{(1-P)+\frac{P}{S}}
$$

Where:

- P = fraction of execution time being optimized.
- S = speedup of that fraction.

Example:

Suppose GPU computation accounts for 20% of total
execution time, and that part becomes 10 times faster.

$$
Speedup=
\frac{1}{0.8+\frac{0.2}{10}}
$$

$$
Speedup\approx1.22
$$

The whole application becomes only about 1.22 times
faster.

The lesson is that optimizing a small part of a program
cannot produce unlimited overall improvement.

## 3. Common GPU Performance Bottlenecks

### A. Memory-Bound Kernels

A memory-bound kernel is limited primarily by memory
bandwidth.

Example: vector addition.

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

Each element requires approximately:

- One read from A.
- One read from B.
- One write to C.
- One floating-point addition.

For 32-bit floats, this represents approximately
12 bytes transferred per addition, ignoring cache effects.

The arithmetic workload is small compared with the
memory traffic.

Potential optimizations:

- Improve memory coalescing.
- Avoid unnecessary data transfers.
- Reuse data where possible.
- Reduce redundant global-memory accesses.

### B. Compute-Bound Kernels

A compute-bound kernel is limited primarily by
arithmetic throughput.

Matrix multiplication is a common example:

$$
C=A\times B
$$

Each loaded matrix element can be reused many times,
creating high arithmetic intensity.

Potential optimizations:

- Use tiling.
- Improve data reuse.
- Use Tensor Cores where appropriate.
- Increase useful parallelism.
- Reduce unnecessary instructions.

### C. Latency-Bound Kernels

A latency-bound kernel spends substantial time waiting
for operations to complete.

For example, a thread may wait for global-memory data.

The GPU hides latency by scheduling other ready warps.

Potential optimizations:

- Provide more eligible warps.
- Improve occupancy when beneficial.
- Increase independent work within threads.
- Reduce dependency chains.

Higher occupancy is not always faster. It depends on
the actual bottleneck.

### D. Launch-Bound Kernels

A launch-bound kernel performs very little work relative
to kernel-launch overhead.

Potential optimizations:

- Combine small kernels.
- Reduce unnecessary launches.
- Use CUDA Graphs for repeated execution patterns.
- Process more data per launch when appropriate.

## 4. Arithmetic Intensity and the Roofline Model

Arithmetic intensity measures the amount of computation
performed for each byte transferred from memory.

$$
AI=
\frac{\text{Floating-point operations}}
{\text{Bytes transferred}}
$$

Its unit is FLOP/byte.

Example:

A kernel performs 1,000 floating-point operations
and transfers 100 bytes.

$$
AI=\frac{1000}{100}=10\text{ FLOP/byte}
$$

The Roofline Model combines two hardware limits:

1. Peak memory bandwidth.
2. Peak arithmetic throughput.

The theoretical performance ceiling is:

$$
Performance\leq
\min(P_{\text{peak}},BW_{\text{peak}}\times AI)
$$

Where:

- P_peak = peak computing throughput.
- BW_peak = peak memory bandwidth.
- AI = arithmetic intensity.

Interpretation:

- Low arithmetic intensity often indicates a
  memory-bandwidth-bound kernel.
- High arithmetic intensity may indicate a
  compute-bound kernel.

The Roofline Model helps determine which optimization
techniques are worth investigating.

## 5. NVIDIA Profiling Tools

Two important NVIDIA profiling tools are:

### Nsight Systems

An application-level profiler.

It shows:

- CPU activity.
- GPU kernel execution.
- Memory transfers.
- CUDA streams.
- Synchronization.
- GPU idle periods.

It answers:

**Where is the application spending its time?**

Useful for identifying:

- Excessive data transfers.
- Unnecessary synchronization.
- Poor stream utilization.
- Kernel-launch overhead.
- CPU delays that leave the GPU idle.

### Nsight Compute

A kernel-level profiler.

It examines individual GPU kernels.

It provides information about:

- Memory throughput.
- Compute throughput.
- Register usage.
- Shared-memory usage.
- Occupancy.
- Warp stalls.
- Instruction execution.

It answers:

**Why is this particular kernel taking so long?**

Typical workflow:

1. Use Nsight Systems to find the slow part
   of the application.
2. Use Nsight Compute to investigate the
   performance of the slow kernel.

## 6. Measuring Kernel Execution Time

CUDA kernel launches are generally asynchronous
from the CPU's perspective.

An ordinary CPU timer may stop before the GPU
finishes executing the kernel.

Use CUDA events instead:

```cpp
cudaEvent_t start, stop;

cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);

myKernel<<<blocks, threads>>>(d_A, d_B, d_C);

cudaEventRecord(stop);
cudaEventSynchronize(stop);

float milliseconds = 0.0f;

cudaEventElapsedTime(
    &milliseconds,
    start,
    stop
);

printf("Kernel time: %f ms\n", milliseconds);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

The sequence:

1. Record the start event.
2. Launch the kernel.
3. Record the stop event.
4. Wait for the stop event.
5. Calculate elapsed GPU time.

Both events and the kernel should be submitted
to the same stream when measuring that stream's work.

### Benchmarking Practices

- Warm up the GPU before measuring.
- Run multiple iterations.
- Measure enough work to reduce timing noise.
- Avoid including memory allocation unless relevant.
- Verify output correctness.
- Use consistent input sizes and conditions.
- Distinguish kernel time from end-to-end time.

For very short kernels, CUDA Graphs or batched repetitions
can help reduce measurement noise and launch overhead.

## 7. Systematic Optimization Workflow

A practical optimization process:

1. **Establish correctness**
   - Compare results with a trusted implementation.
   - Test boundary conditions and different input sizes.

2. **Measure baseline performance**
   - Record kernel execution time.
   - Record data-transfer time.
   - Record end-to-end execution time.

3. **Profile the application**
   - Use Nsight Systems.
   - Identify transfers, idle periods, synchronization,
     and launch overhead.

4. **Identify the kernel bottleneck**
   - Use Nsight Compute.
   - Investigate memory, computation, occupancy,
     and warp stalls.

5. **Form a hypothesis**
   - Example: repeated global-memory loads are
     limiting performance.

6. **Apply one optimization**
   - Try coalescing, tiling, shared-memory reuse,
     or fewer kernel launches.

7. **Measure again**
   - Compare against the original baseline under
     the same conditions.

8. **Keep or reject the change**
   - Keep it only if it improves the relevant
     performance metric without breaking correctness.

Change one major factor at a time so that the effect
of each optimization can be measured.

## 8. Common Optimization Mistakes

| Mistake | Why It Is a Problem |
|---|---|
| Assuming 100% occupancy is always best | Lower occupancy can still produce higher performance |
| Optimizing without profiling | The optimized component may not be the bottleneck |
| Using shared memory everywhere | Copying and synchronization can cost more than they save |
| Increasing threads per block blindly | Resource limits may reduce resident blocks |
| Excessive loop unrolling | Can increase register use and instruction-cache pressure |
| Ignoring CPU-GPU transfers | Transfers can dominate application execution |
| Trusting one timing result | GPU execution has measurement variability |

## 9. Connecting the Entire CUDA Course

| Topic | Main Purpose |
|---|---|
| Grid, block, thread, warp | Distribute parallel work |
| Memory hierarchy | Understand where data is stored |
| Occupancy | Hide execution latency |
| Synchronization | Coordinate threads safely |
| Parallel reduction | Combine parallel results |
| Memory allocation and transfer | Exchange data between CPU and GPU |
| Streams and events | Manage asynchronous execution |
| Error handling | Detect and debug failures |
| Memory coalescing | Access global memory efficiently |
| Shared memory and tiling | Reuse data |
| ILP and loop unrolling | Expose independent instructions |
| Warp specialization | Organize execution pipelines |
| Asynchronous operations | Overlap data movement and computation |
| Tensor Cores | Accelerate matrix operations |
| Profiling | Identify and measure bottlenecks |

## 10. Key Takeaways

1. Measure end-to-end performance, not just kernel time.
2. Amdahl's Law explains why optimizing one component
   may have limited impact on the whole application.
3. Identify whether a kernel is memory-bound,
   compute-bound, latency-bound, or launch-bound.
4. Arithmetic intensity helps distinguish memory
   and compute limitations.
5. The Roofline Model estimates theoretical
   performance limits.
6. Nsight Systems analyzes application execution flow.
7. Nsight Compute analyzes individual kernel performance.
8. CUDA events provide reliable GPU-side timing.
9. Optimize based on profiling evidence, not intuition.
10. Measure every optimization against a consistent
    baseline.

## 11. Final Perspective

High-performance CUDA programming requires coordinating
three major resources:

- Computing hardware.
- Memory hierarchy.
- Execution pipeline.

The objective is to keep computing units busy while
minimizing unnecessary data movement and waiting.

A good GPU programmer does not simply write complicated
kernels.

A good GPU programmer:

1. Understands the algorithm.
2. Understands the hardware.
3. Measures performance.
4. Identifies the bottleneck.
5. Applies an appropriate optimization.
6. Measures again.
