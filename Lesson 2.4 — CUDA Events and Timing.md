
# Lesson 2.4 — CUDA Events and Timing

## 1. Why Is GPU Timing Different from CPU Timing?

CUDA kernel launches are normally asynchronous from the
CPU's perspective.

Consider:

```cpp
auto start = std::chrono::high_resolution_clock::now();

myKernel<<<blocks, threads>>>(d_A, d_B, d_C);

auto end = std::chrono::high_resolution_clock::now();
```

The CPU submits the kernel and continues executing without
necessarily waiting for the GPU to finish.

Therefore, the measured time may represent mostly kernel
launch overhead rather than actual GPU execution time.

### Using cudaDeviceSynchronize()

One solution is:

```cpp
auto start = std::chrono::high_resolution_clock::now();

myKernel<<<blocks, threads>>>(d_A, d_B, d_C);

cudaDeviceSynchronize();

auto end = std::chrono::high_resolution_clock::now();
```

`cudaDeviceSynchronize()` waits for previously submitted
device work to complete.

This makes the CPU timer include GPU execution time.

However, the measurement may also include CPU launch
and synchronization overhead.

It can also interfere with asynchronous execution.

For more precise GPU-side timing, CUDA provides events.

## 2. What Is a CUDA Event?

A CUDA event is a marker inserted into a CUDA stream.

When the GPU reaches the marker, the event is recorded.

We can place two events around a kernel:

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
```

Execution sequence:

1. Create two events.
2. Record the start event.
3. Launch the kernel.
4. Record the stop event.
5. Wait for the stop event to complete.
6. Calculate elapsed time.

The GPU processes operations in stream order:

```text
Start event → Kernel execution → Stop event
```

The elapsed time represents the interval between the
two event timestamps on the GPU timeline.

## 3. Important CUDA Event APIs

| API | Purpose |
|---|---|
| `cudaEventCreate()` | Create an event |
| `cudaEventRecord()` | Insert an event into a stream |
| `cudaEventSynchronize()` | Wait until an event completes |
| `cudaEventElapsedTime()` | Calculate elapsed time in milliseconds |
| `cudaEventDestroy()` | Release an event |

### Recording Events in a Specific Stream

```cpp
cudaEventRecord(start, stream1);
```

The second argument specifies the stream.

For correct timing, record both events in the stream
containing the kernel you want to measure.

## 4. Complete Example: Timing a Kernel

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);

myKernel<<<blocks, threads, 0, stream>>>(
    d_A, d_B, d_C
);

cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);

float milliseconds;

cudaEventElapsedTime(
    &milliseconds,
    start,
    stop
);

printf("GPU time: %f ms\n", milliseconds);

cudaEventDestroy(start);
cudaEventDestroy(stop);
cudaStreamDestroy(stream);
```

Because both events and the kernel are submitted to the
same stream, their execution order is preserved.

The measured interval brackets the kernel execution.

## 5. Why Measure Multiple Iterations?

Short kernels may execute in only a few microseconds.

At this scale, measurement variability, clock frequency
changes, cache state, and launch overhead can affect results.

A common approach is to execute the kernel multiple times:

```cpp
cudaEventRecord(start);

for (int i = 0; i < 100; i++) {
    myKernel<<<blocks, threads>>>(d_A, d_B, d_C);
}

cudaEventRecord(stop);

cudaEventSynchronize(stop);

float milliseconds;

cudaEventElapsedTime(
    &milliseconds,
    start,
    stop
);

printf("Average time: %f ms\n", milliseconds / 100.0f);
```

This measures average time per launch over the batch.

Good benchmarking practices:

- Warm up the GPU before measurement.
- Run multiple measurement batches.
- Exclude memory allocation unless it is part of
  the operation being benchmarked.
- Use consistent input sizes and execution conditions.
- Verify that the kernel produces correct results.

## 6. Different Types of Performance Measurements

| Measurement | What it includes |
|---|---|
| Kernel execution time | GPU work between event markers |
| Memory transfer time | Host-device data transfer |
| End-to-end time | Transfers, launches, computation, and other required work |
| Average kernel time | Average across repeated kernel executions |

Example:

Suppose an application takes:

- 4 ms to copy input to the GPU.
- 1 ms to execute the kernel.
- 4 ms to copy output to the CPU.

If these operations execute sequentially:

Total time = 4 + 1 + 4 = 9 ms.

Even if we optimize the kernel from 1 ms to 0.5 ms,
the total time only decreases from 9 ms to 8.5 ms.

This demonstrates why measuring the entire application
is just as important as measuring individual kernels.

## 7. Key Takeaways

1. CUDA kernel launches are normally asynchronous.
2. A CPU timer may stop before the GPU finishes executing.
3. `cudaDeviceSynchronize()` can make CPU timing include
   GPU execution, but adds synchronization overhead.
4. CUDA events provide GPU-side timing.
5. Events should normally be recorded in the same stream
   as the operations being measured.
6. `cudaEventSynchronize(stop)` waits for the stop event.
7. `cudaEventElapsedTime()` returns elapsed time in milliseconds.
8. Repeated measurements and warm-up improve benchmarking
   reliability.
9. Kernel execution time and end-to-end application time
   are different measurements.

The main lesson is:

**Measure the actual performance bottleneck, not just the
execution time of one kernel.**
