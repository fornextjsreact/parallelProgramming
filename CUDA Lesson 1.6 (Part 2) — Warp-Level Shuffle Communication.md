
# CUDA Lesson 1.6 — Warp-Level Shuffle Communication

## 1. What Is a Warp?

A warp normally contains 32 threads.

Each thread within a warp is called a lane.

- Lane 0: first thread.
- Lane 1: second thread.
- ...
- Lane 31: last thread.

Each lane has its own registers.

CUDA executes threads in a warp together, making
communication between lanes possible through warp-level
instructions.

## 2. What Is Shuffle Communication?

Shuffle communication allows one thread to obtain
a register value from another thread in the same warp.

The main CUDA instruction is:

```cpp
__shfl_down_sync(mask, value, delta);
```

Parameters:

- `mask`: identifies the participating lanes.
- `value`: the register value to exchange.
- `delta`: the distance between the source and destination
  lane indices.

For example:

```cpp
float other =
    __shfl_down_sync(0xffffffff, value, 1);
```

Lane 0 obtains the value from lane 1.

Lane 1 obtains the value from lane 2.

Lane 2 obtains the value from lane 3.

The operation returns another lane's register value.
It does not modify the original value.

The source lane must be valid and participate in the
shuffle operation.

## 3. Shuffle-Based Parallel Reduction

Shuffle communication can be used to calculate a sum
without explicitly moving intermediate values through
shared memory.

Consider eight lanes with these values:

```text
Lane:    0  1  2  3  4  5  6  7
Value:   2  4  6  8 10 12 14 16
```

### Stage 1: Shuffle distance = 1

Each lane adds the value from the next lane.

```text
Lane 0:  2 + 4  = 6
Lane 1:  4 + 6  = 10
Lane 2:  6 + 8  = 14
Lane 3:  8 + 10 = 18
...
```

### Stage 2: Shuffle distance = 2

Lane 0 combines its value with lane 2's value:

$$
6+14=20
$$

### Stage 3: Shuffle distance = 4

Lane 0 combines the partial sums:

$$
20+52=72
$$

The final sum is:

$$
Sum = 72
$$

For a full 32-lane warp, the reduction distances are:

$$
16,\ 8,\ 4,\ 2,\ 1
$$

Therefore, five shuffle-and-add stages are sufficient
to reduce 32 values to one sum in lane 0.

## 4. CUDA Implementation

A typical warp-level reduction looks like this:

```cpp
__device__ float warpReduceSum(float value) {
    unsigned mask = 0xffffffff;

    for (int offset = 16;
         offset > 0;
         offset /= 2) {

        float other =
            __shfl_down_sync(mask, value, offset);

        value += other;
    }

    return value;
}
```

Assumptions:

- All 32 lanes participate.
- All lanes contain valid input values.
- The full-warp mask is appropriate.

After the reduction, lane 0 contains the complete sum.
Other lanes may contain partial sums.

## 5. Why Is Shuffle More Efficient Than Shared Memory?

### Shared-memory communication

A typical shared-memory approach involves:

1. Writing values into shared memory.
2. Synchronizing threads.
3. Reading values from shared memory.
4. Performing additions.

### Shuffle communication

A warp-shuffle approach involves:

1. Obtaining another lane's register value.
2. Adding the value to the current register.

Shuffle communication can avoid:

- Explicit shared-memory writes.
- Explicit shared-memory reads.
- Block-wide barriers for each warp-level stage.

This generally reduces communication overhead.

However, actual performance depends on the GPU architecture
and the surrounding workload.

## 6. An Important Limitation: Warp Boundaries

Shuffle instructions only communicate between lanes
within the same warp.

They cannot directly exchange values between different
warps.

For example, consider a block containing 256 threads:

$$
256/32=8\ warps
$$

Each warp can independently calculate its own sum
using shuffle operations.

The block now has eight partial sums.

To combine them, we need another communication mechanism.

## 7. Hierarchical Reduction

A common design combines shuffle and shared memory.

### Step 1: Warp-level reduction

Each warp independently calculates a partial sum.

### Step 2: Store partial sums

One lane from each warp writes its result into shared memory.

### Step 3: Synchronize

Use:

```cpp
__syncthreads();
```

This ensures all warp sums have been written.

### Step 4: Final warp-level reduction

One warp loads the partial sums and uses shuffle
operations to calculate the final block sum.

This approach uses:

- Shuffle for communication within a warp.
- Shared memory for communication between warps.

## 8. Communication Mechanisms Compared

| Communication | Scope | Mechanism |
|---|---|---|
| Thread to itself | One thread | Registers |
| Lane to lane | One warp | Shuffle |
| Warp to warp | One block | Shared memory + synchronization |
| Block to block | Grid | Global memory + another kernel or suitable grid-wide synchronization |

## 9. Key Takeaways

1. A warp normally contains 32 lanes.
2. Each lane has its own registers.
3. Shuffle instructions allow lanes to exchange register
   values directly.
4. `__shfl_down_sync()` obtains a value from a higher-index
   lane.
5. Shuffle operations can implement efficient warp-level
   reduction.
6. A 32-lane warp requires five reduction stages.
7. Shuffle communication is restricted to one warp.
8. Shared memory is needed to combine results from
   different warps within a block.
9. Hierarchical reduction combines warp shuffle and
   shared memory.

## 10. What Comes Next?

We will study how to optimize a complete CUDA reduction
kernel, including:

- Multiple elements per thread.
- Coalesced global-memory access.
- Warp-level reduction.
- Shared-memory communication between warps.
- Reducing synchronization overhead.
- Measuring GPU performance.
