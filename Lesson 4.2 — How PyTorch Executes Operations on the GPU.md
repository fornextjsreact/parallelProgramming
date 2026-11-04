
# Lesson 4.2 — How PyTorch Executes Operations on the GPU

## 1. Eager Execution

PyTorch normally uses eager execution.

Each Python operation is executed as it is encountered.

Example:

```python
import torch

a = torch.randn(1000, 1000, device="cuda")
b = torch.randn(1000, 1000, device="cuda")

c = a + b
d = torch.relu(c)
e = d * 2
```

Conceptually, PyTorch may launch three separate CUDA kernels:

1. Addition: `c = a + b`
2. ReLU: `d = relu(c)`
3. Multiplication: `e = d * 2`

The actual implementation depends on PyTorch and the hardware.

## 2. Costs of Multiple Kernel Launches

There are two important performance costs.

### A. Kernel Launch Overhead

- The CPU submits work to the GPU.
- Each kernel launch has some overhead.
- For small operations, launch overhead can be significant.
- For large computations, it may be relatively insignificant.

### B. Intermediate Memory Traffic

Consider:

```python
c = a + b
d = torch.relu(c)
```

Without fusion:

1. The addition kernel calculates `c`.
2. It writes `c` to GPU global memory.
3. The ReLU kernel reads `c`.
4. It calculates `d`.

The intermediate tensor requires additional memory traffic.

## 3. Kernel Fusion

Kernel fusion combines multiple operations into one kernel.

Original operations:

```python
c = a + b
d = torch.relu(c)
e = d * 2
```

Equivalent mathematical expression:

\[
e_i = 2\max(0,a_i+b_i)
\]

A fused kernel can calculate the final result in one pass.

### Benefits

- Fewer kernel launches.
- Less intermediate global-memory traffic.
- Better memory efficiency.
- Potentially lower execution time.

Intermediate values can remain in registers instead of being
written to global memory.

**Important:** Fusion changes the execution strategy, not the
mathematical result.

## 4. How torch.compile Helps

Example:

```python
model = torch.compile(model)
```

`torch.compile` can capture supported PyTorch operations into
a computation graph and optimize that graph.

Possible optimizations include:

- Operator fusion.
- Removing unnecessary intermediate operations.
- Optimizing memory access.
- Generating specialized GPU code.

It does not guarantee that every operation will be fused or
that every program will become faster.

Performance depends on:

- Graph structure.
- Compiler support.
- Hardware.
- Input shapes and data types.
- Compilation overhead.

## 5. When Is Fusion Useful?

| Situation | Potential Benefit |
|---|---|
| Many small operations | Reduce kernel launch overhead |
| Large intermediate tensors | Reduce memory traffic |
| Memory-bound computation | Reduce global-memory accesses |
| Large GEMM operation | Efficient matrix computation may matter more |

Fusion is particularly useful for combining smaller operations
in neural networks.

## 6. Important Concepts

| Concept | Meaning |
|---|---|
| Eager execution | Operations are executed as encountered |
| Kernel launch overhead | Cost of submitting a kernel to the GPU |
| Intermediate tensor | Temporary result between operations |
| Kernel fusion | Combining multiple operations into one kernel |
| Global memory | Large GPU memory accessible by GPU threads |
| Registers | Fast, thread-private GPU storage |
| `torch.compile` | PyTorch graph compilation and optimization |

## 7. Key Takeaways

1. Eager execution may launch a separate kernel for each operation.
2. Kernel launches have overhead.
3. Intermediate tensors can create additional global-memory traffic.
4. Kernel fusion combines multiple operations into one kernel.
5. Fusion can reduce both launch overhead and memory traffic.
6. Intermediate values may remain in registers.
7. `torch.compile` can perform fusion automatically for supported
   computation graphs.
8. Fusion is not guaranteed to improve every workload.

## 8. Mental Model

Without fusion:

```text
Operation A
    |
    v
Write intermediate tensor
    |
    v
Read intermediate tensor
    |
    v
Operation B
```

With fusion:

```text
Operation A + Operation B
          |
          v
     One GPU kernel
          |
          v
     Final result
```

**The key idea:**

GPU performance depends not only on how much computation is
performed, but also on how data moves and how efficiently
operations are scheduled.
