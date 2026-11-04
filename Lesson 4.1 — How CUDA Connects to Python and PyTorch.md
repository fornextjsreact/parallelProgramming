
# Lesson 4.1 — How CUDA Connects to Python and PyTorch

## 1. The Big Picture

Python does not directly execute GPU instructions.

When a Python program uses PyTorch with CUDA tensors, the execution
typically follows this path:

```text
Python Code
    |
    v
PyTorch API
    |
    v
PyTorch Dispatcher
    |
    v
CUDA Implementation
    |
    v
GPU Execution
```

Each component has a different responsibility.

- **Python:** Describes the computation and calls PyTorch APIs.
- **PyTorch API:** Provides operations such as `add()`, `matmul()`,
  and `relu()`.
- **Dispatcher:** Selects the appropriate implementation based on
  properties such as device, data type, and layout.
- **CUDA implementation:** Executes GPU operations through CUDA
  kernels or optimized CUDA libraries.
- **GPU:** Executes the actual parallel computation.

## 2. Example: PyTorch GPU Computation

```python
import torch

a = torch.randn(1000, 1000, device="cuda")
b = torch.randn(1000, 1000, device="cuda")

c = torch.matmul(a, b)
```

What happens?

1. Python calls `torch.matmul()`.
2. PyTorch examines the input tensors.
3. The dispatcher selects an appropriate implementation.
4. PyTorch may use an optimized CUDA library such as cuBLAS.
5. The GPU performs the matrix multiplication.

The exact implementation depends on factors such as hardware,
data type, tensor layout, and PyTorch version.

**Important:** Python orchestrates GPU computation, but the GPU
executes the CUDA instructions.

## 3. Three Ways to Execute GPU Operations

### A. Built-in PyTorch Operations

Example:

```python
result = torch.relu(x)
```

- PyTorch provides the implementation.
- The operation can execute on the GPU if `x` is a CUDA tensor.
- The user does not need to write CUDA code.

### B. torch.compile

Example:

```python
model = torch.compile(model)
```

- Captures supported computation into a graph.
- Can optimize supported graph operations.
- May fuse multiple operations into fewer GPU kernels.
- Does not require manually writing CUDA kernels.
- Does not guarantee that every operation will be optimized.

### C. Custom CUDA Extension

A custom extension allows developers to implement their own
GPU operations and call them from Python.

Typical project structure:

```text
my_cuda_project/
|
├── setup.py
|
├── my_cuda_extension/
|   ├── __init__.py
|   └── ops.py
|
└── csrc/
    ├── add.cpp
    └── add_kernel.cu
```

Responsibilities:

| File | Responsibility |
|---|---|
| `add.cpp` | C++ interface and PyTorch binding |
| `add_kernel.cu` | CUDA kernel implementation |
| `setup.py` | Builds the extension |
| `ops.py` | Python interface |

The C++ binding and CUDA kernel have different responsibilities:

- C++ binding connects the custom operation to PyTorch.
- CUDA kernel defines the actual GPU computation.

A custom extension does not require rebuilding all of PyTorch.

## 4. Understanding a CUDA Kernel

Example:

```cpp
__global__ void add_kernel(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

The kernel performs element-wise addition.

Each GPU thread calculates its own global index:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

Then it processes one element, provided the index is within
the valid array range.

A kernel is launched using:

```cpp
add_kernel<<<blocks, threads_per_block>>>(a, b, c, n);
```

For example:

```cpp
add_kernel<<<4, 256>>>(a, b, c, n);
```

This launches:

- 4 blocks
- 256 threads per block
- 1,024 threads in total

## 5. Important Differences

| Mechanism | Main Purpose |
|---|---|
| PyTorch dispatcher | Selects an implementation |
| CUDA kernel | Defines GPU thread computation |
| `torch.compile` | Optimizes supported computation graphs |
| CUDA extension | Adds custom operations to PyTorch |

These mechanisms are not mutually exclusive.

For example, a PyTorch application can use built-in operations,
`torch.compile`, and custom CUDA extensions together.

## 6. Key Takeaways

1. Python does not execute GPU instructions.
2. PyTorch provides the interface between Python and GPU computation.
3. The dispatcher selects the appropriate implementation.
4. CUDA kernels define the work performed by GPU threads.
5. `torch.compile` can optimize supported PyTorch computations.
6. Custom CUDA extensions allow developers to add their own GPU
   implementations.
7. C++ bindings and CUDA kernels perform different jobs.
8. A custom extension does not require rebuilding PyTorch.

## 7. Mental Model

Think of the execution process as three layers:

```text
Application Layer
    Python / PyTorch
          |
          v
Framework Layer
    Dispatcher / Compiler
          |
          v
GPU Implementation Layer
    CUDA Kernels / CUDA Libraries
          |
          v
       GPU Hardware
```

**The key idea:**

PyTorch lets developers work at a high level, while CUDA provides
the ability to control how computation is performed on the GPU.
