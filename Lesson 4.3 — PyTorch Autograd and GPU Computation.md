
# Lesson 4.3 — PyTorch Autograd and GPU Computation

## 1. What Is Autograd?

Autograd is PyTorch's automatic differentiation system.

It calculates gradients automatically during neural network
training.

For example:

\[
y=x^2+3x
\]

The derivative is:

\[
\frac{dy}{dx}=2x+3
\]

If \(x=2\):

\[
\frac{dy}{dx}=7
\]

PyTorch can calculate this automatically:

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

y = x**2 + 3*x

y.backward()

print(x.grad)
```

Output:

```text
tensor(7.)
```

## 2. requires_grad

When creating a tensor:

```python
x = torch.tensor(2.0, requires_grad=True)
```

`requires_grad=True` tells PyTorch to track operations
involving this tensor so that gradients can be calculated.

Without it, PyTorch normally does not track operations
for gradient computation.

## 3. The Computation Graph

Autograd records the operations used to calculate a result.

For:

\[
y=x^2+3x
\]

The conceptual graph is:

```text
          x
         / \
        /   \
      x²     3x
        \   /
         \ /
       Addition
          |
          y
```

Each operation has a corresponding gradient function.

During the backward pass, PyTorch uses these functions
to calculate gradients.

## 4. Forward Pass and Backward Pass

Neural network training has two major stages.

### Forward Pass

The model processes input data and calculates predictions.

```text
Input
  |
  v
Neural Network
  |
  v
Prediction
  |
  v
Loss
```

### Backward Pass

The model calculates gradients of the loss with respect
to its parameters.

```text
Loss
  |
  v
Calculate Gradients
  |
  v
Update Parameters
```

The optimizer uses the gradients to update model parameters.

## 5. Autograd on a GPU

Example:

```python
import torch

x = torch.randn(
    1000, 1000,
    device="cuda",
    requires_grad=True
)

y = x * x
loss = y.mean()

loss.backward()
```

Important points:

- The input tensor is stored on the GPU.
- Forward tensor operations use GPU implementations.
- Backward tensor operations also use GPU implementations
  when supported.
- Autograd manages the computation graph and gradient process.
- CUDA kernels perform the actual GPU computations.

**Autograd is not itself a CUDA kernel.**

It manages which operations need to be executed and how
gradients flow through the computation graph.

## 6. Understanding backward() and optimizer.step()

A common misunderstanding is that `backward()` updates
the model parameters.

It does not.

Consider a typical training loop:

```python
optimizer.zero_grad()

prediction = model(input_data)
loss = loss_function(prediction, target)

loss.backward()

optimizer.step()
```

Each operation has a different responsibility.

| Operation | Responsibility |
|---|---|
| `zero_grad()` | Clears previous gradients |
| `model()` | Performs forward computation |
| `loss_function()` | Calculates prediction error |
| `backward()` | Calculates gradients |
| `optimizer.step()` | Updates model parameters |

## 7. Why Do We Need zero_grad()?

PyTorch accumulates gradients by default.

Suppose the first iteration calculates:

\[
g_1=2
\]

The next iteration calculates:

\[
g_2=3
\]

Without clearing the previous gradient, PyTorch accumulates:

\[
g_{\text{total}}=g_1+g_2=5
\]

Usually, we want each training iteration to start with
fresh gradients.

Therefore:

```python
optimizer.zero_grad()
```

is normally called before the next backward pass.

## 8. Important Concepts

| Concept | Meaning |
|---|---|
| Autograd | Automatic differentiation system |
| `requires_grad` | Enables gradient tracking for a tensor |
| Computation graph | Records operations and dependencies |
| Forward pass | Calculates predictions and loss |
| Backward pass | Calculates gradients |
| `.grad` | Stores accumulated gradients |
| `backward()` | Computes gradients |
| `zero_grad()` | Clears existing gradients |
| `optimizer.step()` | Updates model parameters |

## 9. Key Takeaways

1. Autograd calculates gradients automatically.
2. `requires_grad=True` enables gradient tracking.
3. PyTorch records operations in a computation graph.
4. The forward pass calculates predictions and loss.
5. The backward pass calculates gradients.
6. CUDA tensors use GPU implementations for supported
   forward and backward operations.
7. Autograd manages gradient computation; CUDA kernels
   perform the actual GPU calculations.
8. `backward()` calculates gradients but does not update
   model parameters.
9. `optimizer.step()` performs parameter updates.
10. Gradients accumulate by default, so `zero_grad()` is
    normally needed between training iterations.

## 10. Mental Model

```text
Python / PyTorch
       |
       v
Forward Computation
       |
       v
Loss Calculation
       |
       v
Autograd
       |
       v
Backward Computation
       |
       v
GPU Gradient Operations
       |
       v
Optimizer Updates Parameters
```

**The key idea:**

Autograd determines how gradients are calculated.
CUDA executes the underlying tensor computations on the GPU.
The optimizer uses the gradients to update model parameters.
