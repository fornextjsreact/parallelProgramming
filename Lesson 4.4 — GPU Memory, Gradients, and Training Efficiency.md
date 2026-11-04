
# Lesson 4.4 — GPU Memory, Gradients, and Training Efficiency

## 1. Four Major GPU Memory Consumers

Training a neural network requires more than storing model
parameters.

There are four major memory consumers:

1. Model parameters (weights).
2. Gradients.
3. Optimizer states.
4. Activations.

All four can consume substantial GPU memory.

## 2. Model Parameters

Model parameters are the weights learned during training.

Example: A model with 100 million parameters.

For FP32:

\[
100M \times 4 = 400MB
\]

For FP16 or BF16:

\[
100M \times 2 = 200MB
\]

Lower-precision formats require less memory.

## 3. Gradients

During backpropagation, PyTorch calculates gradients
for model parameters.

For 100 million FP32 parameters:

\[
100M \times 4 = 400MB
\]

Gradients require additional memory during training.

Inference normally does not need parameter gradients.

## 4. Optimizer States

The Adam optimizer maintains two additional values
for each parameter:

- First moment: running average of gradients.
- Second moment: running average of squared gradients.

The equations are:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

For 100 million parameters, assuming both states use FP32:

\[
100M \times 4 \times 2 = 800MB
\]

### Simplified Memory Calculation

| Item | Memory |
|---|---:|
| FP32 parameters | 400 MB |
| FP32 gradients | 400 MB |
| Adam first moment | 400 MB |
| Adam second moment | 400 MB |
| **Total** | **1.6 GB** |

This excludes activations and other memory overhead.

## 5. Activations

Activations are intermediate tensors produced during
the forward pass.

Example:

```python
x = layer1(input)
x = torch.relu(x)

x = layer2(x)
x = torch.relu(x)

output = layer3(x)
```

Why are activations important?

During backpropagation, PyTorch often needs information
from the forward pass to calculate gradients.

For example, the ReLU derivative is:

\[
f'(x)=
\begin{cases}
1 & x>0\\
0 & x\leq0
\end{cases}
\]

The backward operation needs information about which
inputs were positive.

Therefore, some forward-pass results must be retained
for backward computation.

Deep networks and large batch sizes can require
substantial activation memory.

## 6. Activation Checkpointing

Activation checkpointing reduces memory consumption.

Instead of saving every intermediate activation,
PyTorch saves selected checkpoints.

During the backward pass, some forward operations
are recomputed when needed.

Example:

```python
from torch.utils.checkpoint import checkpoint

output = checkpoint(layer, input)
```

### Trade-off

- Less GPU memory consumption.
- More computation due to recomputation.
- Can allow training larger models or using larger batches.

The main idea is to trade additional computation
for reduced memory usage.

## 7. Training vs. Inference

| Memory Consumer | Training | Inference |
|---|---|---|
| Model parameters | Yes | Yes |
| Gradients | Yes | Usually no |
| Optimizer states | Yes | No |
| Activations | Often saved for backward | Needed temporarily |
| Backward computation | Yes | No |

Inference generally requires less memory because it does
not need gradient calculation or optimizer states.

### Disable Gradient Tracking

Using `torch.no_grad()`:

```python
with torch.no_grad():
    output = model(input)
```

Or:

```python
with torch.inference_mode():
    output = model(input)
```

Both are useful for inference.

`inference_mode()` is more restrictive but can provide
additional performance benefits.

## 8. Important Concepts

| Concept | Meaning |
|---|---|
| Model parameters | Learnable weights |
| Gradients | Derivatives used to update weights |
| Optimizer states | Additional data maintained by optimizers |
| Activations | Intermediate forward-pass results |
| Adam | Optimizer maintaining two moment estimates |
| Activation checkpointing | Saves memory by recomputing activations |
| `torch.no_grad()` | Disables gradient tracking |
| `torch.inference_mode()` | Optimized inference execution context |

## 9. Key Takeaways

1. Training memory is not just model parameter memory.
2. Gradients require additional memory.
3. Adam maintains two moment estimates for each parameter.
4. Activations can consume substantial memory.
5. Backpropagation often needs forward-pass information.
6. Activation checkpointing trades computation for memory.
7. Inference generally requires less memory than training.
8. `torch.no_grad()` and `torch.inference_mode()` avoid
   normal autograd tracking.

## 10. Mental Model

```text
GPU Memory During Training
          |
          +-- Model Parameters
          |
          +-- Gradients
          |
          +-- Optimizer States
          |
          +-- Activations
```

**The key idea:**

GPU memory is a major constraint in neural network training.

Understanding where memory is consumed helps developers
choose batch sizes, precision formats, optimizers, and
memory-saving techniques.
