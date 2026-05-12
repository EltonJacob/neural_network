# Day 3 — Neural Network from Scratch

**Phase 1 — Foundations** · 60-minute timed session · NumPy only (no PyTorch, no sklearn for the model)

---

## The Problem

Build a fully-connected neural network with **configurable depth and activations** using only NumPy. Implement the forward pass with caching, vectorized backpropagation through arbitrary depth, multiple activation functions (ReLU, Sigmoid, Tanh) with their exact gradients, mini-batch training, gradient checking, and basic regularization (L2 + dropout). By the end of the hour your network should learn **XOR** — if it doesn't, your backprop has a bug.

---

## Rules (from the 30-Day Plan)

1. Read all 4 stages **before** writing any code.
2. Design your architecture in **comments first** inside `neural_network.py`.
3. Write your own **edge case tests before** implementing — see `test_neural_network.py`.
4. **`print()` is your only debugger.** No notebooks during the timed session.
5. **3 stages clean beats 4 stages with NaN losses.** Build intuition, not just code.

---

## Time Budget — 60 Minutes (compressed from the doc's 120)

| Activity | Clock | Duration | Note |
|---|---|---|---|
| Read problem + design architecture | 0:00–0:05 | 5 min | NEVER skip this |
| Stage 1 — Forward Pass | 0:05–0:20 | 15 min | Cache everything |
| Stage 2 — Backpropagation | 0:20–0:35 | 15 min | Verify with grad check |
| Stage 3 — Activations + Optimization | 0:35–0:50 | 15 min | Tanh, mini-batch, grad check |
| Stage 4 — Regularization + Diagnostics | 0:50–1:00 | 10 min | Dropout, L2, layer stats |

If you reach Stage 3 cleanly by minute 50 you're on track. Stage 4 is bonus.

---

## Data Setup (build inside `neural_network.py`)

```python
# Start with XOR — if your net can't learn this, your backprop is broken.
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Then scale to moons:
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

# Stage 4 benchmark: sklearn digits dataset
from sklearn.datasets import load_digits
```

(`sklearn.datasets` is allowed for data generation only — the model itself stays NumPy-only.)

---

## Architecture Hints

Store layers as a list of dicts: `[{W, b, activation}, ...]`.
On the forward pass, **cache** `(A_prev, Z, W, b)` for every layer — you need them all in backprop.
Backprop core recurrence:

```
dZ      = dA * activation_backward(Z)
dW      = (1/m) * dZ @ A_prev.T
db      = (1/m) * np.sum(dZ, axis=1, keepdims=True)
dA_prev = W.T @ dZ
```

Design `__init__(layer_dims, activations)` from the start — e.g. `[2, 4, 4, 1]` with `["relu", "relu", "sigmoid"]`.

**He init** for ReLU layers: `W = np.random.randn(n, n_prev) * sqrt(2 / n_prev)`. Bias initialized to zeros.
Final layer: `sigmoid` for binary, `softmax` for multi-class.

---

## Stage 1 — Forward Pass (15 min)

**Class:** `NeuralNetworkScratch`

**Methods:**
- `__init__(layer_dims: List[int], activations: List[str])` — e.g. `[2, 4, 4, 1]`, `["relu", "relu", "sigmoid"]`
- `forward(X) -> np.ndarray` — returns final output, stores caches internally
- `relu(Z) -> np.ndarray`
- `sigmoid(Z) -> np.ndarray`

Initialize weights with He init for ReLU layers. Cache every `(A_prev, Z, W, b)` during the forward pass — you'll need them all in Stage 2.

---

## Stage 2 — Backpropagation (15 min)

**Methods:**
- `backward(y_true) -> dict` — returns `{dW1, db1, dW2, db2, ...}` per layer
- `relu_backward(dA, cache_Z) -> np.ndarray` — `dZ = dA * (Z > 0)`
- `sigmoid_backward(dA, cache_Z) -> np.ndarray` — `dZ = dA * sigmoid(Z) * (1 - sigmoid(Z))`
- `update_params(grads, lr) -> None`

Work backwards from the output layer. Compute `dZ`, then `dW`, `db`, and `dA_prev` for each layer.
ReLU gradient: pass `dA` through where `Z > 0`, zero elsewhere.
**Verify with numerical gradient checking** before continuing: `(L(w+eps) - L(w-eps)) / (2*eps)`.

---

## Stage 3 — Activations + Optimization (15 min)

**Methods:**
- `tanh(Z) -> np.ndarray`
- `tanh_backward(dA, cache_Z) -> np.ndarray` — `1 - tanh(Z)^2`
- `fit(X, y, lr, epochs, batch_size) -> List[float]`
- `gradient_check(X, y, epsilon=1e-5) -> float` — returns max relative error

Gradient check: if max relative error `< 1e-5`, your backprop is correct.
Tanh gradient: `1 - tanh(Z)^2` — you can reuse the cached `A` (which is `tanh(Z)`).
Mini-batch: shuffle the data each epoch start, process in `batch_size` chunks.

---

## Stage 4 — Regularization + Diagnostics (10 min, bonus)

**Methods:**
- `fit_with_dropout(X, y, keep_prob, lr, epochs) -> List[float]` — inverted dropout, train-only
- `fit_with_l2(X, y, alpha, lr, epochs) -> List[float]` — `dW += (alpha/m) * W`
- `plot_decision_boundary(X, y, resolution=0.01) -> None` — ASCII print is fine
- `layer_stats(layer_idx) -> dict` — `{mean_W, std_W, mean_grad, std_grad}`

Inverted dropout: scale activations by `1/keep_prob` during training, no scaling at inference.
Layer stats help diagnose vanishing/exploding gradients — large `std_grad` differences across layers are the smoking gun.

---

## Debrief Questions (fill in `notes.md` after)

- Did your network learn XOR? Final loss / accuracy?
- What was your max relative error from `gradient_check`? Below `1e-5`?
- ReLU vs Tanh on moons — which converged faster, and why?
- Did dropout help or hurt on the small dataset? Was that expected?
- Time log: minutes per stage.

---

## Reference — Days 1 & 2

The class skeleton, time-log workflow, and test-first style mirror **Day 1 — Linear Regression Engine** (`../Day01_Linear_Regression_Engine/`) and **Day 2 — Binary Classifier from Scratch** (`../Day02_Binary_Classifier/`). Re-use the patterns you've established:

- `generate_data` / `train_val_split` helpers at the top of the file
- A single class with explicit state (weights, biases, caches, loss history)
- Per-stage edge case tests in `test_neural_network.py` that you can comment out for stages you haven't reached
- `notes.md` for the time log + debrief

Day 1 gave you `dW = (2/n) * X.T @ (y_pred - y)`. Day 2 gave you `dW = (1/m) * X.T @ (p - y)` — same shape. Today the *last layer* gradient is the same formula again; the new thing is **propagating `dA_prev = W.T @ dZ` backwards** through arbitrary depth. The depth is the only new abstraction.

---

## Files in This Folder

- `neural_network.py` — your implementation (start here)
- `test_neural_network.py` — edge case tests, run as you implement each stage
- `notes.md` — time log and debrief answers
- `README.md` — this file
