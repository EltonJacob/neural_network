"""
Edge case tests for Day 3 — Neural Network from Scratch.

Per the 30-Day Mantra: tests are written BEFORE you implement.
Run with: python test_neural_network.py

Tests are organized by stage. Each test prints PASS/FAIL — print() is your debugger.
Comment out tests for stages you haven't reached yet.

DATA CONVENTION (per README — Andrew Ng / column-major style):
    X.shape = (n_features, m)         # features as rows, samples as columns
    y.shape = (1, m)                  # binary
    W[l].shape = (n[l], n[l-1])
    b[l].shape = (n[l], 1)
    forward(X) -> shape (last_layer_dim, m)

If you choose row-major instead, you'll need to flip the assertions accordingly.
"""

import numpy as np
from neural_network import (
    NeuralNetworkScratch,
    generate_xor,
    generate_moons,
    generate_digits,
    train_val_split,
)


def _check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def _make_tiny_net(seed: int = 0):
    """A small reproducible net used across several tests."""
    return NeuralNetworkScratch(
        layer_dims=[3, 4, 2, 1],
        activations=["relu", "relu", "sigmoid"],
        seed=seed,
    )


def _make_tiny_batch(n_features: int = 3, m: int = 8, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_features, m))
    y = (rng.random((1, m)) > 0.5).astype(float)
    return X, y


# -----------------------------------------------------------------------------
# STAGE 1 TESTS — Forward Pass
# -----------------------------------------------------------------------------

def test_stage_1():
    print("\n--- Stage 1: Forward Pass ---")

    # __init__ should reject mismatched activation count.
    raised = False
    try:
        NeuralNetworkScratch(layer_dims=[2, 4, 1], activations=["relu"])  # missing one
    except Exception:
        raised = True
    _check("__init__ rejects len(activations) != len(layer_dims) - 1", raised)

    # ReLU primitives.
    Z = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    relu_out = NeuralNetworkScratch.relu(Z)
    _check("relu zeros negatives",
           np.all(relu_out[Z < 0] == 0), f"got {relu_out}")
    _check("relu passes positives through",
           np.allclose(relu_out[Z > 0], Z[Z > 0]), f"got {relu_out}")
    _check("relu(0) == 0", relu_out[0, 2] == 0.0)

    # Sigmoid primitives.
    extreme = np.array([[-1e6, -1e3, 0.0, 1e3, 1e6]])
    sig = NeuralNetworkScratch.sigmoid(extreme)
    _check("sigmoid handles extreme inputs without NaN/Inf",
           np.all(np.isfinite(sig)), f"got {sig}")
    _check("sigmoid(0) == 0.5", np.isclose(sig[0, 2], 0.5))
    _check("sigmoid bounded in (0, 1)",
           np.all((sig >= 0.0) & (sig <= 1.0)))
    _check("sigmoid is monotonically increasing",
           np.all(np.diff(sig.flatten()) >= 0))

    # He init: stdev of a ReLU-layer's W roughly equals sqrt(2 / n_prev).
    net = NeuralNetworkScratch(
        layer_dims=[256, 256], activations=["relu"], seed=0
    )
    W1 = net.params["W1"]
    expected_std = np.sqrt(2.0 / 256)
    observed_std = float(W1.std())
    _check("He init: W std ≈ sqrt(2 / n_prev) within 25%",
           abs(observed_std - expected_std) / expected_std < 0.25,
           f"expected≈{expected_std:.4f}, got {observed_std:.4f}")
    _check("He init: b is zeros",
           np.allclose(net.params["b1"], 0.0))

    # Forward pass — output shape and cache count.
    net = _make_tiny_net()
    X, _ = _make_tiny_batch(n_features=3, m=8)
    out = net.forward(X)
    _check("forward output shape == (last_layer_dim, m)",
           out.shape == (1, 8), f"got {out.shape}")
    _check("forward output values in (0, 1) for sigmoid head",
           np.all((out > 0) & (out < 1)))
    _check("forward populates one cache per layer",
           len(net.caches) == 3, f"got {len(net.caches)}")


# -----------------------------------------------------------------------------
# STAGE 2 TESTS — Backpropagation
# -----------------------------------------------------------------------------

def test_stage_2():
    print("\n--- Stage 2: Backpropagation ---")

    # ReLU backward — zero where Z <= 0, pass-through where Z > 0.
    Z = np.array([[-1.0, 0.0, 2.0, -0.5]])
    dA = np.array([[0.7, 0.7, 0.7, 0.7]])
    dZ = NeuralNetworkScratch.relu_backward(dA, Z)
    _check("relu_backward zeros dA where Z <= 0",
           np.all(dZ[Z <= 0] == 0), f"got {dZ}")
    _check("relu_backward passes dA where Z > 0",
           np.allclose(dZ[Z > 0], dA[Z > 0]), f"got {dZ}")

    # Sigmoid backward — dZ == dA * s * (1 - s).
    Z = np.array([[0.0, 1.0, -1.0, 2.0]])
    dA = np.ones_like(Z)
    s = 1.0 / (1.0 + np.exp(-Z))
    expected_dZ = dA * s * (1 - s)
    got_dZ = NeuralNetworkScratch.sigmoid_backward(dA, Z)
    _check("sigmoid_backward == dA * s * (1-s)",
           np.allclose(got_dZ, expected_dZ, atol=1e-7),
           f"expected {expected_dZ}, got {got_dZ}")

    # backward() returns gradients with matching shapes.
    net = _make_tiny_net()
    X, y = _make_tiny_batch(n_features=3, m=8)
    net.forward(X)
    grads = net.backward(y)
    shape_ok = True
    detail = ""
    for key in ("W1", "b1", "W2", "b2", "W3", "b3"):
        if net.params[key].shape != grads[f"d{key}"].shape:
            shape_ok = False
            detail = (f"{key}: param {net.params[key].shape} "
                      f"vs grad {grads[f'd{key}'].shape}")
            break
    _check("backward grad shapes match param shapes", shape_ok, detail)

    # One step of update_params should reduce loss (on a slightly larger batch).
    def bce(p, y):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 64))
    y = (rng.random((1, 64)) > 0.5).astype(float)
    net = _make_tiny_net(seed=1)
    p0 = net.forward(X)
    loss0 = bce(p0, y)
    grads = net.backward(y)
    net.update_params(grads, lr=0.5)
    p1 = net.forward(X)
    loss1 = bce(p1, y)
    _check("one update_params step reduces BCE loss",
           loss1 < loss0, f"{loss0:.4f} -> {loss1:.4f}")

    # Numerical gradient check on a small net + small batch.
    small_net = NeuralNetworkScratch(
        layer_dims=[2, 3, 1], activations=["relu", "sigmoid"], seed=0
    )
    rng = np.random.default_rng(7)
    Xg = rng.standard_normal((2, 5))
    yg = (rng.random((1, 5)) > 0.5).astype(float)
    err = small_net.gradient_check(Xg, yg, epsilon=1e-7)
    _check("gradient_check max relative error < 1e-5",
           err < 1e-5, f"got {err:.2e}")


# -----------------------------------------------------------------------------
# STAGE 3 TESTS — Activations + Optimization
# -----------------------------------------------------------------------------

def test_stage_3():
    print("\n--- Stage 3: Activations + Optimization ---")

    # Tanh primitives.
    Z = np.array([[-2.0, 0.0, 2.0]])
    t = NeuralNetworkScratch.tanh(Z)
    _check("tanh(0) == 0", np.isclose(t[0, 1], 0.0))
    _check("tanh bounded in (-1, 1)",
           np.all((t > -1) & (t < 1)), f"got {t}")
    _check("tanh is odd: tanh(-x) == -tanh(x)",
           np.isclose(t[0, 0], -t[0, 2], atol=1e-7))

    # Tanh backward: dZ == dA * (1 - tanh(Z)^2).
    dA = np.ones_like(Z)
    expected = dA * (1 - np.tanh(Z) ** 2)
    got = NeuralNetworkScratch.tanh_backward(dA, Z)
    _check("tanh_backward == dA * (1 - tanh(Z)^2)",
           np.allclose(got, expected, atol=1e-7),
           f"expected {expected}, got {got}")

    # fit() on XOR with [2, 4, 1] (relu, sigmoid) should reach low loss.
    X_xor = np.array([[0, 0, 1, 1],
                      [0, 1, 0, 1]], dtype=float)        # (2, 4)
    y_xor = np.array([[0, 1, 1, 0]], dtype=float)        # (1, 4)
    xor_net = NeuralNetworkScratch(
        layer_dims=[2, 4, 1], activations=["relu", "sigmoid"], seed=0
    )
    losses = xor_net.fit(X_xor, y_xor, lr=0.5, epochs=5000, batch_size=4)
    _check("fit returns a non-empty loss history",
           isinstance(losses, list) and len(losses) > 0,
           f"got len {len(losses) if isinstance(losses, list) else 'N/A'}")
    _check("XOR final loss < 0.1",
           losses[-1] < 0.1, f"final loss = {losses[-1]:.4f}")
    preds = (xor_net.forward(X_xor) > 0.5).astype(int)
    _check("XOR predictions match labels (all 4)",
           np.array_equal(preds, y_xor.astype(int)),
           f"preds={preds.flatten().tolist()}, y={y_xor.flatten().tolist()}")

    # Mini-batch sanity — batch_size=1 should still drive loss down.
    sgd_net = NeuralNetworkScratch(
        layer_dims=[2, 4, 1], activations=["relu", "sigmoid"], seed=1
    )
    sgd_losses = sgd_net.fit(X_xor, y_xor, lr=0.3, epochs=2000, batch_size=1)
    _check("SGD (batch_size=1) still reduces loss meaningfully",
           sgd_losses[-1] < sgd_losses[0] * 0.5,
           f"{sgd_losses[0]:.4f} -> {sgd_losses[-1]:.4f}")

    # Gradient check still passes after training.
    err = xor_net.gradient_check(X_xor, y_xor, epsilon=1e-7)
    _check("gradient_check on trained XOR net stays < 1e-5",
           err < 1e-5, f"got {err:.2e}")


# -----------------------------------------------------------------------------
# STAGE 4 TESTS — Regularization + Diagnostics (bonus)
# -----------------------------------------------------------------------------

def test_stage_4():
    print("\n--- Stage 4: Regularization + Diagnostics ---")

    # Build moons-style data (use generate_moons if implemented; else fall back).
    try:
        Xm, ym = generate_moons(n_samples=400, noise=0.1)
        # Convert to column-major (2, m) and (1, m).
        if Xm.shape[0] != 2:
            Xm = Xm.T
        if ym.ndim == 1:
            ym = ym.reshape(1, -1).astype(float)
    except Exception as e:
        print(f"  [SKIP] could not build moons data: {e}")
        return

    # L2 fit should produce smaller weight norm than plain fit.
    plain = NeuralNetworkScratch(
        layer_dims=[2, 8, 1], activations=["relu", "sigmoid"], seed=0
    )
    plain.fit(Xm, ym, lr=0.1, epochs=500, batch_size=32)
    plain_norm = sum(np.linalg.norm(plain.params[f"W{i}"])
                     for i in range(1, len(plain.layer_dims)))

    l2 = NeuralNetworkScratch(
        layer_dims=[2, 8, 1], activations=["relu", "sigmoid"], seed=0
    )
    l2.fit_with_l2(Xm, ym, alpha=0.5, lr=0.1, epochs=500)
    l2_norm = sum(np.linalg.norm(l2.params[f"W{i}"])
                  for i in range(1, len(l2.layer_dims)))
    _check("L2-fit total W norm < plain fit total W norm",
           l2_norm < plain_norm,
           f"plain={plain_norm:.3f}, l2={l2_norm:.3f}")

    # Dropout fit should still converge to a workable accuracy on moons.
    drop = NeuralNetworkScratch(
        layer_dims=[2, 8, 1], activations=["relu", "sigmoid"], seed=0
    )
    drop.fit_with_dropout(Xm, ym, keep_prob=0.8, lr=0.1, epochs=500)
    preds = (drop.forward(Xm) > 0.5).astype(int)
    acc = float(np.mean(preds == ym.astype(int)))
    _check("dropout-trained net reaches accuracy > 0.85 on moons",
           acc > 0.85, f"acc={acc:.3f}")

    # layer_stats returns the required keys.
    stats = plain.layer_stats(1)
    required = {"mean_W", "std_W", "mean_grad", "std_grad"}
    _check("layer_stats returns mean_W/std_W/mean_grad/std_grad",
           required.issubset(set(stats.keys())),
           f"got keys {list(stats.keys())}")

    # plot_decision_boundary just needs to run without raising.
    ran = True
    try:
        plain.plot_decision_boundary(Xm, ym, resolution=0.1)
    except Exception as e:
        ran = False
        print(f"    plot_decision_boundary raised: {e}")
    _check("plot_decision_boundary runs without raising", ran)


# -----------------------------------------------------------------------------
# RUN ALL
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Day 3 — Neural Network · Edge Case Tests")
    print("=" * 70)
    test_stage_1()
    test_stage_2()
    test_stage_3()
    test_stage_4()
    print("\nDone.")
