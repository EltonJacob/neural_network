"""
Day 3 — Neural Network from Scratch
Phase 1 — Foundations · 60-minute timed session

RULES:
  - NumPy only for the model. sklearn.datasets is allowed for data generation only.
  - No PyTorch.
  - Read all 4 stages in README.md first.
  - Design architecture in comments BEFORE writing code.
  - print() is your only debugger.

ARCHITECTURE NOTES (fill in during your 5-minute design phase):
  -
  -
  -

KEY EQUATIONS (memorize):
  Forward:
    Z[l]      = W[l] @ A[l-1] + b[l]
    A[l]      = activation(Z[l])
  Backward:
    dZ[l]     = dA[l] * activation_backward(Z[l])
    dW[l]     = (1/m) * dZ[l] @ A[l-1].T
    db[l]     = (1/m) * np.sum(dZ[l], axis=1, keepdims=True)
    dA[l-1]   = W[l].T @ dZ[l]
  Init (He, for ReLU):
    W[l]      = np.random.randn(n[l], n[l-1]) * sqrt(2 / n[l-1])
    b[l]      = np.zeros((n[l], 1))
  Loss (binary):
    BCE       = -mean(y*log(p) + (1-y)*log(1-p))   # clip p to [1e-7, 1-1e-7]
  Gradient check:
    max rel error = |grad_numerical - grad_analytical| / (|grad_numerical| + |grad_analytical|)
"""

import numpy as np
from typing import List, Tuple, Dict
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits


# -----------------------------------------------------------------------------
# DATA GENERATION (scaffolding — sklearn allowed for data only)
# -----------------------------------------------------------------------------

def generate_xor() -> Tuple[np.ndarray, np.ndarray]:
    """The classic 4-point XOR — your network MUST learn this."""
    X = np.array([[0,0,1,1],[0,1,0,1]]).astype(float)
    y = np.array([0,1,1,0]).reshape(1,-1).astype(float)
    return (X,y)


def generate_moons(n_samples: int = 1000, noise: float = 0.1,
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Nonlinear 2D blob — sklearn.datasets.make_moons."""
    X,y = make_moons(n_samples=n_samples,noise=noise,random_state=random_state)
    X = X.T
    y = y.reshape(1,-1).astype(float)
    return (X,y)


def generate_digits() -> Tuple[np.ndarray, np.ndarray]:
    """sklearn digits dataset — for Stage 4 multi-class benchmark."""
    data = load_digits()
    X = data.data
    y = data.target
    X = X.T
    y = y.reshape(1,-1)
    return (X,y)


def train_val_split(X: np.ndarray, y: np.ndarray, val_frac: float = 0.2,
                    seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """80/20 split — done by hand, no train_test_split."""
    np.random.seed(seed)
    index = np.random.permutation(X.shape[1])
    train_size = int((1-val_frac) * X.shape[1])
    X = X[:,index]
    y = y[:,index]
    X_train = X[:,:train_size]
    y_train = y[:,:train_size]
    X_val = X[:,train_size:]
    y_val= y[:,train_size:]
    return (X_train,y_train,X_val,y_val)


# -----------------------------------------------------------------------------
# STAGE 1 — FORWARD PASS
# -----------------------------------------------------------------------------

class NeuralNetworkScratch:
    """Fully-connected feedforward network — NumPy only."""

    def __init__(self, layer_dims: List[int], activations: List[str], seed: int = 0):
        # TODO: validate len(activations) == len(layer_dims) - 1
        # TODO: He-init self.params = {"W1": ..., "b1": ..., "W2": ..., ...}
        # TODO: store self.layer_dims, self.activations, self.caches = [], self.loss_history = []
        if len(activations)!=len(layer_dims)-1:
            raise ValueError("activation and layer_dim mismatch")
        np.random.seed(seed)
        self.layer_dims = layer_dims
        self.activations = activations
        self.caches = []
        self.loss_history = []
        self.params = {}
        self.AL = None
        for i in range(1,len(layer_dims)):
            key = f"W{i}"
            self.params[f"W{i}"]=np.random.randn(layer_dims[i],layer_dims[i-1]) * np.sqrt(2/layer_dims[i-1])
            self.params[f"b{i}"]=np.zeros((layer_dims[i],1))

        


    # --- activations ---------------------------------------------------------

    @staticmethod
    def relu(Z: np.ndarray) -> np.ndarray:
        return np.maximum(0,Z)

    @staticmethod
    def sigmoid(Z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid — clip Z to [-500, 500]."""
        Z= np.clip(Z,-500,500)
        return 1/(1+np.exp(-Z))

    # --- forward -------------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Run a forward pass. Cache (A_prev, Z, W, b) per layer for backprop."""
        self.caches.clear()
        A=X
        for i in range(1,len(self.layer_dims)):
            A_prev = A
            Z = self.params[f"W{i}"]@A + self.params[f"b{i}"]
            if self.activations[i-1]=='relu':
                A = self.relu(Z)
            elif self.activations[i-1]=='sigmoid':
                A = self.sigmoid(Z)
            self.caches.append((A_prev,Z,self.params[f"W{i}"],self.params[f"b{i}"]))
        self.AL = A
        return A


# -----------------------------------------------------------------------------
# STAGE 2 — BACKPROPAGATION
# -----------------------------------------------------------------------------

    @staticmethod
    def relu_backward(dA: np.ndarray, cache_Z: np.ndarray) -> np.ndarray:
        """dZ = dA * (Z > 0)."""
        dZ = dA * (cache_Z > 0)
        return dZ

    @staticmethod
    def sigmoid_backward(dA: np.ndarray, cache_Z: np.ndarray) -> np.ndarray:
        """dZ = dA * sigmoid(Z) * (1 - sigmoid(Z))."""
        dZ = dA * NeuralNetworkScratch.sigmoid(cache_Z) * (1-NeuralNetworkScratch.sigmoid(cache_Z))
        return dZ

    def backward(self, y_true: np.ndarray) -> Dict[str, np.ndarray]:
        """Walk caches in reverse. Return {dW1, db1, dW2, db2, ...}."""
        result = {}
        m = y_true.shape[1]
        dA = -(y_true/self.AL) + (1-y_true)/(1-self.AL)
        for i in range(len(self.caches)-1,-1,-1):
            if self.activations[i] == "relu":
                dZ = self.relu_backward(dA,self.caches[i][1])
            elif self.activations[i] == "sigmoid":
                dZ = self.sigmoid_backward(dA,self.caches[i][1])
            dW = (1/m) * dZ @ self.caches[i][0].T
            db = (1/m) * np.sum(dZ,axis=1,keepdims=True)
            dA = self.caches[i][2].T @ dZ
            result[f"dW{i+1}"]=dW
            result[f"db{i+1}"]=db
        return result


    def update_params(self, grads: Dict[str, np.ndarray], lr: float) -> None:
        """Apply W -= lr * dW, b -= lr * db for every layer."""
        for i in range(1,len(self.layer_dims)):
            self.params[f'W{i}'] -= lr * grads[f"dW{i}"]
            self.params[f'b{i}'] -= lr * grads[f"db{i}"]


# -----------------------------------------------------------------------------
# STAGE 3 — ACTIVATIONS + OPTIMIZATION
# -----------------------------------------------------------------------------

    @staticmethod
    def tanh(Z: np.ndarray) -> np.ndarray:
        return np.tanh(Z)

    @staticmethod
    def tanh_backward(dA: np.ndarray, cache_Z: np.ndarray) -> np.ndarray:
        """dZ = dA * (1 - tanh(Z)^2)."""
        return dA * (1-NeuralNetworkScratch.tanh(cache_Z)**2)

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1,
            epochs: int = 1000, batch_size: int = 32) -> List[float]:
        """Mini-batch GD. Shuffle each epoch. Returns loss-per-epoch."""
        for epoch in range(epochs):
            index = np.random.permutation(X.shape[1])
            X = X[:,index]
            y = y[:,index]
            for batch in range(0,X.shape[1],batch_size):
                start = batch
                end = batch + batch_size
                X_batched = X[:,start:end]
                y_batched = y[:,start:end]
                output = self.forward(X_batched)
                grad = self.backward(y_batched)
                self.update_params(grad,lr)
            output = self.forward(X)
            loss = -np.mean(y * np.log(output + 1e-7) + (1-y) * np.log(1 - output + 1e-7))
            self.loss_history.append(loss)
        return self.loss_history
            

    def gradient_check(self, X: np.ndarray, y: np.ndarray,
                       epsilon: float = 1e-5) -> float:
        """Compare analytical vs numerical grads. Return max relative error."""
        output = self.forward(X)
        grads = self.backward(y)
        errors = []
        for key in self.params:
            for idx, val in np.ndenumerate(self.params[key]):
                self.params[key][idx] += epsilon
                output = self.forward(X)
                loss_plus = -np.mean(y * np.log(output + 1e-7) + (1-y) * np.log(1 - output + 1e-7))
                self.params[key][idx] -= 2*epsilon
                output = self.forward(X)
                loss_minus = -np.mean(y * np.log(output + 1e-7) + (1-y) * np.log(1 - output + 1e-7))
                self.params[key][idx] += epsilon
                numerical_grad = (loss_plus - loss_minus)/(2*epsilon)
                grad_key = "d"+key
                analytical_grad = grads[grad_key][idx]
                error = np.abs(numerical_grad-analytical_grad)/(np.abs(numerical_grad)+np.abs(analytical_grad)+1e-8)
                errors.append(error)
        return np.max(errors)



# -----------------------------------------------------------------------------
# STAGE 4 — REGULARIZATION + DIAGNOSTICS (bonus)
# -----------------------------------------------------------------------------

    def fit_with_dropout(self, X: np.ndarray, y: np.ndarray,
                         keep_prob: float = 0.8, lr: float = 0.1,
                         epochs: int = 1000) -> List[float]:
        """Inverted dropout during training only."""
        raise NotImplementedError

    def fit_with_l2(self, X: np.ndarray, y: np.ndarray, alpha: float = 0.01,
                    lr: float = 0.1, epochs: int = 1000) -> List[float]:
        """L2 regularization on every W (bias unregularized)."""
        raise NotImplementedError

    def plot_decision_boundary(self, X: np.ndarray, y: np.ndarray,
                                resolution: float = 0.01) -> None:
        """ASCII decision boundary — print() to terminal is fine."""
        raise NotImplementedError

    def layer_stats(self, layer_idx: int) -> Dict[str, float]:
        """Return {mean_W, std_W, mean_grad, std_grad} for one layer."""
        raise NotImplementedError


# -----------------------------------------------------------------------------
# MAIN — quick smoke run when executing this file directly
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    X, y = generate_xor()
    nn = NeuralNetworkScratch([2, 4, 1], ['relu', 'sigmoid'])
    losses = nn.fit(X, y, lr=0.1, epochs=1000, batch_size=4)
    print("First loss:", losses[0])
    print("Last loss:", losses[-1])
    print("Final output:", nn.forward(X))
    print("Expected:   ", y)
    print("Gradient check error:", nn.gradient_check(X, y))
