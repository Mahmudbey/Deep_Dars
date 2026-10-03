"""1-mavzu (01_neuron): Practice dasturlash topshiriqlari (17-30) yechimlari.

Ishga tushirish:  python -W error::DeprecationWarning solutions.py
Muhit: requirements.txt (numpy 2.4.4, scikit-learn 1.8.0, torch 2.14.0), CPU.
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, make_circles, make_moons
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def header(k, title):
    print(f"\n=== Task {k}: {title} ===")


# ---------------------------------------------------------------- 17
def sigmoid(z):
    """Barqaror sigmoid: exp(-ln(1+e^{-z}))."""
    return np.exp(-np.logaddexp(0, -z))


def activations(z, alpha=0.01):
    """{nom: (phi(z), phi'(z))} lug'ati."""
    s = sigmoid(z)
    t = np.tanh(z)
    return {
        "sigmoid": (s, s * (1 - s)),
        "tanh": (t, 1 - t ** 2),
        "relu": (np.maximum(0, z), (z > 0).astype(float)),
        "leaky_relu": (np.where(z > 0, z, alpha * z), np.where(z > 0, 1.0, alpha)),
        "softplus": (np.logaddexp(0, z), s),
    }


def task_17():
    header(17, "activations and derivatives vs autograd")
    z = np.linspace(-3, 3, 7)
    acts = activations(z)
    tf = {
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "relu": torch.relu,
        "leaky_relu": lambda t: nn.functional.leaky_relu(t, 0.01),
        "softplus": nn.functional.softplus,
    }
    print("z =", z)
    for name, (f, df) in acts.items():
        zt = torch.tensor(z, dtype=torch.float64, requires_grad=True)
        out = tf[name](zt)
        out.sum().backward()
        ok_f = np.allclose(f, out.detach().numpy(), atol=1e-12)
        ok_d = np.allclose(df, zt.grad.numpy(), atol=1e-12)
        print(f"{name:10s} phi'(z) = {np.round(df, 4)}  match: {ok_f and ok_d}")


# ---------------------------------------------------------------- 18
def merge_linear(l1: nn.Linear, l2: nn.Linear) -> nn.Linear:
    """l2(l1(x)) ga teng bitta nn.Linear."""
    W1, b1, W2, b2 = l1.weight, l1.bias, l2.weight, l2.bias
    m = nn.Linear(l1.in_features, l2.out_features)
    with torch.no_grad():
        m.weight.copy_(W2 @ W1)
        m.bias.copy_(W2 @ b1 + b2)
    return m


def task_18():
    header(18, "two linear layers = one linear layer")
    torch.manual_seed(42)
    l1, l2 = nn.Linear(4, 3), nn.Linear(3, 2)
    m = merge_linear(l1, l2)
    x = torch.randn(5, 4)
    with torch.no_grad():
        diff = (l2(l1(x)) - m(x)).abs().max().item()
    print("merged weight shape:", tuple(m.weight.shape), "bias shape:", tuple(m.bias.shape))
    print("params: two layers =", sum(p.numel() for p in [*l1.parameters(), *l2.parameters()]),
          "| merged =", sum(p.numel() for p in m.parameters()))
    print("max|diff| < 1e-6:", diff < 1e-6)


# ---------------------------------------------------------------- 19
def bce_naive(z, y):
    with np.errstate(all="ignore"):
        p = 1 / (1 + np.exp(-z))
        return -y * np.log(p) - (1 - y) * np.log(1 - p)


def bce_stable(z, y):
    # ln(1+e^z) - y z = max(z,0) - y z + ln(1+e^{-|z|})
    return np.maximum(z, 0) - y * z + np.log1p(np.exp(-np.abs(z)))


def task_19():
    header(19, "numerically stable binary cross-entropy")
    z = np.array([-800.0, -30.0, 0.0, 30.0, 800.0])
    y = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    ref = nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(z), torch.tensor(y), reduction="none").numpy()
    fmt = lambda a: "[" + ", ".join(f"{v:.4g}" for v in a) + "]"
    print("naive :", fmt(bce_naive(z, y)))
    print("stable:", fmt(bce_stable(z, y)))
    print("torch :", fmt(ref))
    print("stable == torch:", np.allclose(bce_stable(z, y), ref))


# ---------------------------------------------------------------- 20
def perceptron(X, y, eta=1.0, T=100):
    """Rozenblatt qoidasi. Qaytaradi: w, b, epoxalar, yangilanishlar, yaqinlashdimi."""
    w, b, upd = np.zeros(X.shape[1]), 0.0, 0
    for t in range(1, T + 1):
        err = 0
        for x, yk in zip(X, y):
            d = yk - int(w @ x + b >= 0)
            if d != 0:
                w, b = w + eta * d * x, b + eta * d
                err += 1
        upd += err
        if err == 0:
            return w, b, t, upd, True
    return w, b, T, upd, False


def task_20():
    header(20, "perceptron on iris (setosa vs versicolor)")
    iris = load_iris()
    mask = iris.target < 2
    X = iris.data[mask][:, 2:4]          # petal length, petal width
    y = (iris.target[mask] == 1).astype(int)
    w, b, T, upd, conv = perceptron(X, y)
    acc = np.mean((X @ w + b >= 0).astype(int) == y)
    print(f"w = {np.round(w, 2)}, b = {b:.1f}, epochs = {T}, updates = {upd}, converged = {conv}")
    print(f"train accuracy = {acc:.3f}")


# ---------------------------------------------------------------- 21
def pocket_perceptron(X, y, eta=1.0, T=100):
    """Perceptron + "cho'ntak": har yangilanishdan keyin eng yaxshi (w, b) saqlanadi."""
    w, b = np.zeros(X.shape[1]), 0.0
    best = (w.copy(), b, np.mean((X @ w + b >= 0) == y))
    last_err = None
    for _ in range(T):
        err = 0
        for x, yk in zip(X, y):
            d = yk - int(w @ x + b >= 0)
            if d != 0:
                w, b = w + eta * d * x, b + eta * d
                err += 1
                acc = np.mean((X @ w + b >= 0) == y)
                if acc > best[2]:
                    best = (w.copy(), b, acc)
        last_err = err
    return best, last_err


def task_21():
    header(21, "pocket perceptron on iris (versicolor vs virginica)")
    iris = load_iris()
    mask = iris.target > 0
    X = iris.data[mask][:, 2:4]
    y = (iris.target[mask] == 2).astype(int)
    _, _, T, upd, conv = perceptron(X, y, T=100)
    print(f"plain perceptron: converged = {conv}, updates in {T} epochs = {upd}")
    (w, b, acc), last_err = pocket_perceptron(X, y, T=100)
    print(f"errors in epoch 100 = {last_err}")
    print(f"pocket: w = {np.round(w, 2)}, b = {b:.1f}, best train accuracy = {acc:.2f}")


# ---------------------------------------------------------------- 22
def load_bc():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), ytr.astype(float), yte.astype(float)


def train_logistic(X, y, eta, epochs):
    n, d = X.shape
    w, b, hist = np.zeros(d), 0.0, []
    for _ in range(epochs + 1):
        z = X @ w + b
        hist.append(np.mean(np.logaddexp(0, z) - y * z))
        g = (sigmoid(z) - y) / n
        w, b = w - eta * (X.T @ g), b - eta * g.sum()
    return w, b, hist


def task_22():
    header(22, "logistic neuron with gradient descent (breast cancer)")
    Xtr, Xte, ytr, yte = load_bc()
    w, b, hist = train_logistic(Xtr, ytr, eta=0.5, epochs=300)
    for e in (0, 10, 100, 300):
        print(f"epoch {e:3d}: train BCE = {hist[e]:.4f}")
    acc_tr = np.mean(((Xtr @ w + b) >= 0) == ytr)
    acc_te = np.mean(((Xte @ w + b) >= 0) == yte)
    print(f"train accuracy = {acc_tr:.4f}, test accuracy = {acc_te:.4f}")
    top = np.argsort(-np.abs(w))[:3]
    names = load_breast_cancer().feature_names
    print("largest |w|:", ", ".join(f"{names[i]} ({w[i]:.3f})" for i in top))


# ---------------------------------------------------------------- 23
def mlp_grads(X, y, W1, b1, w2, b2):
    """Bir yashirin ReLU qatlamli tarmoq uchun qo'lda backprop (1-ma'ruza formulalari)."""
    n = X.shape[0]
    Z1 = X @ W1 + b1
    H = np.maximum(0, Z1)
    z = H @ w2 + b2
    L = np.mean(np.logaddexp(0, z) - y * z)
    g2 = (sigmoid(z) - y) / n
    gw2, gb2 = H.T @ g2, g2.sum()
    G1 = np.outer(g2, w2) * (Z1 > 0)
    gW1, gb1 = X.T @ G1, G1.sum(0)
    return L, gW1, gb1, gw2, gb2


def task_23():
    header(23, "hand-written backprop vs autograd")
    rng = np.random.default_rng(42)
    n, d, m = 8, 3, 5
    X = rng.normal(size=(n, d))
    y = rng.integers(0, 2, n).astype(float)
    W1, b1 = rng.normal(size=(d, m)), rng.normal(size=m)
    w2, b2 = rng.normal(size=m), float(rng.normal())
    L, *g = mlp_grads(X, y, W1, b1, w2, b2)
    P = [torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in (W1, b1, w2, b2)]
    Xt, yt = torch.tensor(X), torch.tensor(y)
    zt = torch.relu(Xt @ P[0] + P[1]) @ P[2] + P[3]
    Lt = nn.functional.binary_cross_entropy_with_logits(zt, yt)
    Lt.backward()
    print(f"loss (numpy) = {L:.6f}, loss (torch) = {Lt.item():.6f}")
    for name, a, p in zip(["W1", "b1", "w2", "b2"], g, P):
        print(f"grad {name:2s}: max|diff| < 1e-12: {np.abs(a - p.grad.numpy()).max() < 1e-12}")
    print("dL/db2 =", round(float(g[3]), 6))


# ---------------------------------------------------------------- 24
def task_24():
    header(24, "step size and the smoothness constant beta")
    Xtr, _, ytr, _ = load_bc()
    n = Xtr.shape[0]
    Xt = np.hstack([Xtr, np.ones((n, 1))])
    beta = np.linalg.eigvalsh(Xt.T @ Xt).max() / (4 * n)
    print(f"n = {n}, beta = {beta:.4f}, 2/beta = {2 / beta:.4f}")
    for eta in (0.01, 0.1, 0.9 * 2 / beta, 5.0, 50.0, 500.0):
        _, _, hist = train_logistic(Xtr, ytr, eta=eta, epochs=100)
        mono = bool(np.all(np.diff(hist) <= 1e-12))
        print(f"eta = {eta:7.4f}: BCE after 100 steps = {hist[-1]:.4f}, monotone = {mono}")


# ---------------------------------------------------------------- 25
def task_25():
    header(25, "2-16-1 tanh MLP in NumPy on make_moons")
    X, y = make_moons(n_samples=400, noise=0.2, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y.astype(float), test_size=0.25, random_state=42)
    rng = np.random.default_rng(42)
    W1 = rng.normal(0, 1, (2, 16)) * np.sqrt(1 / 2)
    b1 = np.zeros(16)
    W2 = rng.normal(0, 1, (16, 1)) * np.sqrt(1 / 16)
    b2 = np.zeros(1)
    Y = ytr[:, None]
    n, eta = len(Xtr), 0.5

    def forward(X):
        H = np.tanh(X @ W1 + b1)
        return H, H @ W2 + b2

    for s in range(3001):
        H, z = forward(Xtr)
        if s % 1000 == 0:
            print(f"step {s:4d}: BCE = {np.mean(np.logaddexp(0, z) - Y * z):.4f}")
        G2 = (sigmoid(z) - Y) / n
        gW2, gb2 = H.T @ G2, G2.sum(0)
        G1 = (G2 @ W2.T) * (1 - H ** 2)
        gW1, gb1 = Xtr.T @ G1, G1.sum(0)
        W1 -= eta * gW1; b1 -= eta * gb1; W2 -= eta * gW2; b2 -= eta * gb2
    acc = lambda X, y: np.mean((forward(X)[1].ravel() >= 0) == y)
    print(f"train accuracy = {acc(Xtr, ytr):.4f}, test accuracy = {acc(Xte, yte):.4f}")


# ---------------------------------------------------------------- 26
def task_26():
    header(26, "activation comparison on make_circles (PyTorch)")
    X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    for name, act in [("sigmoid", nn.Sigmoid), ("tanh", nn.Tanh), ("relu", nn.ReLU)]:
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(2, 8), act(), nn.Linear(8, 1))
        opt = torch.optim.SGD(model.parameters(), lr=0.5)
        lossf = nn.BCEWithLogitsLoss()
        for _ in range(1000):
            opt.zero_grad()
            loss = lossf(model(X), y)
            loss.backward()
            opt.step()
        with torch.no_grad():
            out = model(X)
            L = lossf(out, y).item()
            acc = ((out >= 0).float() == y).float().mean().item()
        print(f"{name:8s}: final BCE = {L:.4f}, accuracy = {acc:.3f}")


# ---------------------------------------------------------------- 27
def deep_net(act, depth=20, width=32, init=None):
    layers = []
    for _ in range(depth):
        lin = nn.Linear(width, width)
        if init is not None:
            init(lin.weight)
            nn.init.zeros_(lin.bias)
        layers += [lin, act()]
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def task_27():
    header(27, "vanishing gradients: 20 layers, sigmoid vs ReLU")
    torch.manual_seed(42)
    x = torch.randn(64, 32)
    y = (torch.rand(64, 1) > 0.5).float()
    cfg = [("sigmoid", nn.Sigmoid, None),
           ("relu+He", nn.ReLU, lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"))]
    for name, act, init in cfg:
        torch.manual_seed(42)
        net = deep_net(act, init=init)
        loss = nn.functional.binary_cross_entropy_with_logits(net(x), y)
        loss.backward()
        first = net[0].weight.grad.norm().item()
        last = net[-3].weight.grad.norm().item()
        print(f"{name:8s}: ||grad W_1|| = {first:.3e}, ||grad W_20|| = {last:.3e}, ratio = {first / last:.3e}")


# ---------------------------------------------------------------- 28
def task_28():
    header(28, "initialization and activation scale (10 ReLU layers)")
    torch.manual_seed(42)
    x = torch.randn(1000, 256)
    inits = {"N(0,0.01^2)": lambda w: nn.init.normal_(w, 0.0, 0.01),
             "Xavier": nn.init.xavier_normal_,
             "He": lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu")}
    for name, init in inits.items():
        torch.manual_seed(42)
        h, stds = x, []
        with torch.no_grad():
            for _ in range(10):
                W = torch.empty(256, 256)
                init(W)
                h = torch.relu(h @ W.T)
                stds.append(h.std().item())
        print(f"{name:12s}: std after layer 1/5/10 = {stds[0]:.3e} {stds[4]:.3e} {stds[9]:.3e}")


# ---------------------------------------------------------------- 29
def task_29():
    header(29, "universal approximation: sin(x) with one hidden ReLU layer")
    x = torch.linspace(-math.pi, math.pi, 200).unsqueeze(1)
    y = torch.sin(x)
    for m in (2, 8, 32):
        torch.manual_seed(42)
        net = nn.Sequential(nn.Linear(1, m), nn.ReLU(), nn.Linear(m, 1))
        opt = torch.optim.Adam(net.parameters(), lr=0.01)
        for _ in range(2000):
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(x), y)
            loss.backward()
            opt.step()
        with torch.no_grad():
            err = (net(x) - y).abs()
        print(f"m = {m:2d}: params = {3 * m + 1:3d}, MSE = {err.pow(2).mean().item():.2e}, max|err| = {err.max().item():.4f}")


# ---------------------------------------------------------------- 30
def task_30():
    header(30, "use case: '8' detector on load_digits, class imbalance")
    X, t = load_digits(return_X_y=True)
    y = (t == 8).astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(X / 16.0, y, test_size=0.3, stratify=y, random_state=42)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr).unsqueeze(1)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    npos, nneg = ytr.sum(), len(ytr) - ytr.sum()
    print(f"train: positives = {int(npos)}, negatives = {int(nneg)}, pos_weight = {nneg / npos:.3f}")
    for label, pw in [("plain", None), ("pos_weight", torch.tensor([nneg / npos]))]:
        torch.manual_seed(42)
        model = nn.Linear(64, 1)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        for _ in range(300):
            opt.zero_grad()
            loss = lossf(model(Xtr_t), ytr_t)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = (model(Xte_t).squeeze(1) >= 0).numpy().astype(int)
        print(f"{label:10s}: accuracy = {accuracy_score(yte, pred):.4f}, precision = {precision_score(yte, pred):.4f}, "
              f"recall = {recall_score(yte, pred):.4f}, F1 = {f1_score(yte, pred):.4f}")


def main():
    t0 = time.perf_counter()
    for k in range(17, 31):
        globals()[f"task_{k}"]()
    import sys
    print(f"\n[total runtime {time.perf_counter() - t0:.1f} s]", file=sys.stderr)


if __name__ == "__main__":
    main()
