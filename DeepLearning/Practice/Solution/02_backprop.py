"""2-mavzu (02_backprop): Practice dasturlash topshiriqlari (17-30) yechimlari.

Ishga tushirish:  python -W error::DeprecationWarning solutions.py
Muhit: requirements.txt (numpy 2.4.4, scikit-learn 1.8.0, torch 2.14.0), CPU.
"""
import copy
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def header(k, title):
    print(f"\n=== Task {k}: {title} ===")


# ---------------------------------------------------------------- 17
class Value:
    """Skalyar reverse-mode autograd."""

    def __init__(self, data, parents=()):
        self.data, self.grad, self._parents = float(data), 0.0, parents
        self._backward = lambda: None

    def __add__(self, o):
        out = Value(self.data + o.data, (self, o))

        def bw():
            self.grad += out.grad
            o.grad += out.grad
        out._backward = bw
        return out

    def __mul__(self, o):
        out = Value(self.data * o.data, (self, o))

        def bw():
            self.grad += out.grad * o.data
            o.grad += out.grad * self.data
        out._backward = bw
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,))

        def bw():
            self.grad += out.grad * (1 - t * t)
        out._backward = bw
        return out

    def backward(self):
        order, seen = [], set()

        def topo(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v._parents:
                    topo(p)
                order.append(v)
        topo(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()


def task_17():
    header(17, "micro-autograd: f = tanh(x*y + z) * x")
    x, y, z = Value(0.5), Value(-2.0), Value(1.5)
    f = (x * y + z).tanh() * x
    f.backward()
    xt, yt, zt = (torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (0.5, -2.0, 1.5))
    ft = torch.tanh(xt * yt + zt) * xt
    ft.backward()
    print(f"f = {f.data:.6f} (torch {ft.item():.6f})")
    print(f"df/dx = {x.grad:.6f}, df/dy = {y.grad:.6f}, df/dz = {z.grad:.6f}")
    print(f"torch : {xt.grad.item():.6f}, {yt.grad.item():.6f}, {zt.grad.item():.6f}")


# ---------------------------------------------------------------- 18
def log_softmax(Z):
    m = Z.max(axis=1, keepdims=True)
    return Z - m - np.log(np.exp(Z - m).sum(axis=1, keepdims=True))


def cross_entropy(Z, y):
    return -log_softmax(Z)[np.arange(len(y)), y].mean()


def task_18():
    header(18, "stable softmax cross-entropy")
    Z = np.array([[1000.0, 1001.0, 1002.0], [-5.0, 0.0, 5.0]])
    y = np.array([2, 0])
    with np.errstate(all="ignore"):
        P_naive = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)
        ce_naive = -np.log(P_naive[np.arange(2), y]).mean()
    ce = cross_entropy(Z, y)
    ce_t = F.cross_entropy(torch.tensor(Z), torch.tensor(y)).item()
    print(f"naive CE = {ce_naive}, stable CE = {ce:.6f}, torch CE = {ce_t:.6f}")
    print("softmax row 1 =", np.round(np.exp(log_softmax(Z))[0], 4))


# ---------------------------------------------------------------- 19
def task_19():
    header(19, "softmax Jacobian: autograd vs diag(p) - p p^T")
    z = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    J_auto = torch.autograd.functional.jacobian(lambda t: torch.softmax(t, 0), z)
    p = torch.softmax(z, 0)
    J = torch.diag(p) - torch.outer(p, p)
    print("p =", np.round(p.numpy(), 4))
    print("J =\n", np.round(J.numpy(), 4))
    print("match:", torch.allclose(J, J_auto, atol=1e-12), "| symmetric:", torch.allclose(J, J.T),
          "| J @ 1 = 0:", torch.allclose(J @ torch.ones(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)))


# ---------------------------------------------------------------- 20
class MLP:
    """Qatorlar-namunalar konvensiyasi: Z = A W + b. Chiqish qatlamida softmax + cross-entropy."""

    def __init__(self, sizes, act="relu", rng=None):
        rng = rng if rng is not None else np.random.default_rng(42)
        gain = 2.0 if act == "relu" else 1.0
        self.W = [rng.normal(0, np.sqrt(gain / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.act = act

    def phi(self, Z):
        return np.maximum(0, Z) if self.act == "relu" else np.tanh(Z)

    def dphi(self, Z):
        return (Z > 0).astype(float) if self.act == "relu" else 1 - np.tanh(Z) ** 2

    def forward(self, X):
        self.A, self.Z = [X], []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = self.A[-1] @ W + b
            self.Z.append(Z)
            if i < len(self.W) - 1:
                self.A.append(self.phi(Z))
        return Z

    def loss(self, X, y):
        return cross_entropy(self.forward(X), y)

    def backward(self, y):
        n = len(y)
        P = np.exp(log_softmax(self.Z[-1]))
        delta = P.copy()
        delta[np.arange(n), y] -= 1
        delta /= n                                     # delta^(L) = (P - Y)/n
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = self.A[i].T @ delta
            gb[i] = delta.sum(0)
            if i > 0:
                delta = (delta @ self.W[i].T) * self.dphi(self.Z[i - 1])
        return gW, gb


def task_20():
    header(20, "NumPy MLP backprop + gradient check (tanh, float64)")
    rng = np.random.default_rng(42)
    X = rng.normal(size=(5, 4))
    y = rng.integers(0, 3, 5)
    net = MLP([4, 6, 3], act="tanh", rng=rng)
    net.loss(X, y)
    gW, gb = net.backward(y)
    eps = 1e-6
    for name, params, grads in [("W", net.W, gW), ("b", net.b, gb)]:
        for i, (P, G) in enumerate(zip(params, grads)):
            num = np.zeros_like(P)
            for idx in np.ndindex(P.shape):
                old = P[idx]
                P[idx] = old + eps; lp = net.loss(X, y)
                P[idx] = old - eps; lm = net.loss(X, y)
                P[idx] = old
                num[idx] = (lp - lm) / (2 * eps)
            rel = np.abs(G - num).max() / max(np.abs(num).max(), 1e-12)
            print(f"{name}{i + 1}: shape {P.shape}, relative error = {rel:.1e}, ok: {rel < 1e-7}")


# ---------------------------------------------------------------- 21
def task_21():
    header(21, "finite differences: error vs epsilon for sin at t = 1")
    t, exact = 1.0, math.cos(1.0)
    print(" eps     central    forward")
    best = None
    for k in range(1, 13):
        e = 10.0 ** (-k)
        c = abs((math.sin(t + e) - math.sin(t - e)) / (2 * e) - exact)
        f = abs((math.sin(t + e) - math.sin(t)) / e - exact)
        best = min(best or (c, k), (c, k))
        print(f"1e-{k:02d}  {c:.2e}  {f:.2e}")
    print(f"best central eps = 1e-{best[1]:02d}")


# ---------------------------------------------------------------- 22
def load_digits_split():
    X, y = load_digits(return_X_y=True)
    return train_test_split(X / 16.0, y, test_size=0.3, stratify=y, random_state=42)


def task_22():
    header(22, "NumPy MLP 64-64-10 on digits, mini-batch SGD")
    Xtr, Xte, ytr, yte = load_digits_split()
    rng = np.random.default_rng(42)
    net = MLP([64, 64, 10], act="relu", rng=rng)
    B, eta = 32, 0.1
    for epoch in range(1, 21):
        idx = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), B):
            j = idx[s:s + B]
            net.forward(Xtr[j])
            gW, gb = net.backward(ytr[j])
            for i in range(len(net.W)):
                net.W[i] -= eta * gW[i]
                net.b[i] -= eta * gb[i]
        if epoch % 5 == 0:
            L = net.loss(Xtr, ytr)
            acc = np.mean(net.forward(Xte).argmax(1) == yte)
            print(f"epoch {epoch:2d}: train CE = {L:.4f}, test accuracy = {acc:.4f}")


# ---------------------------------------------------------------- 23
def digits_tensors():
    Xtr, Xte, ytr, yte = load_digits_split()
    f = lambda a: torch.tensor(a, dtype=torch.float32)
    return f(Xtr), f(Xte), torch.tensor(ytr), torch.tensor(yte)


def train_torch(model, opt, Xtr, ytr, epochs=20, lossf=None, B=32):
    lossf = lossf or nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=B, shuffle=True,
                        generator=torch.Generator().manual_seed(42))
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()
    return model


def evaluate(model, X, y, lossf=None):
    lossf = lossf or nn.CrossEntropyLoss()
    with torch.no_grad():
        out = model(X)
        return lossf(out, y).item(), (out.argmax(1) == y).float().mean().item()


def make_mlp():
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))


def task_23():
    header(23, "optimizers on digits (PyTorch, 20 epochs, B = 32)")
    Xtr, Xte, ytr, yte = digits_tensors()
    configs = {
        "SGD lr=0.1": lambda p: torch.optim.SGD(p, lr=0.1),
        "SGD+momentum lr=0.05": lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9),
        "Adam lr=1e-3": lambda p: torch.optim.Adam(p, lr=1e-3),
        "AdamW lr=1e-3 wd=1e-2": lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=1e-2),
    }
    for name, make_opt in configs.items():
        torch.manual_seed(42)
        model = make_mlp()
        train_torch(model, make_opt(model.parameters()), Xtr, ytr)
        Ltr, _ = evaluate(model, Xtr, ytr)
        _, acc = evaluate(model, Xte, yte)
        print(f"{name:22s}: train CE = {Ltr:.4f}, test accuracy = {acc:.4f}")


# ---------------------------------------------------------------- 24
def task_24():
    header(24, "bug: extra Softmax before CrossEntropyLoss")
    Xtr, Xte, ytr, yte = digits_tensors()
    for name, extra in [("correct (logits)", []), ("double softmax", [nn.Softmax(dim=1)])]:
        torch.manual_seed(42)
        model = nn.Sequential(make_mlp(), *extra)
        train_torch(model, torch.optim.Adam(model.parameters(), lr=1e-3), Xtr, ytr)
        Ltr, _ = evaluate(model, Xtr, ytr)
        _, acc = evaluate(model, Xte, yte)
        print(f"{name:17s}: train CE = {Ltr:.4f}, test accuracy = {acc:.4f}")
    print(f"lower bound of CE with double softmax (K=10): ln(e+9) - 1 = {math.log(math.e + 9) - 1:.4f}")


# ---------------------------------------------------------------- 25
def task_25():
    header(25, "gradient accumulation and forgotten zero_grad")
    Xtr, _, ytr, _ = digits_tensors()
    xb, yb = Xtr[:32], ytr[:32]
    torch.manual_seed(42)
    model = nn.Linear(64, 10)
    lossf = nn.CrossEntropyLoss()
    lossf(model(xb), yb).backward()
    g_full = model.weight.grad.clone()
    model.zero_grad()
    for k in range(4):                       # 4 ta mikro-paket, har biri 8 ta namuna
        (lossf(model(xb[8 * k:8 * k + 8]), yb[8 * k:8 * k + 8]) / 4).backward()
    diff = (model.weight.grad - g_full).abs().max().item()
    print(f"accumulated (4 x 8, loss/4) == full batch 32: {diff < 1e-6}")
    model.zero_grad()
    lossf(model(xb), yb).backward()
    lossf(model(xb), yb).backward()          # zero_grad() unutildi
    ratio = (model.weight.grad.norm() / g_full.norm()).item()
    print(f"without zero_grad: ||grad|| / ||grad_true|| = {ratio:.4f}")


# ---------------------------------------------------------------- 26
def task_26():
    header(26, "exploding gradients and clip_grad_norm_")
    Xtr, _, ytr, _ = digits_tensors()

    def deep(std):
        layers = []
        for _ in range(10):
            lin = nn.Linear(64, 64)
            nn.init.normal_(lin.weight, 0.0, std)
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.ReLU()]
        return nn.Sequential(*layers, nn.Linear(64, 10))

    he = math.sqrt(2 / 64)
    for name, std, clip in [("std=0.25, no clip", 0.25, None), ("std=0.25, clip=1.0", 0.25, 1.0),
                            ("He std=0.177, no clip", he, None)]:
        torch.manual_seed(42)
        model = deep(std)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        first_norm = None
        for step in range(100):
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr), ytr)
            loss.backward()
            total = nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip if clip else float("inf"))
            if first_norm is None:
                first_norm = total.item()
            opt.step()
        L, acc = evaluate(model, Xtr, ytr)
        print(f"{name:21s}: initial grad norm = {first_norm:9.3f}, CE after 100 steps = {L:.4f}, train accuracy = {acc:.4f}")


# ---------------------------------------------------------------- 27
def inverted_dropout(a, p, rng):
    """p --- o'chirish ehtimolligi (PyTorch konvensiyasi), q = 1 - p."""
    q = 1 - p
    mask = (rng.random(a.shape) < q).astype(a.dtype)
    return a * mask / q, mask


def inverted_dropout_backward(grad_out, mask, p):
    return grad_out * mask / (1 - p)


def task_27():
    header(27, "inverted dropout: NumPy vs nn.Dropout")
    rng = np.random.default_rng(42)
    a = rng.normal(2.0, 1.0, size=(1000, 100))
    out, mask = inverted_dropout(a, 0.2, rng)
    g = inverted_dropout_backward(np.ones_like(a), mask, 0.2)
    print(f"dropped fraction = {1 - mask.mean():.4f}, mean(out)/mean(a) = {out.mean() / a.mean():.4f}")
    print("backward values:", np.unique(g))
    torch.manual_seed(42)
    drop = nn.Dropout(p=0.2)
    x = torch.ones(1000, 100, requires_grad=True)
    y = drop(x)
    y.sum().backward()
    print(f"nn.Dropout train: zero fraction = {(y == 0).float().mean().item():.4f}, "
          f"nonzero values = {torch.unique(y[y != 0]).tolist()}, grad values = {torch.unique(x.grad).tolist()}")
    drop.eval()
    print("nn.Dropout eval: identity =", torch.equal(drop(x), x))


# ---------------------------------------------------------------- 28
def task_28():
    header(28, "mini-batch gradient noise ~ 1/sqrt(B)")
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    Xtr = StandardScaler().fit_transform(Xtr)
    Xt = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    G = (0.5 - ytr)[:, None] * Xt                       # w = 0: p = 0.5, g_k = (p - y) x_k
    g_full = G.mean(0)
    rng = np.random.default_rng(42)
    for B in (8, 32, 128):
        est = np.array([G[rng.integers(0, len(G), B)].mean(0) for _ in range(2000)])
        sd = est.std(0).mean()
        bias = np.abs(est.mean(0) - g_full).max()
        print(f"B = {B:3d}: mean std = {sd:.4f}, std * sqrt(B) = {sd * np.sqrt(B):.4f}, max|mean - full| = {bias:.4f}")


# ---------------------------------------------------------------- 29
def rosen(t):
    return (1 - t[0]) ** 2 + 100 * (t[1] - t[0] ** 2) ** 2


def rosen_grad(t):
    return np.array([-2 * (1 - t[0]) - 400 * t[0] * (t[1] - t[0] ** 2), 200 * (t[1] - t[0] ** 2)])


def run_numpy(method, eta, steps=500, mu=0.9, b1=0.9, b2=0.999, eps=1e-8):
    th = np.array([-1.2, 1.0])
    v, m, s = np.zeros(2), np.zeros(2), np.zeros(2)
    for t in range(1, steps + 1):
        g = rosen_grad(th)
        if method == "sgd":
            th = th - eta * g
        elif method == "momentum":
            v = mu * v + g
            th = th - eta * v
        else:
            m = b1 * m + (1 - b1) * g
            s = b2 * s + (1 - b2) * g * g
            mh, sh = m / (1 - b1 ** t), s / (1 - b2 ** t)
            th = th - eta * mh / (np.sqrt(sh) + eps)
    return th


def run_torch(method, eta, steps=500):
    th = torch.tensor([-1.2, 1.0], dtype=torch.float64, requires_grad=True)
    opt = {"sgd": lambda: torch.optim.SGD([th], lr=eta),
           "momentum": lambda: torch.optim.SGD([th], lr=eta, momentum=0.9),
           "adam": lambda: torch.optim.Adam([th], lr=eta)}[method]()
    for _ in range(steps):
        opt.zero_grad()
        f = (1 - th[0]) ** 2 + 100 * (th[1] - th[0] ** 2) ** 2
        f.backward()
        opt.step()
    return th.detach().numpy()


def task_29():
    header(29, "SGD / Momentum / Adam from scratch vs torch.optim (Rosenbrock, 500 steps)")
    for method, eta in [("sgd", 3e-3), ("momentum", 3e-3), ("adam", 1e-1)]:
        a, b = run_numpy(method, eta), run_torch(method, eta)
        print(f"{method:8s} eta={eta:g}: theta = ({a[0]:.4f}, {a[1]:.4f}), f = {rosen(a):.2e}, "
              f"matches torch: {np.abs(a - b).max() < 1e-10}")


# ---------------------------------------------------------------- 30
def warmup_cosine(t, T_w, T):
    """Ko'paytuvchi eta_t / eta_max: t = 0, 1, ..., T-1 (epoxalar)."""
    if t < T_w:
        return (t + 1) / T_w
    return 0.5 * (1 + math.cos(math.pi * (t - T_w) / (T - T_w)))


def task_30():
    header(30, "use case: breast cancer MLP, warmup+cosine, early stopping")
    X, y = load_breast_cancer(return_X_y=True)
    Xa, Xte, ya, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    Xtr, Xva, ytr, yva = train_test_split(Xa, ya, test_size=0.25, stratify=ya, random_state=42)
    sc = StandardScaler().fit(Xtr)
    f = lambda a: torch.tensor(sc.transform(a), dtype=torch.float32)
    Xtr_t, Xva_t, Xte_t = f(Xtr), f(Xva), f(Xte)
    ytr_t, yva_t = torch.tensor(ytr), torch.tensor(yva)
    print(f"sizes: train {len(Xtr)}, val {len(Xva)}, test {len(Xte)}")
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(30, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
    T, T_w, patience = 100, 5, 10
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: warmup_cosine(t, T_w, T))
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(42))
    best, best_state, best_epoch, wait = float("inf"), None, -1, 0
    for epoch in range(T):
        lr_now = opt.param_groups[0]["lr"]
        if epoch in (0, 4, 5, 10):
            print(f"epoch {epoch:2d}: lr = {lr_now:.6f}")
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        sched.step()
        model.eval()
        vloss, _ = evaluate(model, Xva_t, yva_t)
        if vloss < best - 1e-4:
            best, best_state, best_epoch, wait = vloss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= patience:
                print(f"early stop at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(1).numpy()
    print(f"best epoch = {best_epoch}, best val CE = {best:.4f}")
    print(f"test accuracy = {np.mean(pred == yte):.4f}, F1 (malignant = 0) = {f1_score(yte, pred, pos_label=0):.4f}")


def main():
    t0 = time.perf_counter()
    for k in range(17, 31):
        globals()[f"task_{k}"]()
    print(f"\n[total runtime {time.perf_counter() - t0:.1f} s]", file=sys.stderr)


if __name__ == "__main__":
    main()
