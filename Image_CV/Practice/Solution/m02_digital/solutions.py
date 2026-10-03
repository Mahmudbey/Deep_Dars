"""m02_digital -- Raqamli tasvir: amaliy mashg'ulot yechimlari (17--30-topshiriqlar).

Ishga tushirish:  python solutions.py        (Python 3.11/3.12, requirements.txt)
Deterministik (seed 42), faqat CPU, internet talab qilinmaydi. Rasmlar ./out/ ga yoziladi.
"""
from pathlib import Path

import numpy as np
import cv2
import torch
import skimage.data
from scipy import ndimage
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
np.set_printoptions(suppress=True)


def small(x):
    """Mashina aniqligidagi farqlarni platformaga bog'liq bo'lmagan ko'rinishda chiqarish."""
    return "<1e-9" if x < 1e-9 else f"{x:.1e}"


# ---------------------------------------------------------------- 17
def sensor_samples(mu, sigma_r, n, rng, g=1.0):
    """Puasson (foton) + normal (o'qish) shovqini: g*Poisson(mu) + N(0, sigma_r^2)."""
    return g * rng.poisson(mu, n) + rng.normal(0.0, sigma_r, n)


def task17():
    rng = np.random.default_rng(42)
    print(" mu    var_meas  mu+sr^2  SNR_meas  SNR_theory  sqrt(mu)")
    for mu in (5, 50, 500, 5000):
        x = sensor_samples(mu, 4.0, 100_000, rng)
        print(f"{mu:5d} {x.var():9.2f} {mu + 16:8d} {x.mean() / x.std():9.3f} "
              f"{mu / np.sqrt(mu + 16):11.3f} {np.sqrt(mu):9.3f}")


# ---------------------------------------------------------------- 18
def photon_transfer(g, sigma_r, mus, rng, shape=(200, 200)):
    """Har bir yoritilganlik uchun ikki tekis kadr; farq kadri orqali dispersiya (PTC)."""
    means, variances = [], []
    for mu in mus:
        a = g * rng.poisson(mu, shape) + g * rng.normal(0, sigma_r, shape)
        b = g * rng.poisson(mu, shape) + g * rng.normal(0, sigma_r, shape)
        means.append(0.5 * (a.mean() + b.mean()))
        variances.append((a - b).var() / 2)          # tekis bo'lmagan yoritishni yo'qotadi
    slope, icpt = np.polyfit(means, variances, 1)
    return np.array(means), np.array(variances), slope, np.sqrt(max(icpt, 0)) / slope


def task18():
    rng = np.random.default_rng(42)
    m, v, gain, sr = photon_transfer(0.5, 3.0, [20, 50, 100, 200, 400, 800, 1600], rng)
    print("mean_DN =", np.round(m, 1).tolist())
    print("var_DN  =", np.round(v, 1).tolist())
    print(f"estimated gain = {gain:.4f} DN/e (true 0.5), read noise = {sr:.2f} e (true 3.0), "
          f"electrons at 4095 DN = {4095 / gain:.0f} e")


# ---------------------------------------------------------------- 19
def dominant_freq(row):
    """1D signalning (DC siz) eng kuchli chastotasi (sikl/px) va amplitudasi."""
    x = row - row.mean()
    F = np.fft.rfft(x)
    k = np.argmax(np.abs(F))
    return k / len(x), 2 * np.abs(F[k]) / len(x)


def task19():
    n = np.arange(512)
    img = (127.5 + 100 * np.cos(2 * np.pi * 0.3125 * n))[None, :].repeat(512, 0)
    f0, a0 = dominant_freq(img[0])
    sl = img[::4, ::4]
    ar = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    f1, a1 = dominant_freq(sl[0])
    f2, a2 = dominant_freq(ar[0])
    theory = abs(np.exp(2j * np.pi * 0.3125 * np.arange(4)).sum()) / 4 * 100
    print(f"original: f={f0:.4f} A={a0:.2f};  slicing ::4: f={f1:.4f} A={a1:.2f};  "
          f"INTER_AREA: f={f2:.4f} A={a2:.2f} (theory {theory:.2f})")
    t = np.arange(20)
    print("cos(2pi*0.7n) == cos(2pi*0.3n) on samples:",
          bool(np.allclose(np.cos(2 * np.pi * 0.7 * t), np.cos(2 * np.pi * 0.3 * t))))


# ---------------------------------------------------------------- 20
def bayer_rggb(rgb):
    """RGB -> RGGB mozaika (bir kanalli uint8)."""
    m = np.zeros(rgb.shape[:2], rgb.dtype)
    m[0::2, 0::2] = rgb[0::2, 0::2, 0]
    m[0::2, 1::2] = rgb[0::2, 1::2, 1]
    m[1::2, 0::2] = rgb[1::2, 0::2, 1]
    m[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return m


def demosaic_bilinear(m):
    """Bilinear demozaiklash: niqoblangan kanallarni yadro bilan normallashtirilgan o'rtachalash."""
    h, w = m.shape
    R = np.zeros((h, w)); G = np.zeros((h, w)); B = np.zeros((h, w))
    R[0::2, 0::2] = 1; B[1::2, 1::2] = 1; G[0::2, 1::2] = 1; G[1::2, 0::2] = 1
    kRB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4.0
    kG = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4.0
    f = m.astype(np.float64)
    out = [ndimage.convolve(f * M, k, mode="mirror") for M, k in ((R, kRB), (G, kG), (B, kRB))]
    return np.clip(np.round(np.stack(out, -1)), 0, 255).astype(np.uint8)


def task20():
    rgb = skimage.data.astronaut()
    m = bayer_rggb(rgb)
    good = cv2.cvtColor(m, cv2.COLOR_BayerBG2RGB)
    bad = cv2.cvtColor(m, cv2.COLOR_BayerRG2RGB)
    own = demosaic_bilinear(m)
    c = (slice(2, -2), slice(2, -2))
    diff = np.abs(own[c].astype(int) - good[c]).max()
    print(f"measured fraction = {m.size / rgb.size:.4f};  PSNR BayerBG2RGB = {psnr(rgb, good):.2f} dB, "
          f"BayerRG2RGB (wrong) = {psnr(rgb, bad):.2f} dB, own bilinear = {psnr(rgb, own):.2f} dB")
    print(f"max |own - OpenCV| (interior) = {diff}")
    err = np.abs(good.astype(int) - rgb).sum(-1)
    y, x = np.unravel_index(err.argmax(), err.shape)
    print(f"worst pixel (y,x)=({y},{x}) error sum = {err.max()}")
    cv2.imwrite(str(OUT / "t20_demosaic_crop.png"),
                cv2.cvtColor(np.hstack([rgb[180:308, 150:278], good[180:308, 150:278]]),
                             cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------- 21
def quantize(img, n):
    """n bitli kvantlash, interval o'rtasiga: Q = q*floor(x/q) + q/2 (q = 256/2^n)."""
    q = 256 // 2 ** n
    return ((img // q) * q + q // 2).astype(np.uint8)


def task21():
    g = skimage.data.camera()
    print(" n   q  var(e)   q^2/12   mean(e)  PSNR   6.02n+10.76")
    for n in (1, 2, 3, 4, 5, 6, 7):
        q = 256 // 2 ** n
        e = quantize(g, n).astype(float) - g
        print(f"{n:2d} {q:3d} {e.var():8.2f} {q * q / 12:8.2f} {e.mean():7.2f} "
              f"{psnr(g, quantize(g, n)):6.2f} {20 * np.log10(2) * n + 10 * np.log10(12 * 255**2 / 2**16):6.2f}")


# ---------------------------------------------------------------- 22
def srgb_encode(L):
    L = np.asarray(L, dtype=np.float64)
    return np.where(L <= 0.0031308, 12.92 * L, 1.055 * np.power(L, 1 / 2.4) - 0.055)


def srgb_decode(V):
    V = np.asarray(V, dtype=np.float64)
    return np.where(V <= 0.04045, V / 12.92, np.power((V + 0.055) / 1.055, 2.4))


def task22():
    codes = np.arange(256)
    rt = np.round(255 * srgb_encode(srgb_decode(codes / 255)))
    print(f"round-trip 0..255 max error = {int(np.abs(rt - codes).max())}")
    L = np.linspace(0, 0.1, 100001)
    lin = np.unique(np.round(255 * L)).size
    sr = np.unique(np.round(255 * srgb_encode(L))).size
    print(f"codes for L in [0,0.1]: linear {lin}, sRGB {sr} (x{sr / lin:.2f})")
    chk = np.indices((256, 256)).sum(0) % 2 * 255            # 1 px shaxmat: 0/255
    naive = cv2.resize(chk.astype(np.float64), (128, 128), interpolation=cv2.INTER_AREA)
    lin_small = cv2.resize(srgb_decode(chk / 255.0), (128, 128), interpolation=cv2.INTER_AREA)
    correct = 255 * srgb_encode(lin_small)
    print(f"2x downscale of 0/255 checker: naive {naive.mean():.2f}, gamma-correct {correct.mean():.2f}")


# ---------------------------------------------------------------- 23
def task23():
    img = cv2.cvtColor(skimage.data.astronaut(), cv2.COLOR_RGB2BGR)
    i, j, c = 100, 200, 1
    off = i * img.strides[0] + j * img.strides[1] + c * img.strides[2]
    chw = img.transpose(2, 0, 1)
    f32 = img.astype(np.float32)
    t = torch.from_numpy(np.ascontiguousarray(chw))
    print(f"shape {img.shape} strides {img.strides}; offset[100,200,1] = {off}, "
          f"check {bool(img.ravel()[off] == img[i, j, c])}")
    print(f"CHW strides {chw.strides}, shares_memory {np.shares_memory(img, chw)}, "
          f"C_CONTIGUOUS {chw.flags['C_CONTIGUOUS']}; float32 strides {f32.strides}, "
          f"nbytes {img.nbytes} -> {f32.nbytes}")
    print(f"torch tensor shape {tuple(t.shape)} stride {t.stride()}; "
          f"10000 x 224x224x3: uint8 {10000 * 224 * 224 * 3 / 1e9:.3f} GB, "
          f"float32 {4 * 10000 * 224 * 224 * 3 / 1e9:.3f} GB")


# ---------------------------------------------------------------- 24
def to_uint8(x):
    """To'g'ri saqlash: clip -> round -> uint8."""
    return np.round(np.clip(x, 0, 255)).astype(np.uint8)


def task24():
    a = np.array([[230, 15, 128]], np.uint8); b = np.array([[40, 20, 200]], np.uint8)
    print(f"numpy a+b {(a + b).ravel().tolist()}, cv2.add {cv2.add(a, b).ravel().tolist()}, "
          f"numpy a-b {(a - b).ravel().tolist()}, cv2.subtract {cv2.subtract(a, b).ravel().tolist()}")
    img = skimage.data.astronaut().astype(np.float64)
    y = img * 1.8
    wrap = y.astype(np.int64) % 256                           # uint8 toshishi modeli
    print(f"x1.8: >255 count {(y > 255).sum()}, mean orig {img.mean():.2f}, "
          f"wrap {wrap.mean():.2f}, clip {to_uint8(y).mean():.2f}")
    z = skimage.data.camera() * 0.7
    print(f"x0.7: exact mean {z.mean():.4f}, astype {z.astype(np.uint8).mean():.4f}, "
          f"round {to_uint8(z).mean():.4f}")


# ---------------------------------------------------------------- 25
def rgb_to_ycrcb_hsv(r, g, b):
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    Cr = 0.713 * (r - Y) + 128; Cb = 0.564 * (b - Y) + 128
    M, m = max(r, g, b), min(r, g, b)
    if M == m:
        H = 0.0
    elif M == r:
        H = (60 * (g - b) / (M - m)) % 360
    elif M == g:
        H = 60 * (2 + (b - r) / (M - m))
    else:
        H = 60 * (4 + (r - g) / (M - m))
    S = 0 if M == 0 else (M - m) / M
    return (Y, Cr, Cb), (H, S, M)


def task25():
    for rgb in ((200, 100, 50), (50, 120, 200)):
        (Y, Cr, Cb), (H, S, V) = rgb_to_ycrcb_hsv(*rgb)
        px = np.array([[rgb[::-1]]], np.uint8)                # BGR
        print(f"RGB{rgb}: own YCrCb=({Y:.2f},{Cr:.2f},{Cb:.2f}) cv2="
              f"{cv2.cvtColor(px, cv2.COLOR_BGR2YCrCb).ravel().tolist()}; own HSV=({H:.2f},{S:.4f},{V}) "
              f"cv2={cv2.cvtColor(px, cv2.COLOR_BGR2HSV).ravel().tolist()}")
    img = cv2.cvtColor(skimage.data.astronaut(), cv2.COLOR_RGB2BGR)
    res = []
    for name, fw, bw in (("YCrCb", cv2.COLOR_BGR2YCrCb, cv2.COLOR_YCrCb2BGR),
                         ("HSV", cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR),
                         ("Lab", cv2.COLOR_BGR2Lab, cv2.COLOR_Lab2BGR)):
        e = np.abs(cv2.cvtColor(cv2.cvtColor(img, fw), bw).astype(int) - img)
        res.append(f"{name} max {e.max()} mean {e.mean():.3f}")
    print("round-trip uint8:", "; ".join(res))


# ---------------------------------------------------------------- 26
def circular_mean_deg(h_deg):
    a = np.radians(h_deg)
    return np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360


def task26():
    bgr = cv2.cvtColor(skimage.data.astronaut(), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 120, 70), (179, 255, 255))
    mask = cv2.morphologyEx(m1 | m2, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, w, h, area = stats[k]
    hdeg = 2.0 * hsv[..., 0][mask > 0]
    print(f"red pixels {int((mask > 0).sum())}, components {n - 1}, largest area {area} "
          f"bbox (x,y,w,h)=({x},{y},{w},{h})")
    print(f"hue of red pixels: naive mean {hdeg.mean():.2f} deg, circular mean "
          f"{circular_mean_deg(hdeg):.2f} deg")
    vis = bgr.copy(); vis[mask > 0] = (0, 255, 0)
    cv2.imwrite(str(OUT / "t26_red_mask.png"), vis)


# ---------------------------------------------------------------- 27
def task27():
    rgb = skimage.data.astronaut()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    print(f"raw {bgr.nbytes} B; PNG {buf.size} B (x{bgr.nbytes / buf.size:.2f}), lossless "
          f"{np.array_equal(cv2.imdecode(buf, cv2.IMREAD_COLOR), bgr)}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    for q in (95, 80, 50, 25, 10):
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        s = ssim(gray, cv2.cvtColor(dec, cv2.COLOR_BGR2GRAY), data_range=255)
        print(f"JPEG q={q:2d}: {buf.size:6d} B  x{bgr.nbytes / buf.size:5.1f}  "
              f"PSNR {psnr(bgr, dec):.2f} dB  SSIM(gray) {s:.4f}")


# ---------------------------------------------------------------- 28
def task28():
    g = skimage.data.camera()
    mask = np.digitize(g, [50, 100, 150, 200]).astype(np.uint8)      # sinflar 0..4
    res = {}
    for ext, par in ((".png", []), (".jpg", [cv2.IMWRITE_JPEG_QUALITY, 90])):
        ok, buf = cv2.imencode(ext, mask, par)
        back = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        res[ext] = back
        print(f"{ext}: changed labels {(back != mask).mean() * 100:.2f}%, "
              f"unique values {np.unique(back).tolist()}")
    small_nn = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    small_lin = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_LINEAR)
    print(f"resize: INTER_NEAREST values {np.unique(small_nn).tolist()}, "
          f"INTER_LINEAR != NEAREST pixels {(small_lin != small_nn).mean() * 100:.2f}%")


# ---------------------------------------------------------------- 29
def subsample_channels(bgr, idx):
    """YCrCb dagi berilgan kanallarni 4:2:0 kabi 2x kichraytirib, qayta kattalashtirish."""
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    h, w = y.shape[:2]
    for c in idx:
        s = cv2.resize(y[..., c], (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        y[..., c] = cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(y, cv2.COLOR_YCrCb2BGR)


def task29():
    bgr = cv2.cvtColor(skimage.data.astronaut(), cv2.COLOR_RGB2BGR)
    for name, idx in (("chroma (Cr,Cb)", (1, 2)), ("luma (Y)", (0,))):
        print(f"subsample {name}: PSNR {psnr(bgr, subsample_channels(bgr, idx)):.2f} dB")
    print(f"frame 1920x1080 bytes: 4:4:4 {1920 * 1080 * 3}, 4:2:0 {1920 * 1080 * 3 // 2}")


# ---------------------------------------------------------------- 30
QY = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
               [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
               [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
               [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], float)


def dct_matrix(n=8):
    """Ortonormal DCT-II matritsasi D: F = D f D^T."""
    k = np.arange(n)[:, None]; x = np.arange(n)[None, :]
    D = np.sqrt(2 / n) * np.cos((2 * x + 1) * k * np.pi / (2 * n))
    D[0] /= np.sqrt(2)
    return D


def jpeg_like(gray, scale=1.0):
    """8x8 bloklar: -128, DCT, round(F/(scale*QY)), teskari. (rekonstruksiya, nol ulushi)."""
    D = dct_matrix()
    h, w = gray.shape
    f = gray.astype(np.float64).reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3) - 128
    F = D @ f @ D.T
    Fq = np.round(F / (scale * QY))
    rec = D.T @ (Fq * scale * QY) @ D + 128
    rec = rec.transpose(0, 2, 1, 3).reshape(h, w)
    return to_uint8(rec), (Fq == 0).mean()


def task30():
    D = dct_matrix()
    blk = np.full((8, 8), 150.0) - 128
    print(f"D orthonormal {np.allclose(D @ D.T, np.eye(8))}; |own-cv2.dct| = "
          f"{small(np.abs(D @ blk @ D.T - cv2.dct(blk)).max())}; F(0,0) of const 150 = {(D @ blk @ D.T)[0, 0]:.2f}")
    g = skimage.data.camera()
    for s in (0.5, 1.0, 2.0, 4.0):
        rec, z = jpeg_like(g, s)
        print(f"scale {s:3.1f}: zero coeffs {z * 100:5.2f}%  PSNR {psnr(g, rec):.2f} dB  "
              f"SSIM {ssim(g, rec, data_range=255):.4f}")
    cv2.imwrite(str(OUT / "t30_jpeg_like_s4.png"), np.hstack([g, jpeg_like(g, 4.0)[0]]))


def main():
    for k in range(17, 31):
        print(f"--- task {k} ---")
        globals()[f"task{k}"]()


if __name__ == "__main__":
    main()
