"""m01_camera -- Kamera modeli: amaliy mashg'ulot yechimlari (17--30-topshiriqlar).

Ishga tushirish:  python solutions.py        (Python 3.11/3.12, requirements.txt)
Deterministik (seed 42), faqat CPU. Rasmlar ./out/ ga, yuklangan fayllar ./data/ ga yoziladi.
"""
from pathlib import Path
import urllib.request

import numpy as np
import cv2
import skimage.data

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DATA = HERE / "data"
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/"
LEFT = [f"left{i:02d}.jpg" for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14)]
PATTERN = (9, 6)                     # ichki burchaklar soni (ustun, qator)
K800 = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
np.set_printoptions(suppress=True)


def small(x):
    """Mashina aniqligidagi farqlarni platformaga bog'liq bo'lmagan ko'rinishda chiqarish."""
    return "<1e-9" if x < 1e-9 else f"{x:.1e}"


def fetch(name):
    """OpenCV sample faylini ./data/ ga yuklab, keshdan o'qiydi."""
    p = DATA / name
    if not p.exists():
        try:
            with urllib.request.urlopen(URL + name, timeout=30) as r:
                p.write_bytes(r.read())
        except Exception as e:
            raise RuntimeError(f"{URL + name} ni yuklab bo'lmadi ({e}). "
                               f"Faylni qo'lda {p} ga joylashtiring.") from None
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"{p} o'qilmadi (fayl buzilgan). O'chirib, qayta ishga tushiring.")
    return img


def rot_y(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


# ---------------------------------------------------------------- 17
def project_points(Xw, K, R, t):
    """(N,3) dunyo nuqtalari -> (N,2) piksel va Z_c>0 niqobi."""
    Xc = Xw @ R.T + t
    uvw = Xc @ K.T
    return uvw[:, :2] / uvw[:, 2:], Xc[:, 2] > 0


def task17():
    cube = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    rvec = np.array([0.3, -0.4, 0.1]); t = np.array([0.0, 0.0, 5.0])
    R = cv2.Rodrigues(rvec)[0]
    uv, front = project_points(cube, K800, R, t)
    uv_cv = cv2.projectPoints(cube, rvec, t, K800, None)[0].reshape(-1, 2)
    uv_s, _ = project_points(2.5 * cube, K800, R, 2.5 * t)
    print("uv =", np.round(uv, 2).tolist())
    print(f"all Z_c>0: {bool(front.all())}  max|own-cv2| = {small(np.abs(uv - uv_cv).max())}  "
          f"max|lambda=2.5 - lambda=1| = {small(np.abs(uv_s - uv).max())}")
    img = np.full((480, 640, 3), 255, np.uint8)
    for i in range(8):
        for j in range(i + 1, 8):
            if np.abs(cube[i] - cube[j]).sum() == 1.0:     # bitta o'q bo'yicha qo'shni
                cv2.line(img, tuple(np.round(uv[i]).astype(int)),
                         tuple(np.round(uv[j]).astype(int)), (120, 60, 20), 2, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "t17_cube.png"), img)


# ---------------------------------------------------------------- 18
def distort_normalized(x, d):
    """Brown--Conrady: (N,2) normallashtirilgan -> buzilgan. d=(k1,k2,p1,p2,k3)."""
    k1, k2, p1, p2, k3 = d
    X, Y = x[:, 0], x[:, 1]
    r2 = X * X + Y * Y
    rad = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = X * rad + 2 * p1 * X * Y + p2 * (r2 + 2 * X * X)
    yd = Y * rad + p1 * (r2 + 2 * Y * Y) + 2 * p2 * X * Y
    return np.stack([xd, yd], 1)


def task18():
    d = np.array([-0.28, 0.08, 0.001, -0.0005, 0.01])
    g = np.linspace(-0.5, 0.5, 5)
    x = np.array([[a, b] for b in g for a in g])
    uv = distort_normalized(x, d) @ K800[:2, :2].T + K800[:2, 2]
    Xw = np.c_[x, np.ones(len(x))]
    uv_cv = cv2.projectPoints(Xw, np.zeros(3), np.zeros(3), K800, d)[0].reshape(-1, 2)
    shift = np.linalg.norm(uv - (x @ K800[:2, :2].T + K800[:2, 2]), axis=1)
    print(f"max|own-cv2| = {small(np.abs(uv - uv_cv).max())} px; shift: centre {shift[12]:.3f} px, "
          f"max {shift.max():.3f} px at x={x[shift.argmax()].tolist()}")


# ---------------------------------------------------------------- 19
def undistort_iter(xd, d, n):
    """Qo'zg'almas nuqta iteratsiyasi (OpenCV undistortPoints algoritmi)."""
    k1, k2, p1, p2, k3 = d
    x = xd.copy()
    for _ in range(n):
        X, Y = x[:, 0], x[:, 1]
        r2 = X * X + Y * Y
        kap = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        dx = 2 * p1 * X * Y + p2 * (r2 + 2 * X * X)
        dy = p1 * (r2 + 2 * Y * Y) + 2 * p2 * X * Y
        x = np.stack([(xd[:, 0] - dx) / kap, (xd[:, 1] - dy) / kap], 1)
    return x


def task19():
    d = np.array([-0.3, 0.1, 0.0, 0.0, 0.0])
    xt = np.array([[0.5, 0.3]])
    xd = distort_normalized(xt, d)
    errs = [np.abs(undistort_iter(xd, d, n) - xt).max() for n in range(1, 11)]
    x_cv = cv2.undistortPoints(xd.reshape(1, 1, 2), np.eye(3), d).reshape(1, 2)
    print("xd =", np.round(xd[0], 6).tolist())
    print("err(n=1..10) =", " ".join(f"{e:.1e}" for e in errs))
    print(f"ratio err10/err9 = {errs[9] / errs[8]:.3f};  |own(n=5)-cv2| = "
          f"{small(np.abs(undistort_iter(xd, d, 5) - x_cv).max())}")


# ---------------------------------------------------------------- 20
def simulate_distortion(img, K, d):
    """Ideal tasvirdan buzilgan tasvir: har bir chiqish pikseli uchun ideal manba nuqtasi."""
    h, w = img.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    xd = np.stack([(u.ravel() - K[0, 2]) / K[0, 0], (v.ravel() - K[1, 2]) / K[1, 1]], 1)
    x = undistort_iter(xd, d, 20)
    mapx = (x[:, 0] * K[0, 0] + K[0, 2]).reshape(h, w).astype(np.float32)
    mapy = (x[:, 1] * K[1, 1] + K[1, 2]).reshape(h, w).astype(np.float32)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def task20():
    img = skimage.data.camera()
    K = np.array([[400.0, 0, 255.5], [0, 400.0, 255.5], [0, 0, 1]])
    d = np.array([-0.3, 0.08, 0.0, 0.0, 0.0])
    dist = simulate_distortion(img, K, d)
    back = cv2.undistort(dist, K, d)
    c = slice(156, 356)
    mae_d = np.abs(dist[c, c].astype(float) - img[c, c]).mean()
    mae_b = np.abs(back[c, c].astype(float) - img[c, c]).mean()
    valid = simulate_distortion(np.full_like(img, 255), K, d) > 127
    print(f"MAE centre 200x200: distorted {mae_d:.2f}, corrected {mae_b:.2f}; "
          f"valid fraction of distorted = {valid.mean():.4f}")
    cv2.imwrite(str(OUT / "t20_distorted.png"), np.hstack([img, dist, back]))


# ---------------------------------------------------------------- 21, 22
def synthetic_views(n, rng, K, d, sigma=0.25, square=0.03):
    obj = np.zeros((54, 3), np.float32)
    obj[:, :2] = (np.mgrid[0:9, 0:6].T.reshape(-1, 2) - [4, 2.5]) * square
    objs, imgs = [], []
    while len(objs) < n:
        rv = rng.uniform(-0.5, 0.5, 3)
        tv = np.r_[rng.uniform(-0.05, 0.05, 2), rng.uniform(0.4, 0.7)]
        p = cv2.projectPoints(obj, rv, tv, K, d)[0].reshape(-1, 2)
        p = p + rng.normal(0, sigma, p.shape)
        if p.min() >= 0 and p[:, 0].max() < 640 and p[:, 1].max() < 480:
            objs.append(obj); imgs.append(p.astype(np.float32))
    return objs, imgs


KTRUE = np.array([[700.0, 0, 318], [0, 705.0, 242], [0, 0, 1]])
DTRUE = np.array([-0.25, 0.07, 0.0, 0.0, 0.0])


def task21():
    rng = np.random.default_rng(42)
    objs, imgs = synthetic_views(15, rng, KTRUE, DTRUE)
    rms, K, d, _, _ = cv2.calibrateCamera(objs, imgs, (640, 480), None, None)
    est = K[[0, 1, 0, 1], [0, 1, 2, 2]]
    print(f"RMS = {rms:.3f} px (sigma*sqrt2 = {0.25 * np.sqrt(2):.3f})")
    print("fx fy cx cy =", np.round(est, 2).tolist(), " k1 k2 =", np.round(d[0, :2], 3).tolist())
    print("fx,fy error % =", np.round(100 * (est[:2] / [700, 705] - 1), 3).tolist())


def task22():
    for n in (3, 5, 10, 20):
        rng = np.random.default_rng(42)
        objs, imgs = synthetic_views(n, rng, KTRUE, DTRUE)
        rms, K, d, _, _ = cv2.calibrateCamera(objs, imgs, (640, 480), None, None)
        print(f"N={n:2d}: RMS={rms:.3f}  |dfx|={abs(K[0, 0] - 700):6.2f}  "
              f"|dcx|={abs(K[0, 2] - 318):5.2f}  k1={d[0, 0]:.3f}")


# ---------------------------------------------------------------- 23, 24, 25
def find_corners(gray):
    ok, c = cv2.findChessboardCorners(gray, PATTERN)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    return cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), crit)


def board_points(square=1.0):
    obj = np.zeros((54, 3), np.float32)
    obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * square
    return obj


def calibrate_real():
    names, objs, imgs = [], [], []
    for n in LEFT:
        c = find_corners(fetch(n))
        if c is not None:
            names.append(n); objs.append(board_points()); imgs.append(c)
    rms, K, d, rv, tv = cv2.calibrateCamera(objs, imgs, (640, 480), None, None)
    return dict(names=names, objs=objs, imgs=imgs, rms=rms, K=K, d=d, rv=rv, tv=tv)


def task23(cal):
    K, d = cal["K"], cal["d"]
    print(f"views found: {len(cal['names'])}/{len(LEFT)};  RMS = {cal['rms']:.4f} px")
    print("fx fy cx cy =", np.round(K[[0, 1, 0, 1], [0, 1, 2, 2]], 2).tolist())
    print("d =", np.round(d.ravel(), 4).tolist())


def task24(cal):
    errs = []
    for o, p, r, t in zip(cal["objs"], cal["imgs"], cal["rv"], cal["tv"]):
        q = cv2.projectPoints(o, r, t, cal["K"], cal["d"])[0]
        errs.append(np.sqrt(((p - q) ** 2).sum(-1).mean()))
    errs = np.array(errs)
    print("per-view RMS:", " ".join(f"{n[4:6]}:{e:.3f}" for n, e in zip(cal["names"], errs)))
    print(f"worst = {cal['names'][errs.argmax()]} ({errs.max():.3f});  "
          f"sqrt(mean(e^2)) = {np.sqrt((errs ** 2).mean()):.4f}")


def line_rms(pts):
    """Nuqtalarga to'g'ri chiziq (TLS) va perpendikulyar qoldiq RMS."""
    q = pts - pts.mean(0)
    s = np.linalg.svd(q, compute_uv=False)
    return s[-1] / np.sqrt(len(pts))


def task25(cal):
    K, d = cal["K"], cal["d"]
    img = fetch("left12.jpg")
    c = find_corners(img).reshape(6, 9, 2)
    cu = cv2.undistortPoints(c.reshape(-1, 1, 2), K, d, P=K).reshape(6, 9, 2)
    before = np.mean([line_rms(r) for r in c])
    after = np.mean([line_rms(r) for r in cu])
    newK, roi = cv2.getOptimalNewCameraMatrix(K, d, (640, 480), 0)
    und = cv2.undistort(img, K, d, None, newK)
    print(f"row straightness RMS: before {before:.3f} px, after {after:.3f} px;  roi(alpha=0) = "
          f"{[int(v) for v in roi]}")
    cv2.imwrite(str(OUT / "t25_left12_undist.png"), np.hstack([img, und]))


# ---------------------------------------------------------------- 26
def dlt_homography(src, dst):
    """Normallashtirilgan DLT (Hartley). src,dst: (N,2), N>=4. H[2,2]=1."""
    def norm(p):
        m = p.mean(0); s = np.sqrt(2) / np.linalg.norm(p - m, axis=1).mean()
        return np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]])
    T1, T2 = norm(src), norm(dst)
    a = (np.c_[src, np.ones(len(src))] @ T1.T)
    b = (np.c_[dst, np.ones(len(dst))] @ T2.T)
    A = []
    for (x, y, _), (u, v, _) in zip(a, b):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    Hn = np.linalg.svd(np.array(A))[2][-1].reshape(3, 3)
    H = np.linalg.inv(T2) @ Hn @ T1
    return H / H[2, 2]


def apply_h(H, p):
    q = np.c_[p, np.ones(len(p))] @ H.T
    return q[:, :2] / q[:, 2:]


def task26():
    img = fetch("left01.jpg")
    c = find_corners(img).reshape(-1, 2).astype(np.float64)
    src = board_points()[:, :2].astype(np.float64)
    H = dlt_homography(src, c)
    Hcv, _ = cv2.findHomography(src, c, 0)
    rms = lambda HH: np.sqrt(((apply_h(HH, src) - c) ** 2).sum(1).mean())
    print(f"||H_own-H_cv||/||H_cv|| = {np.linalg.norm(H - Hcv) / np.linalg.norm(Hcv):.1e};  RMS own {rms(H):.4f} px, "
          f"cv2 {rms(Hcv):.4f} px")
    T = np.array([[40.0, 0, 40], [0, 40.0, 40], [0, 0, 1]])        # 40 px/katak, 40 px chet
    top = cv2.warpPerspective(img, T @ np.linalg.inv(H), (400, 280))
    cv2.imwrite(str(OUT / "t26_left01_top.png"), top)
    print("H =", np.round(H, 4).tolist())


# ---------------------------------------------------------------- 27
def task27(cal):
    img = fetch("left01.jpg")
    c = find_corners(img)
    ok, rv, tv = cv2.solvePnP(board_points(), c, cal["K"], cal["d"])
    R = cv2.Rodrigues(rv)[0]
    C = -R.T @ tv
    q = cv2.projectPoints(board_points(), rv, tv, cal["K"], cal["d"])[0]
    e = np.sqrt(((q - c) ** 2).sum(-1).mean())
    print(f"tvec = {np.round(tv.ravel(), 3).tolist()}  |t| = {np.linalg.norm(tv):.3f} katak")
    print(f"C = {np.round(C.ravel(), 3).tolist()}  tilt(board normal vs axis) = "
          f"{np.degrees(np.arccos(abs(R[2, 2]))):.2f} deg  RMS = {e:.4f} px")


# ---------------------------------------------------------------- 28
def pixel_to_ground(u, v, K, h, pitch_deg):
    """Kamera h balandlikda, optik o'q pitch_deg pastga. (X, Z) metr yoki None (ufqdan yuqori)."""
    a = np.radians(pitch_deg)
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    r = R.T @ np.linalg.solve(K, [u, v, 1.0])
    if r[1] <= 1e-12:
        return None
    lam = h / r[1]
    return lam * r[0], lam * r[2]


def ground_to_pixel(X, Z, K, h, pitch_deg):
    a = np.radians(pitch_deg)
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    p = K @ (R @ np.array([X, h, Z]))
    return p[:2] / p[2]


def task28():
    K = np.array([[700.0, 0, 640], [0, 700.0, 360], [0, 0, 1]])
    rows, back_err = [], 0.0
    for v in (290, 360, 420, 500, 700):
        g = pixel_to_ground(800, v, K, 1.4, 5.0)
        if g is None:
            rows.append(f"v={v}: None")
        else:
            back = ground_to_pixel(*g, K, 1.4, 5.0)
            back_err = max(back_err, np.abs(back - [800, v]).max())
            rows.append(f"v={v}: X={g[0]:.3f} Z={g[1]:.3f}")
    print("; ".join(rows))
    print(f"horizon row v = {360 - 700 * np.tan(np.radians(5)):.2f};  round-trip err = {small(back_err)}")
    K2 = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    print(f"check (h=2, pitch=30, v=480): Z = {pixel_to_ground(320, 480, K2, 2.0, 30.0)[1]:.4f} m")


# ---------------------------------------------------------------- 29
def vanishing_point(p1, p2, q1, q2):
    """Ikki tasvir kesmasi (p1p2) va (q1q2) davomlarining kesishishi (bir jinsli)."""
    h = lambda p: np.array([p[0], p[1], 1.0])
    l1, l2 = np.cross(h(p1), h(p2)), np.cross(h(q1), h(q2))
    v = np.cross(l1, l2)
    return v[:2] / v[2]


def task29():
    R, t = rot_y(-20), np.array([0.5, 0.2, 1.0])
    for d in (np.array([1.0, 0, 1]), np.array([0.0, 0, 1])):
        A = np.array([[0, 1, 4.0], [0, 1, 4.0] + d])
        B = np.array([[1, -1, 3.0], [1, -1, 3.0] + d])
        a, _ = project_points(A, K800, R, t)
        b, _ = project_points(B, K800, R, t)
        vp = vanishing_point(a[0], a[1], b[0], b[1])
        kr = K800 @ R @ d
        print(f"d={d.tolist()}: VP lines = {np.round(vp, 2).tolist()},  KRd = "
              f"{np.round(kr[:2] / kr[2], 2).tolist()}")


# ---------------------------------------------------------------- 30
def task30():
    img = skimage.data.astronaut()[:, :, ::-1]
    K = np.array([[500.0, 0, 255.5], [0, 500.0, 255.5], [0, 0, 1]])
    H = K @ rot_y(8) @ np.linalg.inv(K)
    warped = cv2.warpPerspective(img, H, (512, 512))
    valid = cv2.warpPerspective(np.full((512, 512), 255, np.uint8), H, (512, 512))
    corners = np.array([[0, 0], [511, 0], [511, 511], [0, 511]], np.float32)
    Hest = cv2.getPerspectiveTransform(corners, apply_h(H, corners).astype(np.float32))
    Rest = np.linalg.inv(K) @ Hest @ K
    Rest /= np.cbrt(np.linalg.det(Rest))
    ang = np.degrees(np.linalg.norm(cv2.Rodrigues(Rest)[0]))
    pp = apply_h(H, np.array([[255.5, 255.5]]))[0]
    print(f"principal point -> {np.round(pp, 2).tolist()};  recovered angle = {ang:.4f} deg;  "
          f"valid fraction = {(valid > 127).mean():.4f}")
    cv2.imwrite(str(OUT / "t30_astronaut_rot.png"), warped)


def main():
    cal = None
    for k in range(17, 31):
        print(f"--- task {k} ---")
        f = globals()[f"task{k}"]
        if k in (23, 24, 25, 27):
            if cal is None:
                cal = calibrate_real()
            f(cal)
        else:
            f()


if __name__ == "__main__":
    main()
