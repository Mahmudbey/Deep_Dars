import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from collections import deque


def scene():
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(img, (40, 80), (120, 160), 150, -1)
    cv2.circle(img, (180, 180), 30, 250, -1)
    cv2.circle(img, (50, 220), 5, 255, -1)
    cv2.circle(img, (220, 40), 8, 200, -1)
    return img

def panel(name, outputs, titles):
    n = len(outputs)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))
    fig.canvas.manager.set_window_title(name)
    for ax, out, title in zip(axes, outputs, titles):
        ax.imshow(out, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def save(name, metrics):
    print(f"\n--- Natijalar: {name} ---")
    print(json.dumps(metrics, indent=4))

def convolve(a, k):
    if k.shape[0] % 2 != 1 or k.shape[1] % 2 != 1:
        raise ValueError('Toq kernel talab etiladi')
    pad_width = ((k.shape[0]//2,)*2, (k.shape[1]//2,)*2)
    padded = np.pad(a.astype(float), pad_width, mode='reflect')
    windows = np.lib.stride_tricks.sliding_window_view(padded, k.shape)
    return np.einsum('ijkl,kl->ij', windows, k[::-1, ::-1])

def canny_numpy(a, low=50, high=100):
    x = np.arange(-2, 3)
    g = np.exp(-x*x/2.)
    g /= g.sum()
    smooth = convolve(a, np.outer(g, g))
    
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    gx = convolve(smooth, sx[::-1, ::-1])
    gy = convolve(smooth, sx.T[::-1, ::-1])
    mag = np.hypot(gx, gy)
    direction = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 180
    
    nms = np.zeros_like(mag)
    h, w = a.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            angle = direction[y, x]
            if angle < 22.5 or angle >= 157.5:
                d = (0, 1)
            elif angle < 67.5:
                d = (1, 1)
            elif angle < 112.5:
                d = (1, 0)
            else:
                d = (1, -1)
            
            dy, dx = d
            if mag[y, x] >= max(mag[y+dy, x+dx], mag[y-dy, x-dx]):
                nms[y, x] = mag[y, x]
                
    strong = nms >= high
    weak = nms >= low
    queue = deque(map(tuple, np.argwhere(strong)))
    
    while queue:
        y, x = queue.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y+dy, x+dx
                if 0 <= yy < h and 0 <= xx < w and weak[yy, xx] and not strong[yy, xx]:
                    strong[yy, xx] = True
                    queue.append((yy, xx))
                    
    return smooth, mag, nms, strong.astype(np.uint8)*255


def main():
    a = scene()
    
    smooth, mag, nms, edge = canny_numpy(a)
    
    ref = cv2.Canny(np.rint(smooth).astype(np.uint8), 50, 100, L2gradient=True)
    
    union = np.logical_or(edge, ref).sum()
    agreement = float(np.logical_and(edge, ref).sum()) / max(1, union)
    
    panel('m04', 
          [a, mag, nms, edge, ref], 
          ['Tasvir', 'Gradient', 'NMS', 'Gisterezis', 'OpenCV'])
    
    flat = canny_numpy(np.ones((20, 20))*100)[-1]
    assert flat.sum() == 0, "Bir jinsli maydonda chekka aniqlanmasligi kerak"
    
    save('m04', {
        'custom_edge_pixels': int((edge>0).sum()),
        'opencv_edge_pixels': int((ref>0).sum()),
        'edge_overlap_IoU': agreement,
        'note': 'Agreement between implementations, not accuracy against annotated edges.'
    })

if __name__ == '__main__':
    main()