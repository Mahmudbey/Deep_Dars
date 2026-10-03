import os
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import data
from keras.datasets import cifar10

# TensorFlow ogohlantirishlarini yashirish
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# CIFAR-10 va yuqori aniqlikdagi tasvirni yuklash
(_, _), (x_test, _) = cifar10.load_data()
img_low = cv2.cvtColor(x_test[0], cv2.COLOR_RGB2GRAY).astype(np.float32)
img_high = cv2.cvtColor(data.cat(), cv2.COLOR_RGB2GRAY).astype(np.float32)

rng = np.random.default_rng(0)

def process_image(img, label):
    print(f"\n--- {label} ({img.shape[0]}x{img.shape[1]}) ---")
    gauss = np.clip(img + rng.normal(0, 20, img.shape), 0, 255)
    salt_pepper = img.copy()
    mask = rng.random(img.shape) < 0.05
    salt_pepper[mask] = rng.choice([0, 255], size=mask.sum())

    for name, noisy in [("gauss", gauss), ("tuz-murch", salt_pepper)]:
        noisy8 = noisy.astype(np.uint8)
        for filt_name, out in [
            ("gauss", cv2.GaussianBlur(noisy8, (5, 5), 1.0)),
            ("median", cv2.medianBlur(noisy8, 5)),
            ("bilateral", cv2.bilateralFilter(noisy8, 5, 50, 50)),
        ]:
            p = psnr(img.astype(np.uint8), out, data_range=255)
            s = ssim(img.astype(np.uint8), out, data_range=255)
            print(f"{name:10s} {filt_name:10s} PSNR={p:.2f} SSIM={s:.3f}")

process_image(img_low, "CIFAR-10 (Kichik o'lcham)")
process_image(img_high, "Mushuk (Yuqori o'lcham - Real tekstura)")