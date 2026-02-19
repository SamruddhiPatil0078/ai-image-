import numpy as np
import cv2

def add_gaussian_noise(face_img, strength=8):
    """
    Adds small random noise to image

    strength: higher = more visible change
    """

    noise = np.random.normal(0, strength, face_img.shape).astype(np.float32)

    noisy = face_img.astype(np.float32) + noise

    # clip to valid image range
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return noisy
