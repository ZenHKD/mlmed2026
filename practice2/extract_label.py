import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def label(img_path):
    # Extract contours
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fill
    H, W = mask.shape
    filled = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(filled, contours, -1, 255, -1)

    # Transparency
    alpha = (filled > 0).astype(np.float32)          # 0/1 mask
    alpha = alpha[..., np.newaxis]                   # expand to (H, W, 1) for broadcasting

    return alpha


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    label_path = os.path.join(script_dir, "data/training_set/001_HC_Annotation.png")
    img_path   = os.path.join(script_dir, "data/training_set/001_HC.png")

    alpha = label(label_path)          
    img    = cv2.imread(img_path)       
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Green overlay
    green = np.array([0, 255, 0], dtype=np.float32)  # RGB green
    
    # Blend: where mask is 1, mix 60% image + 40% green
    overlay = (1 - alpha) * img_rgb + alpha * (0.6 * img_rgb + 0.4 * green)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    plt.imshow(overlay)
    plt.show()