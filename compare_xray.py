import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Load images
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    if img is None:
        raise ValueError(f"Image at {path} not found.")
    return img

# Resize images to same dimensions
def preprocess_images(img1, img2, size=(256, 256)):
    img1_resized = cv2.resize(img1, size)
    img2_resized = cv2.resize(img2, size)
    return img1_resized, img2_resized

# Compute metrics
def compute_metrics(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    similarity, _ = ssim(img1, img2, full=True)
    return mse, similarity

# Main function
if __name__ == "__main__":
    path1 = "xray1.png"  # Path to the first X-ray image
    path2 = "xray2.png"  # Path to the second X-ray image

    img1 = load_image(path1)
    img2 = load_image(path2)

    img1, img2 = preprocess_images(img1, img2)

    mse, similarity = compute_metrics(img1, img2)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Structural Similarity Index (SSIM): {similarity}")


