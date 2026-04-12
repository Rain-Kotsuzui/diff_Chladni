import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_target_image_smooth(image_path: str, N: int, sigma: float = 3.0) -> np.ndarray:

    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"image not found: {image_path}")
    
    img_resized = cv2.resize(img_gray, (N, N), interpolation=cv2.INTER_AREA)

    _, binary = cv2.threshold(img_resized, 128, 255, cv2.THRESH_BINARY)

    dist_field = cv2.distanceTransform(binary, distanceType=cv2.DIST_L2, maskSize=5)

    target_pattern = np.exp(-(dist_field / sigma)**2)

    # target_pattern = np.flipud(target_pattern)

    return target_pattern.astype(np.float32)


if __name__ == "__main__":
    N = 128
    
    # img_test = np.ones((500, 500), dtype=np.uint8) * 255
    # cv2.circle(img_test, (250, 250), 150, 0, 5) 
    # cv2.line(img_test, (100, 100), (400, 400), 0, 5)
    # cv2.imwrite("temp_target.jpg", img_test)

    
    plt.figure(figsize=(5, 5), facecolor='black')

    target_pattern = process_target_image_smooth("target.jpg", N, sigma=1.0)
    
    img_to_save = (target_pattern * 255).astype(np.uint8)
    cv2.imwrite("target_processed.jpg", img_to_save)


    ax = plt.imshow(target_pattern, cmap='viridis', interpolation='bilinear')

    plt.title("Target Pattern", color='white')
    plt.axis('off')

    plt.colorbar(ax, fraction=0.046, pad=0.04)
    
    plt.show()