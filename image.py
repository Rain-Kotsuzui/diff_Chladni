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

    target_pattern = 1-np.exp(-(dist_field / sigma)**2)

    # target_pattern = np.flipud(target_pattern)

    return target_pattern.astype(np.float32)


if __name__ == "__main__":
    N = 64
    
    img_test = np.ones((N, N), dtype=np.uint8) * 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "S"
    font_scale = 2  # 足够大的字号
    thickness = 2   # 增加厚度，防止在 64x64 分辨率下丢失
    
    # 3. 计算文字大小以便居中
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (N - text_size[0]) // 2
    text_y = (N + text_size[1]) // 2 # 注意 y 坐标在 OpenCV 里是基线
    
    # 4. 在图上绘制黑色字母 S (0, 0, 0)
    cv2.putText(img_test, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # img_test = np.ones((500, 500), dtype=np.uint8) * 255
    # cv2.circle(img_test, (250, 250), 150, 0, 20) 
    
    cv2.imwrite("target.jpg", img_test)

    
    plt.figure(figsize=(5, 5), facecolor='black')

    target_pattern = process_target_image_smooth("target.jpg", N, sigma=1.0)
    
    img_to_save = (target_pattern * 255).astype(np.uint8)
    cv2.imwrite("target_processed.jpg", img_to_save)


    ax = plt.imshow(target_pattern, cmap='viridis', interpolation='bilinear')

    plt.title("Target Pattern", color='white')
    plt.axis('off')

    plt.colorbar(ax, fraction=0.046, pad=0.04)
    
    plt.show()