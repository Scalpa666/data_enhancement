import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
image = cv2.imread('one.jpg', cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图像

# 检查图像是否成功加载
if image is None:
    print("Error: Could not load image. Please check the file path.")
    exit()

# 使用中值滤波去除噪声
filtered_image = cv2.medianBlur(image, 3)  # 滤波器大小为3x3

# 使用 OpenCV 的直方图均衡化方法
equalized_image = cv2.equalizeHist(filtered_image)  # 注意是对滤波后的图像进行直方图均衡化

# 显示原始图像、滤波后的图像和对比度增强后的图像
plt.figure(figsize=(15, 5))  # 调整窗口大小以容纳三个子图
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Median Filtered Image")
plt.imshow(filtered_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Contrast Enhanced Image")
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

plt.show()