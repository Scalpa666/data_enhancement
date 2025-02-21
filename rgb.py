import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the color image
image = cv2.imread('one.jpg')

# Split the image into its color channels
b, g, r = cv2.split(image)

# Apply histogram equalization to each channel
b_eq = cv2.equalizeHist(b)
g_eq = cv2.equalizeHist(g)
r_eq = cv2.equalizeHist(r)

# Merge the equalized channels back into a single image
image_eq = cv2.merge((b_eq, g_eq, r_eq))

# Display the original and enhanced images
# cv2.imshow('Original Image', image)
# cv2.imshow('Contrast Enhanced Image', image_eq)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# 显示原始图像和对比度增强后的图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Contrast Enhanced Image")
plt.imshow(image_eq, cmap='gray')
plt.axis('off')

plt.show()