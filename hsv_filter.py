import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the color image
image = cv2.imread('one.jpg')

# Apply median filtering to remove noise
kernel_size = 3
filtered_image = cv2.medianBlur(image, kernel_size)

# Convert the image from BGR to HSV color space
hsv_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2HSV)

# Split the HSV image into its channels
h, s, v = cv2.split(hsv_image)

# Apply histogram equalization to the V (Value) channel
v_eq = cv2.equalizeHist(v)

# Merge the channels back into an HSV image
hsv_image_eq = cv2.merge((h, s, v_eq))

# Convert the HSV image back to BGR color space
image_eq = cv2.cvtColor(hsv_image_eq, cv2.COLOR_HSV2BGR)

# Display the original and enhanced images
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Median Filtered Image
plt.subplot(1, 3, 2)
plt.title("Median Filtered Image")
plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Contrast Enhanced Image
plt.subplot(1, 3, 3)
plt.title("Contrast Enhanced Image")
plt.imshow(cv2.cvtColor(image_eq, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()