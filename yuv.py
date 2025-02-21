import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the color image
image = cv2.imread('one.jpg')

# Check if the image is loaded successfully
if image is None:
    print("Error: Could not load image. Please check the file path.")
    exit()

# Convert the image from BGR to YUV color space
yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

# Split the YUV image into its channels
y, u, v = cv2.split(yuv_image)

# Apply histogram equalization to the Y (luminance) channel
y_eq = cv2.equalizeHist(y)

# Merge the channels back into a YUV image
yuv_image_eq = cv2.merge((y_eq, u, v))

# Convert the YUV image back to BGR color space
image_eq = cv2.cvtColor(yuv_image_eq, cv2.COLOR_YUV2BGR)

# Display the original and enhanced images
# Display the original and enhanced images
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