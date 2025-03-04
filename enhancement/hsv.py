import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the color image
image = cv2.imread('../one.jpg')

# Convert the image from BGR to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Split the HSV image into its channels
h, s, v = cv2.split(hsv_image)

# Apply histogram equalization to the V (Value) channel
v_eq = cv2.equalizeHist(v)

# Merge the channels back into an HSV image
hsv_image_eq = cv2.merge((h, s, v_eq))

# Convert the HSV image back to BGR color space
image_eq = cv2.cvtColor(hsv_image_eq, cv2.COLOR_HSV2BGR)

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