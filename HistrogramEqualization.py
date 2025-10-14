import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# Step 1: Load the grayscale image
image = cv2.imread('/content/Screenshot 2025-01-22 004500.png', cv2.IMREAD_GRAYSCALE)

# Step 2: Display the image
cv2_imshow(image)

# Step 3: Compute the histogram
# Parameters: [image], [channel=0], mask=None, [bins=256], [range=0-256]
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# Step 4: Plot the histogram
plt.figure(figsize=(8, 4))
plt.plot(histogram, color='black')
plt.title('Grayscale Image Histogram')
plt.xlabel('Pixel Intensity (0–255)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
