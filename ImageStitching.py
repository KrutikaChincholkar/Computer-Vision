import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Step 1: Read images
image1 = cv2.imread("/content/Screenshot 2025-02-25 155616.png")
image2 = cv2.imread("/content/Screenshot 2025-02-25 155818.png")

# Step 2: Check if images are loaded correctly
if image1 is None or image2 is None:
    print("Error: One or both image paths are incorrect.")
else:
    # Step 3: Resize both images to have the same height (for proper alignment)
    height = min(image1.shape[0], image2.shape[0])

    # ✅ Fix: both should use their own height ratio during resizing
    image1 = cv2.resize(image1, (int(image1.shape[1] * height / image1.shape[0]), height))
    image2 = cv2.resize(image2, (int(image2.shape[1] * height / image2.shape[0]), height))

    # Step 4: Concatenate images horizontally
    panorama = np.hstack((image1, image2))

    # Step 5: Display and save the concatenated result
    cv2_imshow(panorama)
    cv2.imwrite("panorama.jpg", panorama)
    print("Panorama image saved as 'panorama.jpg'")
