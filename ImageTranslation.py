import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load the image
image = cv2.imread('img.png')
cv2_imshow(image)

# Define translation distances
tx, ty = -50, -50  # Shift left by 50 and up by 50 pixels

# Create the translation matrix
m_trans = np.float32([
    [1, 0, tx],   # 1,0 for x-scaling; tx for x-translation
    [0, 1, ty]    # 0,1 for y-scaling; ty for y-translation
])

# Apply the affine transformation for translation
image_translation = cv2.warpAffine(image, m_trans, (image.shape[1], image.shape[0]))

# Display the translated image
cv2_imshow(image_translation)
