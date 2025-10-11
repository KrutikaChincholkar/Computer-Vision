import cv2
from google.colab.patches import cv2_imshow

# Load the image
image_path = "Path_Of_image.jpg"
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Save the grayscale image
cv2.imwrite("grayscale_image.jpg", gray_image)
print("Grayscale image saved as 'grayscale_image.jpg'")

# Display the original and grayscale images
cv2_imshow(image)
cv2_imshow(gray_image)
