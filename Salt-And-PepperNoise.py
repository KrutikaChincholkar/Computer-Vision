import random
import cv2
from google.colab.patches import cv2_imshow

def add_noise(img):
    # Get image dimensions
    row, col = img.shape

    # Add white pixels (salt)
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 255

    # Add black pixels (pepper)
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 0

    return img

# Read image in grayscale
img = cv2.imread("Path_Of_img.png", cv2.IMREAD_GRAYSCALE)
print("Original Image:")
cv2_imshow(img)

# Apply salt-and-pepper noise
noisy_img = add_noise(img.copy())

# Save and display
cv2.imwrite("salt_and_pepper_noise.png", noisy_img)
print("Path_Of_Img.png'")
cv2_imshow(noisy_img)
