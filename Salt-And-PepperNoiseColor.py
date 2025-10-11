import random
import cv2
from google.colab.patches import cv2_imshow

def add_noise(img):
    # Get the image dimensions
    if len(img.shape) == 3:  # Color image
        row, col, ch = img.shape
    else:  # Grayscale image
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

# Load the image
img = cv2.imread('Path_of_Img.png')
cv2_imshow(img)

# Apply salt-and-pepper noise
noisy_img = add_noise(img.copy())

# Save the noisy image
cv2.imwrite('salt_and_pepper_image.png', noisy_img)
print("Image saved as 'salt_and_pepper_image.png'")

# Display the noisy image
cv2_imshow(noisy_img)
