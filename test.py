import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the image using OpenCV
image_path = 'test.jpg'
image = cv2.imread(image_path)

# Convert the image from BGR to RGB format
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to a PIL Image
image_pil = Image.fromarray(image_rgb)

# Convert the PIL Image to a PyTorch tensor
transform = transforms.ToTensor()
image_tensor = transform(image_pil)

# Define the ROI (x, y, width, height) for the photo on the ID card
x, y, width, height = 59, 237, 250, 300  # Adjusted coordinates

# Extract the ROI
roi = image_tensor[:, y:y+height, x:x+width]

# Convert the extracted ROI back to an image
roi_image = transforms.ToPILImage()(roi)

# Save or display the ROI image
roi_image_path = 'extracted_image.jpg'  # Path to save the extracted image
roi_image.save(roi_image_path)
roi_image.show()
