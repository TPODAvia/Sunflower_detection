from torchvision import datasets, transforms
from torchvision.transforms import RandomResizedCrop, RandomRotation, ColorJitter
import torch
import cv2
import numpy as np
import os
import sys
import shutil

source_folder = r'D:\Coding_AI\Sunflower_detection\opencv_blog_content\p\\'
destination_folder = 'D:\\Coding_AI\\Sunflower_detection\\root\\p\\'

os.makedirs("D:\Coding_AI\Sunflower_detection\root\p", exist_ok=True)
for filename in os.listdir(source_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(source_folder, filename)
        symlink_path = os.path.join(destination_folder, filename)
        os.symlink(image_path, symlink_path)

# Define the transformations
transformations = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor
    # RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.333)),  # Random zooming
    RandomRotation((90, 90)),  # Random rotation
    # transforms.Lambda(lambda x: torch.clamp(x, min=0.0, max=0.2)),  # Image sharpening
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)  # Random coloring
])

# Load the dataset
dataset = datasets.ImageFolder(r"D:\Coding_AI\Sunflower_detection\root", transform=transformations)

# Create a new directory to save the augmented images
os.makedirs("D:\Coding_AI\Sunflower_detection\augmented_images", exist_ok=True)
for x in range(0,1):

    for i, (image, label) in enumerate(dataset):
        # Convert PyTorch tensor to numpy array
        image_np = image.numpy().transpose((1, 2, 0))

        # Convert numpy array to OpenCV array, and display the image
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # cv2.imshow(f"Image {i}", image_cv)

        cv2.imwrite(f"D:\Coding_AI\Sunflower_detection/augmented_images/cropped_{x}{i}.jpg", 255*image_cv)

        # Wait for key press to continue

        # sys.exit()
        cv2.waitKey(0)