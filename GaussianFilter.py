import cv2
import numpy as np
import os
from joblib import Memory

# Define the parameters for the Gaussian filter
kernel_size = (5, 5)
sigma_x = 0

# Define the list of directories containing the images
img_dirs = [r'D:\Project KLE\Chilli Cropped\chilli1\adulterated5',
           r'D:\Project KLE\Chilli Cropped\chilli1\adulterated10',
           r'D:\Project KLE\Chilli Cropped\chilli1\adulterated15',
           r'D:\Project KLE\Chilli Cropped\chilli1\adulterated20',
           r'D:\Project KLE\Chilli Cropped\chilli1\adulterated25',
           r'D:\Project KLE\Chilli Cropped\chilli1\adulterated30',
           r'D:\Project KLE\Chilli Cropped\chilli1\adulterated35',
           r'D:\Project KLE\Chilli Cropped\chilli1\adulterated40',
           r'D:\Project KLE\Chilli Cropped\chilli1\adulterated45',
           r'D:\Project KLE\Chilli Cropped\chilli1\adulterated50',
           r'D:\Project KLE\Chilli Cropped\chilli1\adulterated100',
           r'D:\Project KLE\Chilli Cropped\chilli1\pure']

# Set up caching
cache_dir = './cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
memory = Memory(location=cache_dir, verbose=0)

@memory.cache
def apply_filter(img_file, img_dir):
    # Load the image
    img_path = os.path.join(img_dir, img_file)
    img = cv2.imread(img_path)

    # Split the image into its color channels
    b, g, r = cv2.split(img)

    # Apply the Gaussian filter to each color channel
    b = cv2.GaussianBlur(b, kernel_size, sigma_x)
    g = cv2.GaussianBlur(g, kernel_size, sigma_x)
    r = cv2.GaussianBlur(r, kernel_size, sigma_x)

    # Merge the color channels back into an image
    img_filtered = cv2.merge((b, g, r))

    # Save the filtered image with the same filename as the original, but in the output directory
    output_dir = os.path.join(img_dir, 'output')
    img_filtered_path = os.path.join(output_dir, img_file)
    cv2.imwrite(img_filtered_path, img_filtered)
    return img_filtered

# Loop through each directory
for img_dir in img_dirs:

    # Create a new directory to save the filtered images
    output_dir = os.path.join(img_dir, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a list of all the image files in the directory
    img_files = os.listdir(img_dir)

    # Loop through each image file
    for img_file in img_files:
        if img_file.endswith('.jpg') or img_file.endswith('.png'):
            apply_filter(img_file, img_dir)
