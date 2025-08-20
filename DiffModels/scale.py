import os
from PIL import Image
import numpy as np

def scale_image(image):
    array = np.array(image).astype(np.float32)
    min_val = array.min()
    max_val = array.max()
    
    scaled_array = 255 * (array - min_val) / (max_val - min_val)
    scaled_image = Image.fromarray(scaled_array.astype(np.uint8))
    
    return scaled_image

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            
            scaled_img = scale_image(img)
            
            output_path = os.path.join(output_folder, filename)
            scaled_img.save(output_path)
            print(f"Processed and saved {filename}")

input_folder = ''
output_folder = ''

process_images(input_folder, output_folder)
