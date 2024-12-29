import cv2
import os
from tqdm import tqdm

def resize_images(input_dir, output_dir, size=(224, 224)):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all images from input directory
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    # Process each image with progress bar
    for image_path in tqdm(image_files, desc="Resizing images"):
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            continue
            
        # Resize image
        resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        
        # Create output path maintaining folder structure
        rel_path = os.path.relpath(image_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        
        # Create necessary subdirectories
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save resized image
        cv2.imwrite(output_path, resized)

if __name__ == "__main__":
    # Resize images in both folders
    resize_images("raw_data/screw_present", "data/screw_present")
    resize_images("raw_data/screw_missing", "data/screw_missing")
    print("Image resizing completed!")
