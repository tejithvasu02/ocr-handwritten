
import os
import cv2
import random
import numpy as np
import albumentations as A
import json
from PIL import Image

# Recreate the exact augmentation logic from train_trocr_text.py

def add_random_lines(image, **kwargs):
    img = image.copy()
    h, w = img.shape[:2]
    # Draw lines in bottom half
    y = random.randint(int(h*0.6), h-1)
    color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)) # Dark line
    thickness = random.randint(1, 3)
    cv2.line(img, (0, y), (w, y), color, thickness)
    return img

def visualize_augmentation(manifest_path, output_dir, count=10):
    os.makedirs(output_dir, exist_ok=True)
    
    transform = A.Compose([
        A.Lambda(name="add_lines", image=add_random_lines, p=0.9), # High probability for visualization to ensure we see it
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=3, p=0.5, border_mode=0, value=(255, 255, 255)),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.RandomBrightnessContrast(p=0.4),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
    ])
    
    # Load manifest
    samples = []
    with open(manifest_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
            
    # Pick random samples
    selected = random.sample(samples, min(count, len(samples)))
    
    for i, sample in enumerate(selected):
        img_path = sample['image_path']
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Augment
        augmented = transform(image=img)["image"]
        
        # Save side-by-side
        # Resize to same height if needed
        h, w, c = img.shape
        aug_h, aug_w, _ = augmented.shape
        
        # Save
        out_name = f"aug_{i}.jpg"
        # Convert back to BGR for cv2 save
        aug_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, out_name), aug_bgr)
        print(f"Saved {out_name}")

if __name__ == "__main__":
    visualize_augmentation("data/manifests/clean_train.jsonl", "debug/aug_visualization")
