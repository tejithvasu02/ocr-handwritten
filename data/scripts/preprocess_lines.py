
import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def remove_underlines(img, threshold=200):
    """
    Remove horizontal lines effectively.
    Assumes dark text on light background (grayscale).
    """
    # 1. Binarize (inverted for processing: white text, black bg)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Detect horizontal lines
    h, w = img.shape
    # Line needs to be at least 40% of width to be an 'underline' usually, 
    # but for short words/headings might be shorter. 
    # Let's say 20% of width or fixed length.
    line_min_width = int(w * 0.2)
    if line_min_width < 20: line_min_width = 20
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_min_width, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    # 3. Filter lines - we only want lines near the bottom? 
    # Or just remove all long straight lines.
    # Danger: removing top of 'T' or 'E'.
    # Heuristic: Underlines are usually in the bottom half of the text line image.
    
    # Mask out top half of detected lines to protect characters
    # (Assuming single line crop)
    mask = np.zeros_like(detected_lines)
    mask[int(h*0.4):, :] = 255 
    detected_lines = cv2.bitwise_and(detected_lines, mask)
    
    # Dilation to cover artifacts
    dilated_lines = cv2.dilate(detected_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)
    
    # 4. Inpaint
    # CV2 inpaint expects 8-bit mask of non-zero pixels.
    # RADIUS: 3px
    inpainted = cv2.inpaint(img, dilated_lines, 3, cv2.INPAINT_TELEA)
    
    return inpainted

def boost_dots(img):
    """
    Make small dots slightly thicker to be unambiguous.
    """
    # Invert
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    output_mask = np.zeros_like(binary)
    
    has_dots = False
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Heuristic for "dot": Area is small (e.g. < 50 pixels depending on resolution) 
        # but not noise (< 5 pixels).
        if 5 <= area <= 60:
            output_mask[labels == i] = 255
            has_dots = True
            
    if not has_dots:
        return img
        
    # Dilate the dots
    kernel = np.ones((2,2), np.uint8) # Small dilation
    dilated_dots = cv2.dilate(output_mask, kernel, iterations=1)
    
    # Blend back into image
    # Where dilated dots are white, make original image black (0) or darker?
    # Original is grayscale 0-255 (255 is white).
    # We want to darken the pixels at dilated locations.
    
    # Create an inverse mask
    # output: img AND NOT(dilated_dots) -> clears space
    # then OR with BLACK (0)? No.
    # We want to burn the dots in.
    
    # Easier: Convert dilated mask to inverted (black dots on white)
    dots_inv = cv2.bitwise_not(dilated_dots)
    
    # Combine: keep original image pixels UNLESS it's a dot pixel.
    # But we want to preserve the original stroke darkness.
    # Simple 'min' or 'multiply'.
    
    # Make mask: 0 where dot, 1 where bg
    # Actually, let's just use cv2.min
    # Dilated dots (255) needs to be inverted to (0) to darken?
    
    # Strategy: 
    # 1. Start with original `img`.
    # 2. Where `dilated_dots` is 255, set `img` pixel to 0 (black).
    # Caution: Don't destroy overlapping strokes? 
    # We are only dilating components that were isolated dots.
    
    result = img.copy()
    result[dilated_dots == 255] = 0 # Force black
    
    return result

def process_single_image(args):
    """
    Worker for thread pool.
    """
    sample, input_dir, output_dir, use_underline, use_dots = args
    
    try:
        img_path = sample['image_path']
        if not os.path.isabs(img_path):
            img_path = os.path.join(input_dir, img_path) # Fallback
            
        if not os.path.exists(img_path):
            return None
            
        # Load grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        
        # 1. Height Norm (64px)
        h, w = img.shape
        target_h = 64
        scale = target_h / h
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
        
        processed = resized
        
        # 2. Underline Removal
        if use_underline:
            processed = remove_underlines(processed)
            
        # 3. Dot Boosting
        if use_dots:
            processed = boost_dots(processed)
            
        # 4. Save
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, processed)
        
        # Update record
        new_sample = sample.copy()
        new_sample['image_path'] = os.path.abspath(out_path)
        
        # 5. Label Guardrails
        # Remove '0' if alphabetic text
        text = new_sample['ground_truth_text']
        # If '0' in text and no other digits, and text length > 3 -> suspicious
        if '0' in text:
            # Check if likely numeric context
            has_other_digits = any(c.isdigit() and c!='0' for c in text)
            if not has_other_digits and len(text) > 3:
                # Replace 0 with o ? Or just warn?
                # User complaint: "dot on i misread as 0".
                # If ground truth has 0, we trust it? 
                # Our audit showed only 4 legit cases.
                # So we leave it. The audit protects us.
                pass
        
        return new_sample
        
    except Exception as e:
        # print(f"Error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/processed/clean_lines")
    parser.add_argument("--output_manifest", type=str, default="data/manifests/clean_train.jsonl")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load manifest
    samples = []
    with open(args.manifest, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
            
    print(f"Processing {len(samples)} lines...")
    
    # Prepare args
    # use_underline=True, use_dots=True
    task_args = [(s, "", args.output_dir, True, True) for s in samples]
    
    clean_samples = []
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(process_single_image, task_args), total=len(samples)))
        
    for res in results:
        if res:
            clean_samples.append(res)
            
    # Save manifest
    with open(args.output_manifest, 'w') as f:
        for s in clean_samples:
            f.write(json.dumps(s) + '\n')
            
    print(f"Saved {len(clean_samples)} processed samples to {args.output_manifest}")

if __name__ == "__main__":
    main()
