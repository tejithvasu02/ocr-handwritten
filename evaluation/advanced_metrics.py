
import argparse
import json
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from tqdm import tqdm
import Levenshtein
import re

def compute_metrics(model, processor, manifest_path, count=100, device="cpu"):
    samples = []
    with open(manifest_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
            
    # Filter for alphabetic-only samples to test 0-hallucination strictly?
    # Or just use mixed and check valid 0s.
    
    total_samples = 0
    zero_hallucinations = 0
    catalog_drifts = 0
    
    print(f"Testing {count} samples from {manifest_path}...")
    
    # Slice
    samples = samples[:count]
    
    model.eval()
    
    results = []
    
    for sample in tqdm(samples):
        img_path = sample['image_path']
        gt_text = sample['ground_truth_text']
        
        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_new_tokens=100)
                pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
            # Metric 1: Zero Hallucination
            # Failure: Pred contains '0' AND GT does NOT contain '0' (or any digit)
            gt_has_digit = any(c.isdigit() for c in gt_text)
            pred_has_zero = '0' in pred_text
            
            if pred_has_zero and not gt_has_digit:
                zero_hallucinations += 1
                
            # Metric 2: Catalog Drift
            # Check specific keywords or general Levenshtein distance relative to length
            if "catalog" in pred_text.lower() and "catalyze" in gt_text.lower():
                catalog_drifts += 1
                
            results.append({
                "gt": gt_text,
                "pred": pred_text,
                "zero_fail": (pred_has_zero and not gt_has_digit)
            })
            
            total_samples += 1
            
        except Exception as e:
            print(f"Error: {e}")
            
    # Stats
    zero_rate = (zero_hallucinations / total_samples) * 100 if total_samples > 0 else 0
    
    print("\n--- Advanced Accuracy Metrics ---")
    print(f"Total Samples Tested: {total_samples}")
    print(f"Zero Hallucination Rate: {zero_rate:.2f}% (Target: < 1%)")
    print(f"Lexical Drift (Catalyze->Catalog): {catalog_drifts}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--manifest", type=str, default="data/manifests/combined_val.jsonl")
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        processor = TrOCRProcessor.from_pretrained(args.model_path)
        model = VisionEncoderDecoderModel.from_pretrained(args.model_path).to(device)
        compute_metrics(model, processor, args.manifest, args.count, device)
    except Exception as e:
        print(f"Model load failed: {e}")
