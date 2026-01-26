
import os
import argparse
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import time

def quantize_model(args):
    print(f"Loading model from {args.model_path}...")
    processor = TrOCRProcessor.from_pretrained(args.model_path)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_path)
    
    print("Applying dynamic quantization (INT8)...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    
    # Save
    os.makedirs(args.output_path, exist_ok=True)
    print(f"Saving quantized model to {args.output_path}...")
    
    # Quantized models can be saved with torch.save
    # But transformers save_pretrained might not work well with quantized layers directly 
    # if we want to load it back as a Transformers model easily.
    # Standard approach: Save state dict or script.
    # Providing a simple wrapper to save and load.
    
    # Actually, saving as Pytorch generic
    torch.save(quantized_model.state_dict(), os.path.join(args.output_path, "quantized_model.pth"))
    processor.save_pretrained(args.output_path)
    
    # Save config
    model.config.save_pretrained(args.output_path)
    
    size_orig = os.path.getsize(os.path.join(args.model_path, "pytorch_model.bin")) / (1024*1024)
    size_quant = os.path.getsize(os.path.join(args.output_path, "quantized_model.pth")) / (1024*1024)
    
    print(f"Original Size: {size_orig:.2f} MB")
    print(f"Quantized Size: {size_quant:.2f} MB")
    print(f"Reduction: {size_orig/size_quant:.2f}x")

def main():
    parser = argparse.ArgumentParser(description="Quantize TrOCR Model (Dynamic INT8)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (e.g. checkpoints/trocr_text/final)")
    parser.add_argument("--output_path", type=str, default="checkpoints/trocr_text_quantized", help="Output directory")
    
    args = parser.parse_args()
    quantize_model(args)

if __name__ == "__main__":
    main()
