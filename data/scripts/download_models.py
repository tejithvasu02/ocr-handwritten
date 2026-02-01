
import os
import argparse
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def download_model(model_name, output_dir):
    print(f"Downloading {model_name} to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and save processor
    processor = TrOCRProcessor.from_pretrained(model_name)
    processor.save_pretrained(output_dir)
    
    # Load and save model
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.save_pretrained(output_dir)
    
    print(f"Successfully saved {model_name} to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download TrOCR models")
    parser.add_argument("--base", action="store_true", help="Download base handwritten model")
    parser.add_argument("--small", action="store_true", help="Download small handwritten model")
    
    args = parser.parse_args()
    
    if args.base:
        download_model("microsoft/trocr-base-handwritten", "models/trocr_base_handwritten")
        
    if args.small:
        download_model("microsoft/trocr-small-handwritten", "models/trocr_small_handwritten")

    if not args.base and not args.small:
        print("Please specify --base or --small")

if __name__ == "__main__":
    main()
