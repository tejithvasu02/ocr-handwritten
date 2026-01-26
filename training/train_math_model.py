
import os
import sys
import argparse
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm

# Import tokenizer utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer_utils import adapt_tokenizer, STANDARD_LATEX_TOKENS

class MathStreamingDataset(IterableDataset):
    def __init__(self, dataset, processor, max_target_length=256):
        self.dataset = dataset
        self.processor = processor
        self.max_target_length = max_target_length

    def __iter__(self):
        for sample in self.dataset:
            image = sample['image'].convert("RGB")
            # MathWriting 'latex' column
            text = sample.get('latex', "")
            
            # Pixel values
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
            
            # Labels
            labels = self.processor.tokenizer(
                text,
                padding="max_length",
                max_length=self.max_target_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.squeeze(0)
            
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
            yield {
                "pixel_values": pixel_values,
                "labels": labels
            }

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (Streaming Mode)")

    # Load dataset streaming
    print(f"Loading dataset: {args.dataset_name}...")
    # split='train' is huge. We use streaming.
    dataset = load_dataset(args.dataset_name, streaming=True, split="train")
    # Validation? MathWriting doesn't have split="validation" by default on HF? 
    # Usually it's train/test.
    # We'll valid on a buffered 'take' from train or load 'test' split.
    try:
        val_dataset_stream = load_dataset(args.dataset_name, streaming=True, split="test")
    except:
        val_dataset_stream = None
        print("Warning: No test split found.")

    # Processor & Model (Base Printed is better for Math than Handwritten)
    # But for Handwritten Math, maybe handwritten is better?
    # MathWriting is handwritten.
    # We stick to base-printed as foundation because it knows symbols better?
    # Or start from our fine-tuned text model?
    # User can choose. Default to microsoft/trocr-base-printed
    
    print(f"Loading model: {args.model_name}")
    processor = TrOCRProcessor.from_pretrained(args.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)
    
    # Adapt tokenizer for LaTeX
    # We should add symbols.
    # Re-using tokenizer_utils logic
    # Make sure we don't overwrite if not requested
    # But for Math, we NEED latex tokens.
    
    # Simple check if vocab is small
    if len(processor.tokenizer) < 51000: # base printed is 50k BPE
         print("Adapting tokenizer/model for LaTeX...")
         processor, model = adapt_tokenizer(args.model_name, STANDARD_LATEX_TOKENS, output_dir=None) # return objects
    
    model.to(device)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Wrap dataset
    train_ds = MathStreamingDataset(dataset, processor, args.max_target_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0) # Must be 0 for streaming? Or carefully managed. 0 is safe.

    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Scheduler needs max steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)
    
    model.train()
    
    step = 0
    total_loss = 0
    
    progress_bar = tqdm(range(args.max_steps), desc="Training")
    
    # Iterate forever/until max_steps. Reshuffle buffer? HF Streaming does shuffle? 
    # dataset = dataset.shuffle(seed=42, buffer_size=1000) recommended
    dataset = dataset.shuffle(seed=42, buffer_size=1000)
    
    train_iter = iter(train_loader)
    
    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        step += 1
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        if step % args.save_steps == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"\nSaved checkpoint to {save_path}")

    # Final save
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="microsoft/trocr-base-printed")
    parser.add_argument("--dataset-name", type=str, default="deepcopy/MathWriting-human")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=5000) # Streaming uses steps
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="checkpoints/trocr_math")
    parser.add_argument("--max_target_length", type=int, default=256)
    
    args = parser.parse_args()
    train(args)
