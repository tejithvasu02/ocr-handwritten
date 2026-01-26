
from datasets import load_dataset
import io
from PIL import Image

def inspect():
    print("Loading MathWriting dataset (streaming)...")
    try:
        # Search suggested "deepcopy/MathWriting-human"
        dataset = load_dataset("deepcopy/MathWriting-human", streaming=True, split="train") 
        # dataset = load_dataset("deepcopy/MathWriting", streaming=True, split="train") 
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    print("Structure:")
    sample = next(iter(dataset))
    print(sample.keys())
    print("\nSample content:")
    for k, v in sample.items():
        if k != 'image':
            print(f"{k}: {v}")
        else:
            print(f"image: {v}")

    # Check if 'latex' or 'label' exists
    if 'latex' in sample:
        print(f"\nLabel (LaTeX): {sample['latex']}")
    elif 'text' in sample:
        print(f"\nLabel (Text): {sample['text']}")

if __name__ == "__main__":
    inspect()
