"""
Dataset loaders for handwritten text recognition datasets.
Supports IAM, CVL, Bentham, and custom lab notebook formats.
"""

import os
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple, Iterator
from dataclasses import dataclass
from pathlib import Path
import random


@dataclass
class HandwritingSample:
    """Single handwriting sample."""
    image_path: str
    text: str
    writer_id: Optional[str] = None
    partition: str = "train"  # train/val/test
    metadata: Optional[Dict] = None


class BaseDatasetLoader:
    """Base class for dataset loaders."""
    
    def __init__(self, root_dir: str, partition: str = "train"):
        self.root_dir = Path(root_dir)
        self.partition = partition
        self.samples: List[HandwritingSample] = []
    
    def load(self) -> List[HandwritingSample]:
        """Load dataset samples. Override in subclasses."""
        raise NotImplementedError
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self) -> Iterator[HandwritingSample]:
        return iter(self.samples)
    
    def to_manifest(self, output_path: str, mode: str = "text"):
        """Export to JSONL manifest format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in self.samples:
                entry = {
                    'image_path': str(sample.image_path),
                    'ground_truth_text': sample.text,
                    'mode': mode,
                    'writer_id': sample.writer_id,
                    'partition': sample.partition
                }
                f.write(json.dumps(entry) + '\n')
        print(f"Wrote {len(self.samples)} samples to {output_path}")
    
    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple['BaseDatasetLoader', 'BaseDatasetLoader', 'BaseDatasetLoader']:
        """Split samples into train/val/test."""
        random.seed(seed)
        samples = self.samples.copy()
        random.shuffle(samples)
        
        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_loader = self.__class__.__new__(self.__class__)
        train_loader.root_dir = self.root_dir
        train_loader.partition = "train"
        train_loader.samples = samples[:train_end]
        
        val_loader = self.__class__.__new__(self.__class__)
        val_loader.root_dir = self.root_dir
        val_loader.partition = "val"
        val_loader.samples = samples[train_end:val_end]
        
        test_loader = self.__class__.__new__(self.__class__)
        test_loader.root_dir = self.root_dir
        test_loader.partition = "test"
        test_loader.samples = samples[val_end:]
        
        return train_loader, val_loader, test_loader


class IAMDatasetLoader(BaseDatasetLoader):
    """
    Loader for IAM Handwriting Database.
    https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
    
    Expected structure:
    iam/
      ├── ascii/
      │   ├── lines.txt
      │   └── words.txt
      ├── lines/
      │   └── a01/a01-000u/a01-000u-00.png
      └── words/
          └── a01/a01-000u/a01-000u-00-00.png
    """
    
    def __init__(
        self,
        root_dir: str,
        partition: str = "train",
        level: str = "lines",  # 'lines' or 'words'
        partition_file: Optional[str] = None
    ):
        super().__init__(root_dir, partition)
        self.level = level
        self.partition_file = partition_file
        self.partition_ids = set()
        
        if partition_file:
            self._load_partition_file(partition_file)
    
    def _load_partition_file(self, filepath: str):
        """Load partition IDs from file."""
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    self.partition_ids.add(line)
    
    def load(self) -> List[HandwritingSample]:
        """Load IAM dataset."""
        self.samples = []
        
        # Load ground truth
        gt_file = self.root_dir / "ascii" / f"{self.level}.txt"
        
        if not gt_file.exists():
            print(f"Warning: Ground truth file not found: {gt_file}")
            return self.samples
        
        with open(gt_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and headers
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(' ')
                
                if self.level == "lines" and len(parts) >= 9:
                    # Format: id ok graylevel components x y w h text...
                    sample_id = parts[0]
                    status = parts[1]
                    text = ' '.join(parts[8:]).replace('|', ' ')
                    
                    if status != 'ok':
                        continue
                    
                    # Check partition
                    form_id = '-'.join(sample_id.split('-')[:2])
                    if self.partition_ids and form_id not in self.partition_ids:
                        continue
                    
                    # Build image path
                    parts_id = sample_id.split('-')
                    image_path = self.root_dir / "lines" / parts_id[0] / f"{parts_id[0]}-{parts_id[1]}" / f"{sample_id}.png"
                    
                    if image_path.exists():
                        self.samples.append(HandwritingSample(
                            image_path=str(image_path),
                            text=text,
                            writer_id=parts_id[0],
                            partition=self.partition
                        ))
                
                elif self.level == "words" and len(parts) >= 9:
                    # Format: id ok graylevel x y w h grammar text
                    sample_id = parts[0]
                    status = parts[1]
                    text = parts[8] if len(parts) > 8 else ""
                    
                    if status != 'ok' or not text:
                        continue
                    
                    # Build image path
                    parts_id = sample_id.split('-')
                    image_path = self.root_dir / "words" / parts_id[0] / f"{parts_id[0]}-{parts_id[1]}" / f"{sample_id}.png"
                    
                    if image_path.exists():
                        self.samples.append(HandwritingSample(
                            image_path=str(image_path),
                            text=text,
                            writer_id=parts_id[0],
                            partition=self.partition
                        ))
        
        print(f"Loaded {len(self.samples)} IAM {self.level} samples")
        return self.samples


class CVLDatasetLoader(BaseDatasetLoader):
    """
    Loader for CVL Database.
    https://cvl.tuwien.ac.at/research/cvl-databases/an-off-line-database-for-writer-retrieval-writer-identification-and-word-spotting/
    
    Expected structure:
    cvl/
      ├── trainset/
      │   ├── 0001-1-1.png
      │   └── ...
      ├── testset/
      └── transcription/
          └── transcription.txt
    """
    
    def __init__(self, root_dir: str, partition: str = "train"):
        super().__init__(root_dir, partition)
    
    def load(self) -> List[HandwritingSample]:
        """Load CVL dataset."""
        self.samples = []
        
        # Determine image directory
        if self.partition == "train":
            img_dir = self.root_dir / "trainset"
        else:
            img_dir = self.root_dir / "testset"
        
        # Load transcriptions
        trans_file = self.root_dir / "transcription" / "transcription.txt"
        transcriptions = {}
        
        if trans_file.exists():
            with open(trans_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        transcriptions[parts[0]] = parts[1]
        
        # Load images
        if img_dir.exists():
            for img_path in img_dir.glob("*.png"):
                sample_id = img_path.stem
                text = transcriptions.get(sample_id, "")
                
                if text:
                    # Extract writer ID from filename (e.g., "0001-1-1")
                    writer_id = sample_id.split('-')[0] if '-' in sample_id else None
                    
                    self.samples.append(HandwritingSample(
                        image_path=str(img_path),
                        text=text,
                        writer_id=writer_id,
                        partition=self.partition
                    ))
        
        print(f"Loaded {len(self.samples)} CVL samples")
        return self.samples


class BenthamDatasetLoader(BaseDatasetLoader):
    """
    Loader for Bentham Manuscripts Dataset.
    
    Expected structure:
    bentham/
      ├── Images/Lines/
      │   └── *.png
      └── Transcriptions/
          └── *.txt (one per image)
    """
    
    def __init__(self, root_dir: str, partition: str = "train"):
        super().__init__(root_dir, partition)
    
    def load(self) -> List[HandwritingSample]:
        """Load Bentham dataset."""
        self.samples = []
        
        img_dir = self.root_dir / "Images" / "Lines"
        trans_dir = self.root_dir / "Transcriptions"
        
        if not img_dir.exists():
            # Try alternative structure
            img_dir = self.root_dir / "lines"
            trans_dir = self.root_dir / "transcriptions"
        
        if img_dir.exists():
            for img_path in img_dir.glob("*.png"):
                # Look for matching transcription
                trans_path = trans_dir / f"{img_path.stem}.txt"
                
                if trans_path.exists():
                    with open(trans_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    if text:
                        self.samples.append(HandwritingSample(
                            image_path=str(img_path),
                            text=text,
                            partition=self.partition
                        ))
        
        print(f"Loaded {len(self.samples)} Bentham samples")
        return self.samples


class LabNotebookLoader(BaseDatasetLoader):
    """
    Loader for custom lab notebook/exam sheet data.
    
    Expected structure:
    data/
      ├── images/
      │   └── *.jpg/png
      └── labels.json or labels.csv
    
    labels.json format:
    [
      {"image": "001.png", "text": "..."},
      ...
    ]
    """
    
    def __init__(
        self,
        root_dir: str,
        partition: str = "train",
        labels_file: str = "labels.json"
    ):
        super().__init__(root_dir, partition)
        self.labels_file = labels_file
    
    def load(self) -> List[HandwritingSample]:
        """Load lab notebook data."""
        self.samples = []
        
        labels_path = self.root_dir / self.labels_file
        img_dir = self.root_dir / "images"
        
        if labels_path.suffix == '.json':
            self._load_json_labels(labels_path, img_dir)
        elif labels_path.suffix == '.csv':
            self._load_csv_labels(labels_path, img_dir)
        elif labels_path.suffix == '.jsonl':
            self._load_jsonl_labels(labels_path, img_dir)
        
        print(f"Loaded {len(self.samples)} lab notebook samples")
        return self.samples
    
    def _load_json_labels(self, labels_path: Path, img_dir: Path):
        """Load labels from JSON file."""
        with open(labels_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for entry in data:
            image_name = entry.get('image') or entry.get('filename')
            text = entry.get('text') or entry.get('label') or entry.get('transcription')
            
            if image_name and text:
                image_path = img_dir / image_name
                if image_path.exists():
                    self.samples.append(HandwritingSample(
                        image_path=str(image_path),
                        text=text,
                        partition=self.partition,
                        metadata=entry.get('metadata')
                    ))
    
    def _load_csv_labels(self, labels_path: Path, img_dir: Path):
        """Load labels from CSV file."""
        import csv
        
        with open(labels_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_name = row.get('image') or row.get('filename')
                text = row.get('text') or row.get('label')
                
                if image_name and text:
                    image_path = img_dir / image_name
                    if image_path.exists():
                        self.samples.append(HandwritingSample(
                            image_path=str(image_path),
                            text=text,
                            partition=self.partition
                        ))
    
    def _load_jsonl_labels(self, labels_path: Path, img_dir: Path):
        """Load labels from JSONL file."""
        with open(labels_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                image_name = entry.get('image') or entry.get('image_path')
                text = entry.get('ground_truth_text') or entry.get('text')
                
                if image_name and text:
                    # Handle both absolute and relative paths
                    if os.path.isabs(image_name):
                        image_path = Path(image_name)
                    else:
                        image_path = img_dir / image_name
                    
                    if image_path.exists():
                        self.samples.append(HandwritingSample(
                            image_path=str(image_path),
                            text=text,
                            partition=self.partition
                        ))


class CombinedDatasetLoader(BaseDatasetLoader):
    """
    Combines multiple dataset loaders into one.
    """
    
    def __init__(self, loaders: List[BaseDatasetLoader]):
        self.loaders = loaders
        self.samples = []
    
    def load(self) -> List[HandwritingSample]:
        """Load all datasets."""
        self.samples = []
        
        for loader in self.loaders:
            loader.load()
            self.samples.extend(loader.samples)
        
        print(f"Combined {len(self.samples)} total samples from {len(self.loaders)} datasets")
        return self.samples


def create_training_manifest(
    datasets: Dict[str, str],
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
):
    """
    Create training manifests from multiple datasets.
    
    Args:
        datasets: Dict of {dataset_type: root_path}
            e.g., {"iam": "/path/to/iam", "cvl": "/path/to/cvl"}
        output_dir: Output directory for manifests
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_samples = []
    
    for dataset_type, root_path in datasets.items():
        if dataset_type == "iam":
            loader = IAMDatasetLoader(root_path, level="lines")
        elif dataset_type == "cvl":
            loader = CVLDatasetLoader(root_path)
        elif dataset_type == "bentham":
            loader = BenthamDatasetLoader(root_path)
        elif dataset_type == "lab":
            loader = LabNotebookLoader(root_path)
        else:
            print(f"Unknown dataset type: {dataset_type}")
            continue
        
        loader.load()
        all_samples.extend(loader.samples)
    
    # Shuffle and split
    random.shuffle(all_samples)
    n = len(all_samples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_samples = all_samples[:train_end]
    val_samples = all_samples[train_end:val_end]
    test_samples = all_samples[val_end:]
    
    # Write manifests
    for samples, name in [
        (train_samples, "train"),
        (val_samples, "val"),
        (test_samples, "test")
    ]:
        output_path = os.path.join(output_dir, f"{name}.jsonl")
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                entry = {
                    'image_path': sample.image_path,
                    'ground_truth_text': sample.text,
                    'mode': 'text',
                    'writer_id': sample.writer_id,
                    'partition': name
                }
                f.write(json.dumps(entry) + '\n')
        print(f"Wrote {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create training manifests from datasets")
    parser.add_argument("--iam", type=str, help="Path to IAM dataset")
    parser.add_argument("--cvl", type=str, help="Path to CVL dataset")
    parser.add_argument("--bentham", type=str, help="Path to Bentham dataset")
    parser.add_argument("--lab", type=str, help="Path to lab notebook data")
    parser.add_argument("--output", type=str, default="data/manifests", help="Output directory")
    
    args = parser.parse_args()
    
    datasets = {}
    if args.iam:
        datasets["iam"] = args.iam
    if args.cvl:
        datasets["cvl"] = args.cvl
    if args.bentham:
        datasets["bentham"] = args.bentham
    if args.lab:
        datasets["lab"] = args.lab
    
    if datasets:
        create_training_manifest(datasets, args.output)
    else:
        print("No datasets specified. Use --iam, --cvl, --bentham, or --lab")
