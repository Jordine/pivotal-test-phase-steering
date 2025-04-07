#!/usr/bin/env python
"""
Script for generating contrastive steering vectors.

Usage:
    python generate_vectors.py --model MODEL_PATH --data DATASET_PATH --output OUTPUT_DIR
"""
import argparse
import os
import torch
from tqdm import tqdm

import sys
sys.path.append('.')
from src.vector_generation import run_experiment
from src.utils import load_model, load_contrastive_dataset

def main():
    parser = argparse.ArgumentParser(description="Generate contrastive steering vectors")
    parser.add_argument("--model", required=True, help="Model path or name")
    parser.add_argument("--data", required=True, help="Path to contrastive dataset")
    parser.add_argument("--output", help="Output directory name prefix")
    parser.add_argument("--layers", help="Comma-separated list of layers (default: all)")
    parser.add_argument("--device", default="cuda", help="Device to run on (default: cuda)")
    parser.add_argument("--dtype", default="bfloat16", help="Data type (default: bfloat16)")
    args = parser.parse_args()
    
    # Set dtype
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    # Load model and data
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model, device=args.device, dtype=dtype)
    
    print(f"Loading dataset: {args.data}")
    dataset = load_contrastive_dataset(args.data, tokenizer)
    print(f"Loaded {len(dataset)} examples")
    
    # Parse layers
    layers = None
    if args.layers:
        layers = [int(l) for l in args.layers.split(",")]
        print(f"Using layers: {layers}")
    
    # Run experiment
    output_dirs, raw_vectors, norm_vectors = run_experiment(
        model=model,
        dataset=dataset,
        layers=layers,
        output_dir=args.output
    )
    
    print(f"Experiment completed. Results saved to {output_dirs['base']}")

if __name__ == "__main__":
    main()
