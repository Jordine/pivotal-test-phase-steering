#!/usr/bin/env python
"""
Script for visualizing probe performance from existing results.

Usage:
    python visualize_results.py --results RESULTS_DIR
"""
import argparse
import os
import sys

sys.path.append('.')
from src.visualization import plot_all_rocs_in_directory

def main():
    parser = argparse.ArgumentParser(description="Visualize probe performance")
    parser.add_argument("--results", required=True, help="Results directory with CSV files")
    parser.add_argument("--output", help="Output directory for visualizations (default: results_dir/visualizations)")
    parser.add_argument("--show", action="store_true", help="Show plots")
    parser.add_argument("--include-control", action="store_true", help="Include control directories")
    args = parser.parse_args()
    
    if not os.path.isdir(args.results):
        print(f"Error: {args.results} is not a valid directory")
        return
    
    output_dir = args.output if args.output else os.path.join(args.results, "visualizations")
    
    # Process main results directory
    print(f"Generating visualizations for: {args.results}")
    metrics = plot_all_rocs_in_directory(args.results, output_dir, show=args.show)
    
    # Process control directories if requested
    if args.include_control:
        control_dir = os.path.join(args.results, "control")
        if os.path.isdir(control_dir):
            for item in os.listdir(control_dir):
                probe_dir = os.path.join(control_dir, item)
                if os.path.isdir(probe_dir):
                    probe_viz_dir = os.path.join(probe_dir, "visualizations")
                    print(f"Generating visualizations for: {probe_dir}")
                    plot_all_rocs_in_directory(probe_dir, probe_viz_dir, show=args.show)
    
    print(f"Visualizations completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()