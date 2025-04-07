"""
Functions for visualizing probe analysis results.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(csv_file, output_dir=None, show=True):
    """
    Plot ROC curve from CSV file.
    
    Args:
        csv_file: Path to CSV with probe stats
        output_dir: Directory to save visualizations
        show: Whether to display the plot
        
    Returns:
        tuple: (figure, metrics_dict)
    """
    # Load the data
    df = pd.read_csv(csv_file)
    
    # Extract filename for title
    filename = os.path.basename(csv_file)
    
    # Prepare data for ROC calculation
    y_true = (df['type'] == 'deployment').astype(int)
    y_score = df['mean_score']
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Calculate optimal threshold (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    sensitivity = tpr[optimal_idx]
    specificity = 1-fpr[optimal_idx]
    
    # Create wider figure
    plt.figure(figsize=(12, 6))
    
    # Plot the ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {filename}')
    plt.legend(loc="lower right")
    
    # Add text box with metrics
    metrics_text = f'AUROC: {roc_auc:.3f}\nThreshold: {optimal_threshold:.3f}\nSensitivity: {sensitivity:.3f}\nSpecificity: {specificity:.3f}'
    plt.text(0.05, 0.95, metrics_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.8),
             verticalalignment='top',
             fontfamily='monospace')
    
    # Save visualizations if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(filename)[0]
        
        # Save as PDF
        pdf_path = os.path.join(output_dir, f'{base_name}_roc.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
        
        # Save as SVG
        svg_path = os.path.join(output_dir, f'{base_name}_roc.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        
        # Save as PNG
        png_path = os.path.join(output_dir, f'{base_name}_roc.png')
        plt.savefig(png_path, format='png', bbox_inches='tight', dpi=150)
    
    # Display the plot if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    # Return figure and metrics
    metrics = {
        'auroc': roc_auc,
        'threshold': optimal_threshold,
        'sensitivity': sensitivity,
        'specificity': specificity
    }
    
    return plt.gcf(), metrics

def plot_all_rocs_in_directory(directory, output_dir=None, show=False):
    """
    Plot ROC curves for all CSV files in directory.
    
    Args:
        directory: Directory with CSV files
        output_dir: Directory to save visualizations
        show: Whether to display plots
        
    Returns:
        dict: Metrics for all files
    """
    if output_dir is None:
        output_dir = os.path.join(directory, 'visualizations')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of CSV files
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return {}
    
    # Plot ROC curve for each file
    metrics = {}
    for file_name in csv_files:
        file_path = os.path.join(directory, file_name)
        try:
            _, file_metrics = plot_roc_curve(file_path, output_dir, show=show)
            metrics[file_name] = file_metrics
        except Exception as e:
            print(f"Error plotting ROC for {file_name}: {e}")
    
    # Create summary plot of all ROC curves
    plt.figure(figsize=(12, 10))
    
    # Sort files by layer number (assuming filenames contain layer numbers)
    sorted_files = sorted(csv_files, key=lambda x: int(x.split('layer')[1].split('.')[0]) 
                          if 'layer' in x else 0)
    
    # Plot ROC curves with a color gradient
    cmap = plt.cm.viridis
    colors = [cmap(i) for i in np.linspace(0, 1, len(sorted_files))]
    
    for i, file_name in enumerate(sorted_files):
        file_path = os.path.join(directory, file_name)
        
        try:
            # Load the data
            df = pd.read_csv(file_path)
            
            # Extract layer number
            layer_num = int(file_name.split('layer')[1].split('.')[0]) if 'layer' in file_name else i
            
            # Prepare data for ROC calculation
            y_true = (df['type'] == 'deployment').astype(int)
            y_score = df['mean_score']
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            # Plot the ROC curve
            plt.plot(fpr, tpr, color=colors[i], lw=1.5, 
                     label=f'Layer {layer_num} (AUC = {roc_auc:.3f})')
        
        except Exception as e:
            print(f"Error plotting ROC for {file_name} in summary: {e}")
    
    # Add reference line
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Layers')
    plt.legend(loc="lower right", fontsize='small', ncol=2)
    
    # Save summary plot
    summary_path = os.path.join(output_dir, 'all_layers_roc.pdf')
    plt.savefig(summary_path, format='pdf', bbox_inches='tight', dpi=300)
    
    summary_svg = os.path.join(output_dir, 'all_layers_roc.svg')
    plt.savefig(summary_svg, format='svg', bbox_inches='tight')
    
    summary_png = os.path.join(output_dir, 'all_layers_roc.png')
    plt.savefig(summary_png, format='png', bbox_inches='tight', dpi=150)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Create performance across layers plot
    layer_nums = []
    aurocs = []
    
    for file_name in sorted_files:
        if 'layer' in file_name:
            layer_num = int(file_name.split('layer')[1].split('.')[0])
            layer_nums.append(layer_num)
            aurocs.append(metrics[file_name]['auroc'])
    
    if layer_nums:
        plt.figure(figsize=(12, 6))
        plt.plot(layer_nums, aurocs, 'o-', color='blue', lw=2)
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        plt.grid(True, alpha=0.3)
        plt.xlabel('Layer')
        plt.ylabel('AUROC')
        plt.title('AUROC by Layer')
        
        # Save layer performance plot
        layer_perf_path = os.path.join(output_dir, 'layer_performance.pdf')
        plt.savefig(layer_perf_path, format='pdf', bbox_inches='tight', dpi=300)
        
        layer_perf_svg = os.path.join(output_dir, 'layer_performance.svg')
        plt.savefig(layer_perf_svg, format='svg', bbox_inches='tight')
        
        layer_perf_png = os.path.join(output_dir, 'layer_performance.png')
        plt.savefig(layer_perf_png, format='png', bbox_inches='tight', dpi=150)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    return metrics
