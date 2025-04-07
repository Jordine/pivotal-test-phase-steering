"""
Functions for analyzing steering vector probes and evaluating their performance.
"""
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

def get_probe_scores(model, steering_vectors, layer, prompt, tokenizer=None):
    """
    Calculate probe activation values for each token in a prompt.
    
    Args:
        model: HookedTransformer model
        steering_vectors: Dictionary of steering vectors
        layer: Layer to probe
        prompt: Input prompt
        tokenizer: Optional tokenizer
        
    Returns:
        tuple: (tokens, scores, token_strings)
    """
    if tokenizer is None:
        tokenizer = model.tokenizer
    
    # Ensure the steering vector for the requested layer exists
    if layer not in steering_vectors:
        raise ValueError(f"No steering vector available for layer {layer}")
    
    # Get the steering vector for the requested layer
    steering_vector = steering_vectors[layer]
    
    # Tokenize the prompt
    tokens = model.to_tokens(prompt) if hasattr(model, 'to_tokens') else torch.tensor([tokenizer.encode(prompt)]).to(model.device)
    
    # Store activation scores for each token position
    scores = []
    
    # Define the hook function to capture activations at each position
    def hook_fn(activation, hook):
        # For each token position in the sequence
        for pos in range(activation.shape[1]):
            # Get activation at this position
            act = activation[0, pos, :]
            
            # Project the activation onto the steering vector
            # (dot product of activation with normalized steering vector)
            projection = torch.dot(act, steering_vector[0]) / steering_vector[0].norm()
            
            # Save the scalar projection value
            if pos == len(scores):  # Only add if we haven't processed this position yet
                scores.append(projection.item())
        
        return activation
    
    # Run the model with the hook attached
    model.reset_hooks()
    with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", hook_fn)]):
        with torch.no_grad():
            model(tokens)
    
    # Get the token strings for better interpretability
    token_strings = [tokenizer.decode([t]) for t in tokens[0].tolist()]
    
    return tokens, scores, token_strings

def analyze_dataset_with_probe(model, steering_vectors, layer, dataset, 
                              tokenizer=None, output_file=None):
    """
    Loop through a dataset and collect probe statistics.
    
    Args:
        model: HookedTransformer model
        steering_vectors: Dictionary of steering vectors
        layer: Layer to probe
        dataset: Dataset to analyze
        tokenizer: Optional tokenizer
        output_file: Path to save results
        
    Returns:
        list: Results for each item in dataset
    """
    if tokenizer is None:
        tokenizer = model.tokenizer
    
    # Prepare results storage
    results = []
    
    # Process each question in the dataset
    for i, item in enumerate(tqdm(dataset, desc="Analyzing questions")):
        question = item['question']
        
        try:
            # Get probe scores for the question
            tokens, scores, token_strings = get_probe_scores(model, steering_vectors, layer, question, tokenizer)
            
            # Filter out BOS tokens
            filtered_scores = []
            filtered_tokens = []
            filtered_indices = []
            
            for j, (token, score) in enumerate(zip(token_strings, scores)):
                # Skip the BOS token (different models might have different BOS tokens)
                if ("<|begin_of_text|>" in token or 
                    "<s>" in token or 
                    "<BOS>" in token or
                    token.strip() == ""):
                    continue
                
                filtered_indices.append(j)
                filtered_scores.append(score)
                filtered_tokens.append(token)
            
            # Calculate statistics
            if filtered_scores:
                max_score = max(filtered_scores)
                min_score = min(filtered_scores)
                mean_score = sum(filtered_scores) / len(filtered_scores)
                std_score = (sum((x - mean_score) ** 2 for x in filtered_scores) / len(filtered_scores)) ** 0.5
            else:
                max_score = min_score = mean_score = std_score = float('nan')
            
            # Record results
            result = {
                'question_id': i,
                'question': question,
                'type': item['type'],
                'max_score': max_score,
                'min_score': min_score,
                'mean_score': mean_score,
                'std_score': std_score,
            }
            
            results.append(result)
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(dataset)} questions")
        
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            # Add a row with the error
            results.append({
                'question_id': i,
                'question': question,
                'type': item['type'],
                'max_score': float('nan'),
                'min_score': float('nan'),
                'mean_score': float('nan'),
                'std_score': float('nan'),
                'error': str(e)
            })
    
    # Save results to CSV if output file is specified
    if output_file:
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    return results

def calculate_auroc_from_csv(file_path):
    """
    Calculate AUROC metrics from a CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        tuple: (metrics_dict, error_message)
    """
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Check if the required columns exist
        required_columns = ['type', 'mean_score']
        if not all(col in df.columns for col in required_columns):
            return None, "Missing required columns (type and/or mean_score)"
        
        # Prepare data for ROC calculation
        # 1 for deployment (positive class), 0 for testing (negative class)
        y_true = (df['type'] == 'deployment').astype(int)
        
        # Use mean_score as the prediction score
        y_score = df['mean_score']
        
        # Calculate ROC curve points
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Calculate optimal threshold (Youden's J statistic)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        sensitivity = tpr[optimal_idx]
        specificity = 1-fpr[optimal_idx]
        
        return {
            'auroc': roc_auc,
            'optimal_threshold': optimal_threshold,
            'sensitivity': sensitivity,
            'specificity': specificity
        }, None
    
    except Exception as e:
        return None, str(e)

def process_directory(directory_path, verbose=True):
    """
    Process all CSV files in a directory and calculate AUROC.
    
    Args:
        directory_path: Path to directory
        verbose: Whether to print results
        
    Returns:
        dict: Results for each file
    """
    results = {}
    
    # Check if directory exists
    if not os.path.isdir(directory_path):
        if verbose:
            print(f"Error: '{directory_path}' is not a valid directory.")
        return results
    
    # Get list of CSV files
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    
    if not csv_files:
        if verbose:
            print(f"No CSV files found in {directory_path}")
        return results
    
    if verbose:
        print(f"Found {len(csv_files)} CSV files in {directory_path}")
        print(f"{'File Name':<40} {'AUROC':<10} {'Threshold':<10} {'Sensitivity':<12} {'Specificity':<12}")
        print("-" * 90)
    
    # Process each CSV file
    for file_name in csv_files:
        file_path = os.path.join(directory_path, file_name)
        stats, error = calculate_auroc_from_csv(file_path)
        
        if error:
            if verbose:
                print(f"{file_name:<40} Error: {error}")
        else:
            auroc = stats['auroc']
            threshold = stats['optimal_threshold']
            sensitivity = stats['sensitivity']
            specificity = stats['specificity']
            
            if verbose:
                print(f"{file_name:<40} {auroc:.4f}    {threshold:.4f}    {sensitivity:.4f}      {specificity:.4f}")
            results[file_name] = stats
    
    # Find and print the file with the highest AUROC
    if results and verbose:
        best_file = max(results.items(), key=lambda x: x[1]['auroc'])
        print("\nHighest AUROC:")
        print(f"File: {best_file[0]}, AUROC: {best_file[1]['auroc']:.4f}")
    
    return results