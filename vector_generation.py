"""
Functions for generating steering vectors by contrasting test vs deployment behaviors.
"""
import os
import gc
import torch
from datetime import datetime
from tqdm import tqdm

def extract_activations_all_layers(model, tokens, token_type, layers, tokenizer=None):
    """
    Extract activations for specific token positions across multiple layers.
    
    Args:
        model: HookedTransformer model
        tokens: Tokenized input
        token_type: Token type to extract (e.g., 'A' for the (A) token)
        layers: List of layer indices to extract from
        tokenizer: Optional tokenizer (uses model's tokenizer if None)
        
    Returns:
        dict: Dictionary mapping layer indices to activation tensors
    """
    cached_activations = {}

    def make_hook(layer):
        def hook_fn(activation, hook):
            token_positions = (tokens == tokenizer.convert_tokens_to_ids(token_type)).nonzero()[-1]
            last_pos = token_positions[-1]
            print(f"Extracting activation for token '{token_type}' at position {last_pos} (layer {layer})")
            cached_activations[layer] = activation[:, last_pos, :].clone().detach()
            return activation
        return hook_fn

    # Prepare fwd_hooks for all layers at once
    fwd_hooks = []
    for layer in layers:
        hook = make_hook(layer)
        hook_name = f"blocks.{layer}.hook_resid_pre"
        fwd_hooks.append((hook_name, hook))

    model.reset_hooks()
    with model.hooks(fwd_hooks=fwd_hooks):
        with torch.no_grad():
            model(tokens)

    return cached_activations

def generate_steering_vectors_all_layers(model, dataset, layers):
    """
    Generate steering vectors by contrasting test vs deployment activations.
    
    Args:
        model: HookedTransformer model
        dataset: List of dictionaries with 'question', 'test', 'deploy' keys
        layers: List of layer indices
        
    Returns:
        dict: Dictionary mapping layer indices to steering vectors
    """
    # Initialize per-layer accumulators
    test_acts_by_layer = {layer: [] for layer in layers}
    deploy_acts_by_layer = {layer: [] for layer in layers}

    for item in dataset:
        question = item["question"]

        test_answer = item['test']
        deploy_answer = item['deploy']
        
        # Build test and deploy prompts with a space after newline to avoid tokenization issues
        prompt = question
        test_prompt = f"{prompt}\n\n {test_answer}"
        deploy_prompt = f"{prompt}\n\n {deploy_answer}"
        
        # Determine token type (assumed to be the character at index 1 of the answer string, e.g. "(A)" -> "A")
        test_token_type = test_answer[1]
        deploy_token_type = deploy_answer[1]
        
        # Tokenize the prompts once each
        test_tokens = model.to_tokens(test_prompt)
        deploy_tokens = model.to_tokens(deploy_prompt)
        
        # For test prompt: extract activations for all layers at once
        model.reset_hooks()
        test_activations = extract_activations_all_layers(model, test_tokens, test_token_type, layers)
        
        model.reset_hooks()
        deploy_activations = extract_activations_all_layers(model, deploy_tokens, deploy_token_type, layers)
        
        # Accumulate activations per layer
        for layer in layers:
            if layer in test_activations:
                test_acts_by_layer[layer].append(test_activations[layer])
            else:
                print(f"Warning: Test activation missing for layer {layer} in example.")
            if layer in deploy_activations:
                deploy_acts_by_layer[layer].append(deploy_activations[layer])
            else:
                print(f"Warning: Deploy activation missing for layer {layer} in example.")
        
        # Clean up tokens and activations
        del test_tokens, deploy_tokens, test_activations, deploy_activations
        gc.collect()

    # Compute steering vector per layer (mean test activation - mean deploy activation)
    steering_vectors = {}
    for layer in layers:
        if test_acts_by_layer[layer] and deploy_acts_by_layer[layer]:
            test_mean = torch.stack(test_acts_by_layer[layer]).mean(dim=0)
            deploy_mean = torch.stack(deploy_acts_by_layer[layer]).mean(dim=0)
            steering_vectors[layer] = test_mean - deploy_mean
        else:
            print(f"Layer {layer} has missing activations; skipping.")
            steering_vectors[layer] = None

    print("\n----- Steering vector generation DONE for all layers -----\n")
    return steering_vectors

def normalize_vector(vector):
    """
    Normalize a vector to unit length.
    
    Args:
        vector: Input tensor
        
    Returns:
        torch.Tensor: Normalized vector
    """
    return vector * (1.0 / vector.norm())

def generate_all_layers(model, dataset, layers, output_dirs):
    """
    Generate steering vectors for all specified layers and save them.
    
    Args:
        model: HookedTransformer model
        dataset: Dataset of examples
        layers: List of layer indices
        output_dirs: Dictionary of directories for saving outputs
        
    Returns:
        tuple: (raw_vectors, normalized_vectors)
    """
    raw_vectors = generate_steering_vectors_all_layers(model, dataset, layers)
    normalized_vectors = {}
    
    for layer, vector in raw_vectors.items():
        if vector is not None:
            normalized_vectors[layer] = normalize_vector(vector)
        else:
            normalized_vectors[layer] = None
        
        # Save raw vector
        raw_path = os.path.join(output_dirs['vectors'], f"layer_{layer}.pt")
        torch.save(vector, raw_path)
        # Save normalized vector
        norm_path = os.path.join(output_dirs['normalized_vectors'], f"layer_{layer}.pt")
        torch.save(normalized_vectors[layer], norm_path)
        print(f"Saved layer {layer}: raw vector to {raw_path} and normalized vector to {norm_path}")
    
    return raw_vectors, normalized_vectors

def setup_output_dirs(experiment_name=None):
    """
    Set up output directories for an experiment.
    
    Args:
        experiment_name: Optional name for the experiment
        
    Returns:
        dict: Dictionary of directory paths
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if experiment_name is None:
        base_dir = f"LLAMA_70B_CONTRASTIVE_{timestamp}"
    else:
        base_dir = f"{experiment_name}_{timestamp}"
    
    # Create directories
    dirs = {
        'base': base_dir,
        'vectors': os.path.join(base_dir, 'vectors'),
        'normalized_vectors': os.path.join(base_dir, 'normalized_vectors'),
        'results': os.path.join(base_dir, 'results')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return dirs

def run_experiment(model, dataset, layers=None, output_dir=None):
    """
    Run a complete steering vector generation experiment.
    
    Args:
        model: HookedTransformer model
        dataset: Dataset of contrastive examples
        layers: Optional list of layers (defaults to all)
        output_dir: Optional output directory name
        
    Returns:
        tuple: (output_dirs, raw_vectors, normalized_vectors)
    """
    output_dirs = setup_output_dirs(output_dir)
   
    with open(os.path.join(output_dirs['base'], 'experiment_info.txt'), 'w') as f:
        f.write(f"Model: {model.cfg.model_name}\n")
        f.write(f"Number of layers: {model.cfg.n_layers}\n")
        f.write(f"Training dataset size: {len(dataset)}\n")

    if layers is None:
        layers = range(model.cfg.n_layers)
        
    raw_vectors, normalized_vectors = generate_all_layers(model, dataset, layers, output_dirs)

    print("Extracted all vectors")
    
    return output_dirs, raw_vectors, normalized_vectors