"""
Functions for preparing datasets for sandbagging detection.
"""
import os
from datasets import load_dataset
import pandas as pd
import json

def reformat_data(dataset, type_label, phase_prompt=""):
    """
    Reformat dataset for sandbagging detection.
    
    Args:
        dataset: HuggingFace dataset
        type_label: Label for the dataset (e.g., "bio", "chem")
        phase_prompt: Optional prompt to insert before "Answer:"
        
    Returns:
        list: Reformatted dataset
    """
    new_data = []
    answer_mapping = {0: "(A)", 1: "(B)", 2: "(C)", 3: "(D)"}

    for item in dataset['test']:
        letters = ['A', 'B', 'C', 'D']
        choices_text = '\n '.join(f'({letter}) {item}' for letter, item in zip(letters, item["choices"]))

        message = f"""{item['question']}
Choices:
 {choices_text}
{phase_prompt}Answer:"""
        correct_answer = answer_mapping.get(item["answer"])

        new_data.append({
            "question": message,
            "correct_answer": correct_answer,
            "type": type_label
        })
    return new_data

def load_wmdp_subset(deploy_prompt="", size_bio=347, size_chem=111, size_cyber=542):
    """
    Load and prepare WMDP dataset subset.
    
    Args:
        deploy_prompt: Optional prompt for deployment context
        size_bio: Number of biology examples
        size_chem: Number of chemistry examples
        size_cyber: Number of cybersecurity examples
        
    Returns:
        tuple: (test_data, deploy_data)
    """
    # Load datasets
    ds_bio = load_dataset("cais/wmdp", "wmdp-bio")
    ds_chem = load_dataset("cais/wmdp", "wmdp-chem")
    ds_cyber = load_dataset("cais/wmdp", "wmdp-cyber")
    
    # Prepare standard test data
    data_bio = reformat_data(ds_bio, "bio", "")
    data_chem = reformat_data(ds_chem, "chem", "")
    data_cyber = reformat_data(ds_cyber, "cyber", "")
    
    # Prepare deployment data
    data_bio_deploy = reformat_data(ds_bio, "bio", deploy_prompt)
    data_chem_deploy = reformat_data(ds_chem, "chem", deploy_prompt)
    data_cyber_deploy = reformat_data(ds_cyber, "cyber", deploy_prompt)
    
    # Create subsets
    test_data = []
    test_data += data_bio[:size_bio]
    test_data += data_chem[:size_chem]
    test_data += data_cyber[:size_cyber]
    
    deploy_data = []
    deploy_data += data_bio_deploy[:size_bio]
    deploy_data += data_chem_deploy[:size_chem]
    deploy_data += data_cyber_deploy[:size_cyber]
    
    return test_data, deploy_data

def save_datasets(test_data, deploy_data, output_dir):
    """
    Save datasets to JSON files.
    
    Args:
        test_data: Test dataset
        deploy_data: Deployment dataset
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "test_data.json"), "w") as f:
        json.dump(test_data, f, indent=2)
    
    with open(os.path.join(output_dir, "deploy_data.json"), "w") as f:
        json.dump(deploy_data, f, indent=2)
