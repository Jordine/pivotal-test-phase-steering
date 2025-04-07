"""
Functions for extracting features using the Goodfire API.
"""
import goodfire
import os
import json

def setup_goodfire_client(api_key=None):
    """
    Set up Goodfire client.
    
    Args:
        api_key: Goodfire API key (uses env var if None)
        
    Returns:
        tuple: (client, variant)
    """
    if api_key is None:
        api_key = os.environ.get("GOODFIRE_API_KEY")
        if not api_key:
            raise ValueError("GOODFIRE_API_KEY environment variable not set")
    
    client = goodfire.Client(api_key)
    variant = goodfire.Variant("meta-llama/Llama-3.3-70B-Instruct")
    
    return client, variant

def extract_feature(client, variant, features_list, mode="search"):
    """
    Extract features using Goodfire API.
    
    Args:
        client: Goodfire client
        variant: Model variant
        features_list: Dictionary of feature prompts and indices
        mode: "search" or "contrast"
        
    Returns:
        list: Extracted features
    """
    features = []
    variant.reset()

    if mode == "contrast":
        for feature_prompt in features_list:
            for index in features_list[feature_prompt]:
                _, test_features = client.features.contrast(
                    dataset_1=[
                        [
                            {
                                "role": "user",
                                "content": "Hello how are you?"
                            },
                            {
                                "role": "assistant",
                                "content": "I am a helpful assistant. How can I help you?"
                            }
                        ]
                    ],
                    dataset_2=[
                        [
                            {
                                "role": "user",
                                "content": "Hello how are you?"
                            },
                            {
                                "role": "user",
                                "content": f"{feature_prompt}"
                            }
                        ],
                    ],
                    model=variant,
                    top_k=10
                )
                features.append(test_features[index])

    elif mode == "search":
        for feature_prompt in features_list:
            for index in features_list[feature_prompt]:
                test_features = client.features.search(
                    f"{feature_prompt}",
                    model=variant,
                    top_k=10
                )
                features.append(test_features[index])
    
    return features

def save_features(features, output_path):
    """
    Save extracted features to JSON file.
    
    Args:
        features: List of features
        output_path: Output file path
    """
    with open(output_path, "w") as f:
        json.dump([str(feature) for feature in features], f, indent=2)

def load_features(features_path):
    """
    Load features from JSON file.
    
    Args:
        features_path: Path to features JSON
        
    Returns:
        list: List of features
    """
    with open(features_path, "r") as f:
        features_str = json.load(f)
  
    return features_str
