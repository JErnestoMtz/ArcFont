"""
Simple example of using ArcFont model to generate font embeddings.
"""

import os
import sys
import numpy as np

# Add parent directory to path to import arcfont when running from this directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from arcfont import load_model, get_image_embedding, get_embeddings_batch

def process_single_image(model_path, image_path):
    """Example of processing a single image."""
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded successfully!")
    
    print(f"Processing image: {image_path}")
    embedding = get_image_embedding(image_path, model)
    
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    
    return embedding

def process_batch(model_path, image_paths):
    """Example of processing multiple images."""
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded successfully!")
    
    print(f"Processing {len(image_paths)} images...")
    embeddings = get_embeddings_batch(image_paths, model, batch_size=16)
    
    print(f"Embeddings shape: {embeddings.shape}")
    for i, emb in enumerate(embeddings):
        print(f"Image {i+1} embedding norm: {np.linalg.norm(emb):.4f}")
    
    return embeddings

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ArcFont example usage')
    parser.add_argument('--model', type=str, default='arcfont_standalone.pt',
                        help='Path to the model file')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the image file (or directory for batch mode)')
    parser.add_argument('--batch', action='store_true',
                        help='Process all images in the directory')
    
    args = parser.parse_args()
    
    if args.batch:
        # Process all images in directory
        if not os.path.isdir(args.image):
            print(f"Error: {args.image} is not a directory")
            sys.exit(1)
            
        image_files = []
        for filename in os.listdir(args.image):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_files.append(os.path.join(args.image, filename))
        
        if not image_files:
            print(f"No image files found in {args.image}")
            sys.exit(1)
            
        print(f"Found {len(image_files)} images")
        embeddings = process_batch(args.model, image_files)
    else:
        # Process a single image
        if not os.path.isfile(args.image):
            print(f"Error: {args.image} is not a file")
            sys.exit(1)
            
        embedding = process_single_image(args.model, args.image) 