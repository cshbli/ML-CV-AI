import numpy as np
import argparse
import os
from gguf import GGUFReader
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Extract tensor weights from GGUF model files")
    parser.add_argument("--model", "-m", required=True, help="Path to the GGUF model file")
    parser.add_argument("--tensor", "-t", help="Name of tensor to extract. If not provided, lists all tensors")
    parser.add_argument("--index", "-i", type=int, help="Index of tensor to extract (alternative to name)")
    parser.add_argument("--output", "-o", help="Output file path (default: tensor_name.npy)")
    parser.add_argument("--extract-all", "-a", action="store_true", help="Extract all tensors from the model")
    parser.add_argument("--output-dir", "-d", default=".", help="Directory to save extracted tensors (default: current directory)")
    
    return parser.parse_args()

# Extract a specific tensor from a GGUF file
def extract_tensor(gguf_file_path, tensor_name, output_file=None):
    """
    Extract a specific tensor from a GGUF file and optionally save to a file
    
    Parameters:
    - gguf_file_path: Path to the GGUF file
    - tensor_name: Name of the tensor to extract
    - output_file: Path to save the tensor data (optional)
    
    Returns:
    - The tensor data as a numpy array
    """
    # Load the GGUF file
    reader = GGUFReader(gguf_file_path)
    
    # Find the tensor by name
    target_tensor = None
    for tensor in reader.tensors:
        if tensor.name == tensor_name:
            target_tensor = tensor
            break
    
    if target_tensor is None:
        raise ValueError(f"Tensor '{tensor_name}' not found in the model")
    
    # Get the tensor data
    tensor_data = target_tensor.data
    
    # Save to file if requested
    if output_file:
        np.save(output_file, tensor_data)
        print(f"Tensor saved to {output_file}")
    
    return tensor_data    

# Extract all tensors
def extract_all_tensors(gguf_file_path, output_dir="."):
    """
    Extract all tensors from a GGUF file and save each to a separate .npy file
    
    Parameters:
    - gguf_file_path: Path to the GGUF file
    - output_dir: Directory to save the extracted tensors
    
    Returns:
    - Dictionary mapping tensor names to their saved file paths
    """
    # Load the GGUF file
    reader = GGUFReader(gguf_file_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract and save all tensors
    saved_files = {}
    total_tensors = len(reader.tensors)
    
    print(f"Extracting {total_tensors} tensors from {gguf_file_path}")
    
    for i, tensor in enumerate(reader.tensors):
        # Create a safe filename
        safe_name = tensor.name.replace('/', '_').replace('\\', '_')
        output_file = os.path.join(output_dir, f"{safe_name}.npy")
        
        # Get tensor data
        tensor_data = tensor.data
        
        # Save tensor
        np.save(output_file, tensor_data)
        saved_files[tensor.name] = output_file
        
        # Print progress
        print(f"[{i+1}/{total_tensors}] Extracted: {tensor.name} (Shape: {tensor.shape}, Type: {tensor_data.dtype}) -> {output_file}")
    
    print(f"All tensors extracted to {output_dir}")
    return saved_files


def list_tensors(model_path):
    """List all tensors in the model"""
    reader = GGUFReader(model_path)
    print("Available tensors:")
    for i, tensor in enumerate(reader.tensors):
        print(f"{i}: {tensor.name} (Shape: {tensor.shape})")
    return reader.tensors


# Example usage
if __name__ == '__main__':
    args = parse_args()

    if args.extract_all:
        # Extract all tensors
        extract_all_tensors(args.model, args.output_dir)
    else:
        # List all tensors if no specific tensor is requested
        tensors = list_tensors(args.model)
        
        if args.tensor is None and args.index is None:
            # Just list tensors and exit
            exit(0)
        
        # Determine which tensor to extract
        if args.index is not None:
            if args.index < 0 or args.index >= len(tensors):
                print(f"Error: Index {args.index} out of range (0-{len(tensors)-1})")
                exit(1)
            tensor_name = tensors[args.index].name
        else:
            tensor_name = args.tensor
        
        # Determine output file name
        if args.output:
            output_file = args.output
        else:
            # Create a safe filename
            safe_name = tensor_name.replace('/', '_').replace('\\', '_')
            output_file = f"{safe_name}.npy"
        
        # Extract the tensor
        tensor_data = extract_tensor(args.model, tensor_name, output_file)
        
        # Print information about the extracted tensor
        print(f"Extracted tensor: {tensor_name}")
        print(f"Shape: {tensor_data.shape}")
        print(f"Data type: {tensor_data.dtype}")
        print(f"Sample values: {tensor_data.flatten()[:5]}")  # Show first 5 values