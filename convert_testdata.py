#!/usr/bin/env python3
"""
Script to convert TestDomain_Inputs text files to Python NumPy arrays for the PriorityFlow package.
"""

import numpy as np
import os

def convert_testdata_to_numpy():
    """Convert TestDomain_Inputs text files to Python NumPy arrays."""
    
    # Define paths
    input_dir = "/home/ga6/workspace/PriorityFlow/TestDomain_Inputs"
    output_dir = "/home/ga6/workspace/priority_flow/src/priority_flow/data"
    
    # List of data files to convert
    data_files = {
        'dem_test.txt': 'DEM',
        'mask_test.txt': 'watershed_mask', 
        'river_mask_test.txt': 'river_mask'
    }
    
    for input_file, output_name in data_files.items():
        print(f"Converting {input_file}...")
        
        input_path = os.path.join(input_dir, input_file)
        output_path = os.path.join(output_dir, f"{output_name}.npy")
        
        # Read the text file
        with open(input_path, 'r') as f:
            lines = f.readlines()
        
        # Parse each line as space-separated values
        data_rows = []
        for line in lines:
            # Split by whitespace and convert to float
            values = [float(x) for x in line.strip().split()]
            data_rows.append(values)
        
        # Convert to numpy array
        np_data = np.array(data_rows)
        
        # Save as .npy file
        np.save(output_path, np_data)
        
        print(f"  Saved {output_name} with shape {np_data.shape} to {output_path}")
        
        # Also save as .npz for metadata
        npz_path = os.path.join(output_dir, f"{output_name}.npz")
        np.savez(npz_path, data=np_data, name=output_name, 
                description=f"Converted from {input_file}")
        
        print(f"  Also saved as {npz_path}")

if __name__ == "__main__":
    convert_testdata_to_numpy()
