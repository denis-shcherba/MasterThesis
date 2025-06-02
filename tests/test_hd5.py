import h5py
import numpy as np

def print_all_arrays(h5file):
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"\nDataset: {name}")
            print(f"Shape: {obj.shape}, Dtype: {obj.dtype}")
            try:
                data = obj[()]  # Load the dataset into a NumPy array
                print("Data:")
                #print(data)
            except Exception as e:
                print(f"Could not load data: {e}")
    
    with h5py.File(h5file, "r") as f:
        f.visititems(visitor)

# Usage
print_all_arrays("variable_demo.h5")
