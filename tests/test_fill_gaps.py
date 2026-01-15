import h5py
import re

def sort_key(name):
    """Extracts the number from 'demo_idx_N' for proper sorting."""
    match = re.search(r'(\d+)', name)
    return int(match.group(1)) if match else -1

def process_h5(input_file, output_file, log_file):
    with h5py.File(input_file, 'r') as src, \
         h5py.File(output_file, 'w') as dst, \
         open(log_file, 'w') as log:
        
        # 1. Get and sort existing keys
        keys = sorted(src.keys(), key=sort_key)
        
        missing_indices = []
        current_new_idx = 0
        expected_idx = 0

        for key in keys:
            actual_idx = sort_key(key)
            
            # 2. Check for gaps
            while expected_idx < actual_idx:
                missing_indices.append(expected_idx)
                expected_idx += 1
            
            # 3. Copy data to the new structure: demo_N
            new_name = f"demo_{current_new_idx}"
            src.copy(key, dst, name=new_name)
            
            print(f"Renamed: {key} -> {new_name}")
            
            current_new_idx += 1
            expected_idx += 1

        # 4. Save the gaps to a text file
        if missing_indices:
            log.write("Gaps found at original indices:\n")
            for idx in missing_indices:
                log.write(f"{idx}\n")
        else:
            log.write("No gaps detected.")

        print(f"\nProcessing complete!")
        print(f"Gaps logged to: {log_file}")

# Usage
process_h5('shelf_demos/shelf_demo.h5', 'cleaned_data.h5', 'missing_indices.txt')