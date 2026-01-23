import os
import json
import math
import csv

def calculate_stats(data_list):
    """Calculates mean and 1.96 * standard deviation."""
    n = len(data_list)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return data_list[0], 0.0
    
    mean = sum(data_list) / n
    variance = sum((x - mean) ** 2 for x in data_list) / (n - 1)
    std_dev = math.sqrt(variance)
    
    return mean, 1.96 * std_dev

def analyze_eval_results(eval_dir, output_csv="evaluation_results.csv", limit_entries=None):
    results = []

    # Walk through the directory tree
    for root, dirs, files in os.walk(eval_dir):
        if 'results.json' in files:
            file_path = os.path.join(root, 'results.json')
            folder_name = os.path.basename(root)
            
            # Local metrics
            folder_successes = 0
            folder_failures = 0
            distances = []
            min_distances = []
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Get items as a list
                items = list(data if isinstance(data, list) else data.values())

                # Apply the limit here if specified
                if limit_entries is not None:
                    items = items[:limit_entries]

                for entry in items:
                    if not isinstance(entry, dict): continue
                        
                    # 1. Success Count
                    if 'success' in entry:
                        if entry['success'] is True:
                            folder_successes += 1
                        else:
                            folder_failures += 1
                    
                    # 2. Collect Distances
                    if 'distance_to_target' in entry:
                        distances.append(float(entry['distance_to_target']))
                    elif 'last_dist_to_target' in entry:
                        distances.append(float(entry['last_dist_to_target']))
                    elif 'last_dist' in entry:  
                        distances.append(float(entry['last_dist']))

                    # 3. Collect Min Distances
                    if 'min_dist_to_target' in entry:
                        min_distances.append(float(entry['min_dist_to_target']))

                # Perform Stats
                d_mean, d_err = calculate_stats(distances)
                md_mean, md_err = calculate_stats(min_distances)
                total = folder_successes + folder_failures
                success_rate = (folder_successes / total) if total > 0 else 0

                # Store result for CSV and Printing
                results.append({
                    "folder": folder_name,
                    "successes": folder_successes,
                    "failures": folder_failures,
                    "success_rate": success_rate,
                    "dist_mean": d_mean,
                    "dist_196std": d_err,
                    "min_dist_mean": md_mean,
                    "min_dist_196std": md_err
                })
                
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error processing {root}: {e}")

    # --- 1. Print to Console ---
    print(f"\n{'Folder':<25} | {'Succ/Fail':<10} | {'Dist (Mean ± 1.96σ)':<25} | {'Min Dist (Mean ± 1.96σ)':<25}")
    print("-" * 95)
    for r in results:
        print(f"{r['folder']:<25} | {r['successes']}/{r['failures']:<9} | "
              f"{r['dist_mean']:.4f} ± {r['dist_196std']:.4f} | "
              f"{r['min_dist_mean']:.4f} ± {r['min_dist_196std']:.4f}")

    # --- 2. Save to CSV ---
    if results:
        keys = results[0].keys()
        with open(output_csv, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
        print(f"\n[✔] Data successfully exported to {output_csv}")

if __name__ == "__main__":
    EVAL_DIR = "outputs/eval_sweep" 
    
    if os.path.exists(EVAL_DIR):
        # Pass the desired limit here (e.g., 100)
        analyze_eval_results(EVAL_DIR, limit_entries=500)
    else:
        print(f"Error: Path '{EVAL_DIR}' not found.")