#!/bin/bash

# Array of config names you want to evaluate
# Add or remove names based on your .yaml files in ../configs
CONFIGS=(
    # "inference_table_transformer_depth"
    # "inference_table_transformer_rgb"
    # "inference_table_transformer_dino"
    # "inference_table_diffusion_depth"
    # "inference_table_diffusion_rgb"
    # "inference_table_diffusion_dino"

    "inference_shelf_transformer_depth"
    # "inference_shelf_transformer_rgb"
    # "inference_shelf_transformer_dino"
    # "inference_shelf_diffusion_depth"
    # "inference_shelf_diffusion_rgb"
    # "inference_shelf_diffusion_dino"
)



echo "Starting evaluation sweep..."

for CONFIG in "${CONFIGS[@]}"
do
    echo "------------------------------------------------"
    echo "Running evaluation for: $CONFIG"
    echo "------------------------------------------------"

    # Run the python script with Hydra overrides
    # 'hydra.run.dir' creates a unique folder for each run based on the config name
    python scripts/eval_policy_alt1.py \
        --config-name="$CONFIG" \
        hydra.run.dir="outputs/eval_sweep/${CONFIG}_$(date +%Y%m%d_%H%M%S)"

    echo "Finished $CONFIG"
done

echo "All evaluations complete!"