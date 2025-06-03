# eval_policy.py
# TODO hard

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import logging
import os # For path joining

# Adjust imports based on your project structure
# Assuming 'src' is in your PYTHONPATH or you run from the project root
from src.models.policy_head.policy_network import create_model # From your project
# You will likely need preprocessing functions from here:
# from src.data_handling.processing import your_specific_preprocess_function
# from src.data_handling.dataset import YOUR_DATA_NORMALIZATION_CONSTANTS # if any

log = logging.getLogger(__name__)

def preprocess_inference_input(raw_input_data, cfg: DictConfig, device: torch.device):
    """
    Preprocesses a single raw input data point for inference.
    This function MUST replicate the transformations your Dataset applies to validation data.

    Args:
        raw_input_data: The raw data for a single inference sample.
                        This could be a dictionary, a path to a file, a numpy array, etc.
                        Example: {'point_cloud': np.array(...), 'robot_state': np.array(...)}
        cfg: The Hydra configuration object, useful for accessing normalization
             parameters or other preprocessing settings.
        device: The torch device to move tensors to.

    Returns:
        A dictionary of processed tensors, ready for the model and moved to the specified device.
        The keys should match what your model's forward() method expects.
        Each tensor should have a batch dimension of 1.
    """
    log.info("Preprocessing inference input...")
    processed_input = {}

    # --- THIS IS THE CRITICAL PART YOU NEED TO IMPLEMENT ---
    # Based on your `data_handling.dataset.create_dataloaders_from_config`
    # and the `Dataset` class used there.

    # Example: If your model expects a point cloud and a robot state:
    # 1. Load/access raw data parts:
    #    point_cloud_raw = raw_input_data['point_cloud'] # e.g., a (N, 3) numpy array
    #    robot_state_raw = raw_input_data['robot_state'] # e.g., a (D,) numpy array

    # 2. Convert to Tensors:
    #    pc_tensor = torch.from_numpy(point_cloud_raw).float()
    #    state_tensor = torch.from_numpy(robot_state_raw).float()

    # 3. Apply any normalization/transformations:
    #    (These should be the same as in your training pipeline)
    #    If you have normalization stats in cfg:
    #    if hasattr(cfg.dataset, 'normalization_stats'):
    #        mean = torch.tensor(cfg.dataset.normalization_stats.mean).float()
    #        std = torch.tensor(cfg.dataset.normalization_stats.std).float()
    #        state_tensor = (state_tensor - mean) / std

    # 4. Add batch dimension (since this is a single sample for inference):
    #    pc_tensor = pc_tensor.unsqueeze(0)    # Becomes (1, N, 3)
    #    state_tensor = state_tensor.unsqueeze(0) # Becomes (1, D)

    # 5. Assemble into the dictionary format your model expects:
    #    (Check your model's forward() method and the batch structure in train_policy.py)
    #    processed_input['point_cloud'] = pc_tensor.to(device)
    #    processed_input['robot_state'] = state_tensor.to(device)

    # --- Placeholder for model input structure (ADAPT THIS!) ---
    # This is a generic example. Your `create_model` and the model's `forward` method define
    # the actual expected input structure (e.g., dictionary keys, tensor names).
    # Refer to `log.info(f"Batch shapes:")` in your `train_policy.py` for hints on the expected keys and shapes.
    if hasattr(cfg.model, 'input_features_eval'): # You might define this in your config
        for key, shape_without_batch in cfg.model.input_features_eval.items():
            batch_shape = [1] + list(shape_without_batch)
            # This creates dummy data, replace with your actual preprocessed data
            log.warning(f"Creating DUMMY input for '{key}'. REPLACE with actual preprocessing.")
            processed_input[key] = torch.randn(batch_shape, device=device)
    else:
        # A very generic fallback if your model takes a single tensor named 'input_tensor'
        # input_dim = cfg.model.get('some_input_dim_config', 128) # Get this from your actual model config
        # log.warning(f"Creating DUMMY 'input_tensor'. REPLACE with actual preprocessing.")
        # processed_input['input_tensor'] = torch.randn(1, input_dim, device=device)
        log.error(f"Could not determine input structure for preprocessing. Please implement `preprocess_inference_input` based on your model's requirements and training data.")
        raise NotImplementedError("`preprocess_inference_input` needs to be implemented with actual data loading and preprocessing logic.")

    log.info(f"Processed input data keys: {list(processed_input.keys())}")
    for key, value in processed_input.items():
        log.info(f"  {key} shape: {value.shape}, device: {value.device}")
    return processed_input


@hydra.main(config_path="configs", config_name="config", version_base=None)
def eval_policy(cfg: DictConfig) -> None:
    """
    Main evaluation/inference function for the manipulation policy.
    Args:
        cfg: Hydra configuration object.
    """
    log.info("Starting policy evaluation/inference...")
    log.info(f"Using experiment config: {cfg.experiment_name}")
    # log.info(f"Full config: {OmegaConf.to_yaml(cfg)}") # For debugging

    # --- 1. Setup Device ---
    default_device_str = "cuda" if torch.cuda.is_available() else "cpu"
    # Allow overriding device via inference config or command line
    device_str = cfg.get("inference", {}).get("device", default_device_str)
    device = torch.device(device_str)
    log.info(f"Using device: {device}")

    # --- 2. Initialize Model ---
    log.info("Initializing model...")
    # The model architecture is defined by cfg.model
    model = create_model(cfg.model).to(device)
    log.info(f"Model created: {type(model).__name__}")

    # --- 3. Load Trained Weights ---
    # Determine checkpoint path.
    # Option 1: Directly in config (e.g., cfg.inference.checkpoint_path)
    # Option 2: Construct from training output directory (more robust if following a pattern)
    checkpoint_path_cfg = cfg.get("inference", {}).get("checkpoint_path", None)

    if checkpoint_path_cfg is None:
        # Try to infer from hydra's output directory if not specified
        # This assumes your train_policy.py saves checkpoints in a standard location
        # within its hydra run directory.
        # Example: <hydra_run_dir>/checkpoints/best_model.pth or <hydra_run_dir>/checkpoints/epoch_X.pth
        # You might need to adjust this logic based on how your Trainer saves models.
        # For now, let's require it in the config.
        log.error("Checkpoint path not found. Please specify `inference.checkpoint_path` in your config or command line.")
        log.error("Example: python eval_policy.py inference.checkpoint_path=outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/model_epoch_N.pth")
        return

    # Ensure the path is absolute or correctly relative if hydra changes CWD
    if not os.path.isabs(checkpoint_path_cfg) and hydra.utils.get_original_cwd() != os.getcwd():
        checkpoint_path = os.path.join(hydra.utils.get_original_cwd(), checkpoint_path_cfg)
    else:
        checkpoint_path = checkpoint_path_cfg

    log.info(f"Loading model checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        log.error(f"Checkpoint file not found at {checkpoint_path}")
        log.error(f"Original CWD: {hydra.utils.get_original_cwd()}, Current CWD: {os.getcwd()}")
        return

    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        # Handle potential 'module.' prefix if DataParallel was used during training
        if any(key.startswith('module.') for key in state_dict.keys()):
            log.info("Removing 'module.' prefix from state_dict keys (model likely trained with DataParallel).")
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        log.info("Model weights loaded successfully.")
    except Exception as e:
        log.error(f"Error loading checkpoint: {e}")
        raise

    model.eval() # IMPORTANT: Set the model to evaluation mode
    log.info("Model set to evaluation mode.")

    # --- 4. Prepare Input Data for Inference ---
    # This is where you'd get your *single* piece of data for inference.
    # For example, load a point cloud from a file, get current robot state, etc.
    # This `raw_inference_data` needs to be structured in a way that
    # `preprocess_inference_input` can understand it.
    #
    # EXAMPLE: Replace this with your actual data source
    log.warning("USING DUMMY RAW INPUT DATA. Replace with your actual data source.")
    example_num_points = cfg.model.get("num_points", 1024) # Example: if your model uses a fixed number of points
    example_point_dim = cfg.model.get("point_feature_dim", 3) # Example
    example_state_dim = cfg.model.get("state_dim", 6) # Example
    raw_inference_data = {
        'point_cloud': np.random.rand(example_num_points, example_point_dim).astype(np.float32),
        'robot_state': np.random.rand(example_state_dim).astype(np.float32)
        # Add other raw data parts your `preprocess_inference_input` expects
    }
    # --- End Example Data ---

    # Preprocess the raw data
    # The `cfg` object is passed in case preprocessing needs access to config values
    # (e.g., normalization stats, image sizes, etc.)
    input_for_model = preprocess_inference_input(raw_inference_data, cfg, device)

    if not input_for_model:
        log.error("Preprocessing did not return any data. Aborting.")
        return

    # --- 5. Perform Inference ---
    log.info("Running model inference...")
    with torch.no_grad(): # Disable gradient calculations
        # The structure of input_for_model (e.g. dictionary, single tensor)
        # must match what your model's forward() method expects.
        # If your model's forward method is `def forward(self, observation, state):`
        # then you would call it as `output = model(observation=input_for_model['observation'], state=input_for_model['state'])`
        # or if it's `def forward(self, batch_dict):` then `output = model(input_for_model)`
        #
        # Check your model's `forward` signature in `src/models/policy_head/policy_network.py`
        # Let's assume it takes a dictionary, similar to your training loop.
        try:
            output = model(input_for_model)
        except Exception as e:
            log.error(f"Error during model forward pass: {e}")
            log.error("Ensure the `input_for_model` structure and tensor shapes/types match your model's `forward` method.")
            log.error(f"Input keys: {input_for_model.keys()}")
            for k, v in input_for_model.items():
                if isinstance(v, torch.Tensor):
                    log.error(f"  {k}: shape {v.shape}, dtype {v.dtype}, device {v.device}")
            raise

    log.info(f"Inference output raw: {output}")
    if isinstance(output, torch.Tensor):
        log.info(f"Output tensor shape: {output.shape}")
    elif isinstance(output, dict):
        log.info(f"Output dictionary keys: {output.keys()}")
        for k,v in output.items():
            if isinstance(v, torch.Tensor):
                log.info(f"  {k} shape: {v.shape}")


    # --- 6. Post-process Output (Optional) ---
    # Depending on your task, you might need to convert the output.
    # E.g., apply softmax, convert to specific actions, denormalize, etc.
    # final_action = postprocess_model_output(output, cfg)
    # log.info(f"Post-processed action: {final_action}")

    log.info("Policy evaluation/inference finished.")


if __name__ == "__main__":
    # Optional: Basic logging setup if Hydra doesn't handle it to your liking for standalone script runs
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    eval_policy()