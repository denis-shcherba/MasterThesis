# scripts/test_validation_loss.py

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import logging
import os
from tqdm import tqdm
import gymnasium as gym
import envs.env  # noqa: F401 
# --- Assumed imports from your project structure ---
# You might need to adjust these paths based on your file organization
from models.policy_head.policy_network import create_model
from data_handling.dataset import create_dataloaders_from_config 
import yaml
from utils.data_utils import  denormalize_actions
import robotic as ry

log = logging.getLogger(__name__)

SHOW_RAI = True

def show_data_against_prediction_rai(target_action_seq, predicted_action_seq):
    env = gym.make("ShelfEnv-v0")
    obs, info = env.reset()
    

    with open("/home/denis/git/MasterThesis/outputs/final_outputs/normalization_stats_5000.yaml", 'r') as file:
        # todo change to config
        normalization_stats = yaml.safe_load(file)
    target_action_seq = denormalize_actions(target_action_seq, normalization_stats["action_stats"])
    previous_action_seq = denormalize_actions(predicted_action_seq, normalization_stats["action_stats"])


    for i in range(target_action_seq.shape[0]):
        target_pos = target_action_seq.squeeze(0).cpu().numpy()[i]
        predicted_pos = previous_action_seq.squeeze(0).cpu().numpy()[i]

        env.unwrapped.C.addFrame(f"target_pos_{i}").setPosition(target_pos).setShape(ry.ST.sphere, [.012]).setColor([.1*i,1-.1*i,1])
        env.unwrapped.C.addFrame(f"predicted_pos{i}").setPosition(predicted_pos).setShape(ry.ST.box, [.02, .02, .02]).setColor([1-.1*i,.1*i,.2])
        print("Target Position:", target_pos)
        print("Predicted Position:", predicted_pos)

    env.unwrapped.C.view(True)


        # for j in range(target_action_seq.shape[0]):
        #     env.unwrapped.C.delFrame(f"prev_gripper{j}")
        # for j in range(target_action_seq.shape[0]):
        #     env.unwrapped.C.delFrame(f"gripper{j}")

@hydra.main(config_path="../configs", config_name="test_validation", version_base=None)
def calculate_validation_loss(cfg: DictConfig) -> None:
    """
    Loads a model from a checkpoint and computes the average loss 
    over the entire validation dataset.
    """
    log.info("Starting validation loss calculation...")
    log.info(f"Using experiment config: {cfg.experiment_name}")

    # 1. Setup Device
    device_str = cfg.get("inference", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    log.info(f"Using device: {device}")
    
    # Set seed for deterministic dataloader shuffling (if any)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # 2. Create Validation Dataloader
    log.info("Creating dataloaders...")
    # We only need the validation loader for this task
    _, val_loader = create_dataloaders_from_config(cfg)
    log.info(f"Validation dataset contains {len(val_loader.dataset)} samples.")

    # 3. Initialize Model
    log.info("Initializing model...")
    model = create_model(cfg.model).to(device)

    # 4. Load Model Checkpoint
    checkpoint_path_cfg = cfg.get("inference", {}).get("checkpoint_path", None)
    if not checkpoint_path_cfg:
        log.error("Checkpoint path ('inference.checkpoint_path') not specified in the config.")
        raise ValueError("Checkpoint path is required.")

    # Resolve path relative to the original working directory if needed
    if not os.path.isabs(checkpoint_path_cfg) and hydra.utils.get_original_cwd() != os.getcwd():
        checkpoint_path = os.path.join(hydra.utils.get_original_cwd(), checkpoint_path_cfg)
    else:
        checkpoint_path = checkpoint_path_cfg

    if not os.path.exists(checkpoint_path):
        log.error(f"Checkpoint file not found at {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    log.info(f"Loading checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)['model_state_dict']
    
    # Handle DataParallel prefix if it exists
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    log.info("Model weights loaded successfully.")

    # 5. Define Loss Function
    # Assuming Mean Squared Error, which is common for trajectory prediction.
    # Change this if your training script uses a different loss (e.g., L1Loss).
    loss_fn = torch.nn.MSELoss()
    log.info(f"Using loss function: {type(loss_fn).__name__}")

    # 6. Evaluation Loop
    model.eval()  # Set the model to evaluation mode (disables dropout, etc.)
    total_loss = 0.0
    
    log.info("Starting evaluation over the validation set...")
    with torch.no_grad():
        # Outer loop iterates through batches from the dataloader
        i = 0
        for batch in tqdm(val_loader, desc="Validating"):
            # Get the full batch tensors from the dataloader
            # We append '_batch' to clarify these are multi-item tensors
            context_depth_batch = batch['observation_sequence'].to(device)
            context_state_batch = batch['previous_actions_sequence'].to(device)
            target_actions_batch = batch['target_actions_sequence'].to(device)
            
            # Get the number of items in the current batch (usually cfg.train.batch_size, except for the last batch)
            current_batch_size = context_depth_batch.size(0)

            # Inner loop to process each sample in the batch individually
            for i in range(current_batch_size):
                # --- Select the i-th sample from the batch ---
                # .unsqueeze(0) adds a batch dimension of size 1, so the shape becomes [1, ...]
                # This is required by the model.
                depth_single = context_depth_batch[i].unsqueeze(0)
                state_single = context_state_batch[i].unsqueeze(0)
                target_single = target_actions_batch[i].unsqueeze(0)

                # Forward pass: get model prediction for the single sample
                prediction_single = model(depth_single, state_single)
                
                if SHOW_RAI:
                    pass
                    show_data_against_prediction_rai(target_single.squeeze(0).cpu(), prediction_single.squeeze(0).cpu())
                # Calculate the loss for this single prediction vs. its single target
                loss = loss_fn(prediction_single, target_single)
                
                # Accumulate the loss for each individual sample
                total_loss += loss.item()


            i+= 1
            if i >= 10:  # For quicker testing, limit to first 10 batches
                break

    # 7. Calculate and Report Average Loss
    # The average loss is the total accumulated loss divided by the number of batches
    avg_loss = total_loss / len(val_loader)
    
    log.info("--- 📊 Validation Complete ---")
    log.info(f"Total Batches: {len(val_loader)}")
    log.info(f"Average Validation Loss (MSE): {avg_loss:.6f}")
    
    # You could also compute other metrics here, like Mean Absolute Error (MAE)
    # or endpoint error if desired.

if __name__ == "__main__":
    calculate_validation_loss()