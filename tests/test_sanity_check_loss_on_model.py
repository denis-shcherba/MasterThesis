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
import time 

log = logging.getLogger(__name__)

SHOW_RAI = True
REUSE_DATA = False 
PADD_DATA = False

def simulate_data_against_prediction(cfg, env, target_action_seq, predicted_action_seq, book_params, state_input_seq=None):
    env.unwrapped._spawn_book(book_params)

    action_prediction_horizon = 5 # cfg.model.get("action_prediction_horizon", 10)
    
    with open("/home/denis/git/MasterThesis/outputs/final_outputs/normalization_stats_1000.yaml", 'r') as file:
        # todo change to config
        normalization_stats = yaml.safe_load(file)
    target_action_seq = denormalize_actions(target_action_seq, normalization_stats["action_stats"])
    previous_action_seq = denormalize_actions(predicted_action_seq, normalization_stats["action_stats"])

    if state_input_seq is not None:
        # Denormalize just once outside the loop for efficiency
        state_input_seq = denormalize_actions(state_input_seq, normalization_stats["action_stats"])
            
    for i in range(target_action_seq.shape[0]):
        # Use the same reverse indexing logic as your first function
        previous_pos = state_input_seq[-(i + 1)].cpu().numpy()
        env.unwrapped.C.addFrame(f"previous_pos_{i}").setPosition(previous_pos).setShape(ry.ST.sphere, [.015]).setColor([1, 0, 0])
        print("Previous Position:", previous_pos)
            
    env.unwrapped.C.view(True)
    for i in range(target_action_seq.shape[0]):
        target_pos = target_action_seq.squeeze(0).cpu().numpy()[i]
        predicted_pos = previous_action_seq.squeeze(0).cpu().numpy()[i]

        env.unwrapped.C.addFrame(f"target_pos_{i}").setPosition(target_pos).setShape(ry.ST.sphere, [.012]).setColor([.1*i,1-.1*i,1])
        env.unwrapped.C.addFrame(f"predicted_pos_{i}").setPosition(predicted_pos).setShape(ry.ST.box, [.02, .02, .02]).setColor([1-.1*i,.1*i,.2])
        print("Target Position:", target_pos)
        print("Predicted Position:", predicted_pos)

    env.unwrapped.C.view(True)

def show_state_input_seq(cfg, env, state_input_seq):
    for i in range(state_input_seq.shape[0]):
        # Use the same reverse indexing logic as your first function
        previous_pos = state_input_seq[-(i + 1)].cpu().numpy()
        env.unwrapped.C.addFrame(f"previous_pos_{i}").setPosition(previous_pos).setShape(ry.ST.sphere, [.015]).setColor([1, 0, 0, .9])
        print("Previous Position:", previous_pos)

def show_data_agains_prediction(cfg, env, target_action_seq, predicted_action_seq):
    # if book_params is not None:
    #     env.unwrapped._spawn_book(book_params)

    env.unwrapped.C.view(True)
    for i in range(target_action_seq.shape[0]):
        target_pos = target_action_seq.squeeze(0).cpu().numpy()[i]
        predicted_pos = predicted_action_seq.squeeze(0).cpu().numpy()[i]

        env.unwrapped.C.addFrame(f"target_pos_{i}").setPosition(target_pos).setShape(ry.ST.sphere, [.012]).setColor([.1*i,1-.1*i,1, .9])
        env.unwrapped.C.addFrame(f"predicted_pos_{i}").setPosition(predicted_pos).setShape(ry.ST.box, [.02, .02, .02]).setColor([1-.1*i,.1*i,.2, .9])
        print("Target Position:", target_pos)
        print("Predicted Position:", predicted_pos)

    env.unwrapped.C.view(True)

def delete_all_extra_frame(C):
    for name in C.getFrameNames():
        if "target_pos_" in name or "predicted_pos_" in name or "previous_pos_" in name:
            C.delFrame(name)

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
    env = gym.make("ShelfEnv-v0", obs_type="depth_agent_pos")
    obs, info = env.reset()
    env.unwrapped._delete_books()

    normalization_stats_path = cfg.get("inference", {}).get("normalization_stats_path", None)
    if normalization_stats_path is None:
        log.error("normalization_stats_path not found.")
        return

    with open(normalization_stats_path, 'r') as file:
        normalization_stats = yaml.safe_load(file)


    with torch.no_grad():
        # Outer loop iterates through batches from the dataloader
        for batch in tqdm(val_loader, desc="Validating"):
            # Get the full batch tensors from the dataloader
            # We append '_batch' to clarify these are multi-item tensors
            context_depth_batch = batch['observation_sequence'].to(device)
            context_state_batch = batch['previous_actions_sequence'].to(device)
            target_actions_batch = batch['target_actions_sequence'].to(device)
            book_params = batch['book_params'].to(device)

            # Get the number of items in the current batch
            current_batch_size = context_depth_batch.size(0)

            # Inner loop to process each sample in the batch individually
            for i in range(current_batch_size):
                
                # --- Grab the full sequence and target for the i-th sample ---
                full_context_depth = context_depth_batch[i] # Shape: [M, C, H, W]
                full_context_state = context_state_batch[i] # Shape: [M, state_dim]
                target_single = target_actions_batch[i].unsqueeze(0) # Shape: [1, N, action_dim]
                book_single = book_params[i]
                env.unwrapped._spawn_book(book_single.cpu())

                # Denormalize the ground truth target once for loss calculation
                denormalized_target = denormalize_actions(target_single, normalization_stats["action_stats"])

                # =================================================================
                # START: LOGIC FOR PADDED DATA VALIDATION
                # =================================================================
                if PADD_DATA:
                    context_len = full_context_state.size(0) # This is M

                    # Loop from 1 to M to simulate a growing context window
                    for j in range(1, context_len + 1):
                        # --- 1. Create zero-padding for the current step ---
                        # Number of steps to pad is M - j
                        num_pads = context_len - j
                        
                        # Create padding tensors with the correct dimensions and device
                        state_pads = torch.zeros(num_pads, full_context_state.size(1), device=device)
                        depth_pads = torch.zeros(num_pads, *full_context_depth.size()[1:], device=device)

                        # --- 2. Get the real data seen so far (from step 0 to j-1) ---
                        real_states = full_context_state[:j]
                        real_depths = full_context_depth[:j]

                        # --- 3. Concatenate padding and real data to form model input ---
                        # The input will be [zeros, ..., zeros, real_data_0, ..., real_data_j-1]
                        current_input_state = torch.cat([state_pads, real_states], dim=0).unsqueeze(0)
                        current_input_depth = torch.cat([depth_pads, real_depths], dim=0).unsqueeze(0)

                        # --- 4. Run forward pass and calculate loss for this step ---
                        prediction_single = model(current_input_depth, current_input_state)
                        
                        # Denormalize prediction for loss and visualization
                        denormalized_prediction = denormalize_actions(prediction_single, normalization_stats["action_stats"])
                        
                        # Denormalize the input states for visualization
                        denormalized_input_state = denormalize_actions(current_input_state, normalization_stats["action_stats"])

                        if SHOW_RAI:
                            show_data_agains_prediction(cfg, env, denormalized_target.squeeze(0).cpu(), denormalized_prediction.squeeze(0).cpu(), book_single.cpu(), denormalized_input_state.squeeze(0).cpu())
                        
                        loss = loss_fn(denormalized_prediction, denormalized_target)
                        total_loss += loss.item()
                # =================================================================
                # END: LOGIC FOR PADDED DATA VALIDATION
                # =================================================================
                else:
                    # --- Logic for full, unpadded sequences ---
                    
                    # Grab the initial ground-truth context from the dataloader.
                    # This serves as the starting point for both one-shot and rollout predictions.
                    depth_single = full_context_depth.unsqueeze(0)
                    state_single = full_context_state.unsqueeze(0)

                    denormalized_state = denormalize_actions(state_single, normalization_stats["action_stats"])
                    show_state_input_seq(cfg, env, denormalized_state.squeeze(0).cpu())

                    # =================================================================
                    # START: LOGIC FOR REUSING/ROLLING OUT PREDICTIONS
                    # =================================================================
                    if REUSE_DATA:
                        # --- Autoregressive Rollout Simulation ---
                        
                        # 1. Initialize the context. We'll update this in a loop.
                        current_state_context = state_single.clone()
                        current_depth_context = depth_single.clone()
                        
                        # Determine the prediction horizon (N) from the target tensor
                        prediction_horizon_N = denormalized_target.size(1)
                        
                        # Store the sequence of predicted actions during the rollout
                        rollout_predictions_list = []
                        
                        # 2. Loop for N steps, generating one action at a time
                        for _ in range(prediction_horizon_N):
                            # Get the model's prediction for the next N steps
                            # Note: The model still predicts a full sequence, but we only use the first step
                            predicted_action_sequence = model(current_depth_context, current_state_context)
                            
                            # Isolate the very first action from the predicted sequence
                            next_action = predicted_action_sequence[:, 0:1, :] # Shape: [1, 1, action_dim]
                            
                            # Denormalize and store this single action
                            denormalized_next_action = denormalize_actions(next_action, normalization_stats["action_stats"])
                            rollout_predictions_list.append(denormalized_next_action)
                            
                            # --- Update Context for the Next Step ---
                            # Update state context: remove the oldest state and append the new predicted one
                            current_state_context = torch.cat([current_state_context[:, 1:, :], next_action], dim=1)
                            
                            try:
                                # Execute the predicted action in the environment
                                env_action = denormalized_next_action.squeeze().cpu().numpy()
                                obs, _, _, _ , _ = env.step(env_action) 
                                new_depth = torch.from_numpy(obs['depth']).to(device).unsqueeze(0).unsqueeze(0) # Shape: [1, 1, C, H, W]
                                
                                # Update depth context: remove oldest, append newest from env
                                current_depth_context = torch.cat([current_depth_context[:, 1:, :, :], new_depth], dim=1)
                            except Exception as e:
                                print(f"Warning: Could not step environment for data reuse. Using placeholder depth. Error: {e}")
                                # If env fails or is not available, use a placeholder (e.g., repeat the last known depth)
                                current_depth_context = torch.cat([current_depth_context[:, 1:, :, :], current_depth_context[:, -1:, :, :]], dim=1)

                        # 3. After the loop, combine the list of single actions into one trajectory tensor
                        denormalized_prediction = torch.cat(rollout_predictions_list, dim=1)

                        i+=prediction_horizon_N-1
                    # =================================================================
                    # END: LOGIC FOR REUSING/ROLLING OUT PREDICTIONS
                    # =================================================================
                    else:
                        # --- Original One-Shot Prediction Logic ---
                        prediction_single = model(depth_single, state_single)
                        denormalized_prediction = denormalize_actions(prediction_single, normalization_stats["action_stats"])

                    # --- Common operations for both REUSE_DATA true/false ---
                    
                    # Denormalize the initial state context for visualization

                    if SHOW_RAI:
                        # Note: Corrected a typo in the function name from your snippet
                        show_data_agains_prediction(cfg, env, denormalized_target.squeeze(0).cpu(), denormalized_prediction.squeeze(0).cpu())
                        
                    # Calculate the loss between the final prediction (either one-shot or rollout) and the target
                    loss = loss_fn(denormalized_prediction, denormalized_target)
                    
                    # Accumulate the loss
                    total_loss += loss.item()
                    delete_all_extra_frame(env.unwrapped.C)
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