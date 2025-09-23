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
from utils.data_utils import  denormalize_actions, normalize_state
import robotic as ry
import time 

log = logging.getLogger(__name__)

SHOW_RAI = True
REUSE_DATA = True 
PADD_DATA = False

def simulate_data_against_prediction(cfg, env, target_action_seq, predicted_action_seq, book_params, state_input_seq=None):
    env.unwrapped._spawn_book(book_params)

    
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

def show_state_input_seq(cfg, env, state_input_seq, color=[1, 0, 0, .9], prefix=""):
    for i in range(state_input_seq.shape[0]):
        # Use the same reverse indexing logic as your first function
        previous_pos = state_input_seq[-(i + 1)].cpu().numpy()
        env.unwrapped.C.addFrame(prefix+f"previous_pos_{i}").setPosition(previous_pos).setShape(ry.ST.sphere, [.015]).setColor(color)
        env.unwrapped.C.view(False)
        print("Previous Position:", previous_pos)

def show_data_agains_prediction(cfg, env, target_action_seq, predicted_action_seq):

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
    env.unwrapped.C.view(True)
    env.unwrapped.C.delFrame("big_box_inside_0_2")

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
            i = 0
            prior_book_single = torch.zeros(7)
            while i < current_batch_size:
                
                # --- Grab the full sequence and target for the i-th sample ---
                full_context_depth = context_depth_batch[i] # Shape: [M, C, H, W]
                full_context_state = context_state_batch[i] # Shape: [M, state_dim]
                target_single = target_actions_batch[i].unsqueeze(0) # Shape: [1, N, action_dim]
                book_single = book_params[i]
                diff = np.linalg.norm(book_single.cpu().numpy() - prior_book_single.cpu().numpy())
                if diff > 1e-4:
                    env.unwrapped.reset()
                    env.unwrapped._delete_books()
                    env.unwrapped._spawn_book(book_single.cpu())
                
                prior_book_single = book_single.clone()

                # Denormalize the ground truth target once for loss calculation
                denormalized_target = denormalize_actions(target_single, normalization_stats["action_stats"])

                # =================================================================
                # START: LOGIC FOR PADDED DATA VALIDATION
                # =================================================================
                if PADD_DATA:
                    # This mode simulates starting a trajectory from scratch.
                    # It uses only the very first state/observation from the context window,
                    # padding the rest with zeros, and then autoregressively rolls out
                    # the full prediction sequence.

                    context_len = full_context_state.size(0) # This is M

                    # --- 1. Create the initial, zero-padded context ---
                    # We need M-1 pads to fill the context window initially.
                    num_pads = context_len - 1
                    
                    # Create padding tensors.
                    state_pads = torch.zeros(num_pads, full_context_state.size(1), device=device)
                    depth_pads = torch.zeros(num_pads, *full_context_depth.size()[1:], device=device)

                    # Get the single, real starting state and depth image.
                    # We use slicing [0:1] to maintain the sequence dimension.
                    first_real_state = full_context_state[0:1]
                    first_real_depth = full_context_depth[0:1]

                    # Concatenate pads and the first real data point to form the initial model input.
                    # The shape will be [1, M, ...], ready for the model.
                    current_state_context = torch.cat([state_pads, first_real_state], dim=0).unsqueeze(0)
                    current_depth_context = torch.cat([depth_pads, first_real_depth], dim=0).unsqueeze(0)

                    # --- 2. Autoregressively roll out the trajectory ---
                    # This logic is similar to the REUSE_DATA block.
                    execution_stepsize = 10 # Predict N steps, execute a chunk, update context, repeat.
                    prediction_horizon_N = denormalized_target.size(1)
                    rollout_predictions_list = []
                    num_actions_generated = 0

                    while num_actions_generated < prediction_horizon_N:
                        # Get the model's prediction for the next N steps.
                        predicted_action_sequence = model(current_depth_context, current_state_context)
                        
                        # Determine the size of the action chunk to execute in this iteration.
                        remaining_steps = prediction_horizon_N - num_actions_generated
                        current_chunk_size = min(execution_stepsize, remaining_steps)
                        
                        # Isolate and denormalize the action chunk.
                        action_chunk = predicted_action_sequence[:, :current_chunk_size, :]
                        denormalized_action_chunk = denormalize_actions(action_chunk, normalization_stats["action_stats"])
                        
                        # Execute each action in the chunk one by one to update the context.
                        for k in range(current_chunk_size):
                            # Get the k-th action (normalized for context, denormalized for env).
                            next_action_normalized = action_chunk[:, k:k+1, :]
                            denormalized_next_action = denormalized_action_chunk[:, k:k+1, :]
                            
                            # Store this single denormalized action for the final loss calculation.
                            rollout_predictions_list.append(denormalized_next_action)
                            
                            # Update state context: remove the oldest, append the new predicted one.
                            current_state_context = torch.cat([current_state_context[:, 1:, :], next_action_normalized], dim=1)
                            
                            try:
                                # Execute the action in the environment to get the next depth image.
                                env_action = denormalized_next_action.squeeze().cpu().numpy()
                                obs, _, _, _, _ = env.step(env_action) 
                                new_depth = torch.from_numpy(obs['depth']).to(device).unsqueeze(0).unsqueeze(0)
                                
                                # Update depth context: remove oldest, append newest from env.
                                current_depth_context = torch.cat([current_depth_context[:, 1:, :, :], new_depth], dim=1)
                            except Exception as e:
                                print(f"Warning: Env step failed during padded rollout. Using placeholder depth. Error: {e}")
                                # If env fails, repeat the last known depth image as a fallback.
                                current_depth_context = torch.cat([current_depth_context[:, 1:, :, :], current_depth_context[:, -1:, :, :]], dim=1)

                        # Update the counter for the while loop.
                        num_actions_generated += current_chunk_size
                    
                    # --- 3. Finalize prediction and calculate loss ---
                    # Combine the list of single predicted actions into one final trajectory tensor.
                    denormalized_prediction = torch.cat(rollout_predictions_list, dim=1)
                    
                    # For visualization, we can show the initial (padded) input state.
                    denormalized_input_state = denormalize_actions(current_state_context, normalization_stats["action_stats"])

                    if SHOW_RAI:
                        show_data_agains_prediction(cfg, env, denormalized_target.squeeze(0).cpu(), denormalized_prediction.squeeze(0).cpu())
                    
                    loss = loss_fn(denormalized_prediction, denormalized_target)
                    total_loss += loss.item()

                    # CRITICAL: Increment the sample counter to avoid an infinite loop.
                    i += prediction_horizon_N
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
                    
                    if SHOW_RAI:
                        show_state_input_seq(cfg, env, denormalized_state.squeeze(0).cpu())

                    # =================================================================
                    # START: LOGIC FOR REUSING/ROLLING OUT PREDICTIONS
                    # =================================================================\
                    if REUSE_DATA:
                        if i != 0 and SHOW_RAI and diff < 1e-4:
                            show_state_input_seq(cfg, env, old_seq.squeeze(0).cpu(), [0, 0, 1, .9], prefix="old_")
                        # --- Autoregressive Rollout Simulation with a Step Size ---

                        # A value > 1 will execute actions in chunks.
                        execution_stepsize = 10
                        
                        # 1. Initialize the context. We'll update this in a loop.
                        current_state_context = state_single.clone()
                        current_depth_context = depth_single.clone()
                        
                        prediction_horizon_N = denormalized_target.size(1)
                        rollout_predictions_list = []
                        
                        # CHANGED: Use a 'while' loop to keep track of generated actions.
                        # This is more robust than a 'for' loop for handling chunks.
                        num_actions_generated = 0
                        while num_actions_generated < prediction_horizon_N:
                            # Get the model's prediction for the next N steps based on the current context
                            if i == 0:
                                predicted_action_sequence = model(current_depth_context, current_state_context)
                            elif i>0 and diff < 1e-4:
                                predicted_action_sequence = model(current_depth_context, normalize_state(old_seq, normalization_stats["action_stats"]))
                            else:
                                predicted_action_sequence = model(current_depth_context, current_state_context)

                            # --- NEW: Logic to handle chunks ---
                            # Determine how many actions to take in this iteration.
                            # This handles the final chunk if N isn't divisible by the stepsize.
                            remaining_steps = prediction_horizon_N - num_actions_generated
                            current_chunk_size = min(execution_stepsize, remaining_steps)
                            
                            # Isolate the chunk of actions we are going to execute
                            action_chunk = predicted_action_sequence[:, :current_chunk_size, :]
                            
                            # Denormalize the entire chunk at once
                            denormalized_action_chunk = denormalize_actions(action_chunk, normalization_stats["action_stats"])
                            
                            # --- NEW: Inner loop to execute the chunk of actions ---
                            for k in range(current_chunk_size):
                                # Get the k-th action from the (normalized) chunk for context update
                                next_action_normalized = action_chunk[:, k:k+1, :]
                                
                                # Get the k-th action from the (denormalized) chunk for the environment
                                denormalized_next_action = denormalized_action_chunk[:, k:k+1, :]
                                
                                # Store this single action for the final loss calculation
                                rollout_predictions_list.append(denormalized_next_action)
                                
                                # --- Update Context for the Next Step (inside the chunk loop) ---
                                # Update state context: remove oldest, append the new predicted one
                                current_state_context = torch.cat([current_state_context[:, 1:, :], next_action_normalized], dim=1)
                                
                                try:
                                    # Execute the predicted action in the environment
                                    env_action = denormalized_next_action.squeeze().cpu().numpy()
                                    obs, _, _, _, _ = env.step(env_action) 
                                    new_depth = torch.from_numpy(obs['depth']).to(device).unsqueeze(0).unsqueeze(0)
                                    
                                    # Update depth context: remove oldest, append newest from env
                                    current_depth_context = torch.cat([current_depth_context[:, 1:, :, :], new_depth], dim=1)
                                except Exception as e:
                                    print(f"Warning: Could not step environment for data reuse. Using placeholder depth. Error: {e}")
                                    # If env fails, use a placeholder (e.g., repeat the last known depth)
                                    current_depth_context = torch.cat([current_depth_context[:, 1:, :, :], current_depth_context[:, -1:, :, :]], dim=1)

                            # Update the counter for the outer while loop
                            num_actions_generated += current_chunk_size

                        # 3. After the loop, combine the list of single actions into one trajectory tensor
                        denormalized_prediction = torch.cat(rollout_predictions_list, dim=1)
                        
                        old_seq = denormalized_prediction
                        # Your i+= logic would now be handled by a while loop in the outer scope
                        i += prediction_horizon_N # if using a while loop for 'i'
                    # =================================================================
                    # END: LOGIC FOR REUSING/ROLLING OUT PREDICTIONS
                    # =================================================================
                    else:
                        # --- Original One-Shot Prediction Logic ---
                        prediction_single = model(depth_single, state_single)
                        denormalized_prediction = denormalize_actions(prediction_single, normalization_stats["action_stats"])

                        
                        i+=1
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