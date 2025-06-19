# TODO fix inference

# eval_policy.py
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import logging
import os
import time

from envs.create_env import ShelfPullDataCollector
from models.policy_head.policy_network import create_model
from data_handling.processing import pose_7d_to_9d

# Import our new helper classes
from evaluation.utils import InferencePreprocessor, ActionPostprocessor

log = logging.getLogger(__name__)


class PolicyEvaluator:
    """Orchestrates the policy evaluation process."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = self._setup_device()
        
        log.info("Initializing environment...")
        self.env = ShelfPullDataCollector(**cfg.env)
        
        log.info("Initializing model...")
        self.model = create_model(cfg.model).to(self.device).eval() # Set to eval mode on creation
        self._load_checkpoint()

        log.info("Initializing data processors...")
        self.preprocessor = InferencePreprocessor(cfg.model, cfg.get('data', {}), self.device)
        self.postprocessor = ActionPostprocessor(cfg.model, cfg.env)
        
        self.num_eval_steps = cfg.inference.get("num_steps", 64)
        self.policy_type = cfg.model.type

    def _setup_device(self) -> torch.device:
        device_str = self.cfg.inference.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {device_str}")
        return torch.device(device_str)

    # In PolicyEvaluator class in eval_policy.py

    def _load_checkpoint(self):
        """Loads the model weights from the specified checkpoint."""
        checkpoint_path = self.cfg.inference.checkpoint_path
        if not os.path.isabs(checkpoint_path):
            original_cwd = hydra.utils.get_original_cwd()
            checkpoint_path = os.path.join(original_cwd, checkpoint_path)

        log.info(f"Loading checkpoint from: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        # Load the entire checkpoint dictionary, allowing non-tensor objects
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Extract the state_dict
        state_dict = checkpoint['model_state_dict']
        
        # --- THE FIX IS HERE ---
        # Correctly check if any key in the state_dict's keys starts with 'module.'
        if any(k.startswith('module.') for k in state_dict.keys()):
            log.info("Removing 'module.' prefix from state_dict keys.")
            # Create a new dictionary with the corrected keys
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
            
        self.model.load_state_dict(state_dict)
        log.info("Model weights loaded successfully.")

    def get_observation(self) -> dict:
        """Gathers raw observation data from the environment."""
        point_cloud = self.env.render()
        current_pose_7d = self.env.C.getJointState()
        
        # Prepare robot state based on action dimension requirements
        if self.cfg.model.action_dim == 9:
            robot_state = pose_7d_to_9d(current_pose_7d)
        else: # For 3D action spaces, use the 3D position part of the state
            robot_state = current_pose_7d[:3]
            
        return {
            "point_cloud": point_cloud,
            "robot_state": robot_state,
            "current_pose_7d": current_pose_7d
        }

    def run(self):
        """Executes the main evaluation loop."""
        log.info(f"Starting evaluation for {self.num_eval_steps} steps...")
        self.env.spawn_books_scene()
        self.env.C.view(True)
        q0 = self.env.C.getJointState()
        
        hidden_state = None

        for i in range(self.num_eval_steps):
            log.info(f"--- Step {i+1}/{self.num_eval_steps} ---")
            
            # 1. Get observation from the environment
            raw_obs = self.get_observation()

            # 2. Preprocess data for the model
            input_for_model = self.preprocessor.process(raw_obs)
            
            # 3. Perform model inference
            with torch.no_grad():
                if self.policy_type == "multimodal":
                    # This logic remains specific to the model's forward pass
                    if self.cfg.model.get("policy_head_type") == "gru":
                        output, hidden_state = self.model(
                            input_for_model["point_cloud"], 
                            input_for_model["state"], 
                            torch.tensor(i).reshape(1).to(self.device),
                            hidden_state=hidden_state
                        )
                    else: # mlp
                        output = self.model(input_for_model["point_cloud"], input_for_model["state"], torch.tensor(i).reshape(1).to(self.device))
                
                elif self.policy_type == "regression":
                    output = self.model(input_for_model["point_cloud"])
                
                else:
                    raise ValueError(f"Unknown policy type: {self.policy_type}")

            # 4. Post-process model output to get an executable action
            action = self.postprocessor.process(output, raw_obs["current_pose_7d"])
            log.info(f"Predicted Action (7D Pose): {np.round(action, 3)}")

            # 5. Execute action in the environment
            self.env.C.setJointState(action)
            self.env.C.view(False)
            time.sleep(0.1) # A small pause for visualization
            
            # Special handling for regression model to reset scene after each prediction
            if self.policy_type == "regression":
                log.info("Regression mode: resetting scene for next prediction.")
                time.sleep(2) # Pause to see the result
                self.env.C.setJointState(q0)
                self.env.C.delFrame("target_book_0")
                self.env.C.view(False)
                self.env.spawn_books_scene()
                self.env.C.view(True)

        log.info("Evaluation finished.")

@hydra.main(config_path="configs", config_name="inference", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation, managed by Hydra."""
    log.info("="*80)
    log.info("Policy Evaluation Script")
    log.info("="*80)
    log.info(f"Full config:\n{OmegaConf.to_yaml(cfg)}")

    try:
        evaluator = PolicyEvaluator(cfg)
        evaluator.run()
    except (FileNotFoundError, ValueError, TypeError, NotImplementedError) as e:
        log.error(f"A configuration or setup error occurred: {e}")
    except Exception as e:
        log.error(f"An unexpected error occurred during evaluation: {e}", exc_info=True)

if __name__ == "__main__":
    main()