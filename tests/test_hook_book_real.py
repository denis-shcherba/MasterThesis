import numpy as np
import matplotlib.pyplot as plt
import robotic as ry
import gymnasium as gym
import envs  # noqa: F401  
from envs.high_level_methods import RobotEnviroment
from envs.utils import rescale_img
import hydra
from omegaconf import DictConfig, open_dict
import logging
from envs.book_spawning import generate_uniform_box_params
import robotic as ry
import pandas as pd
from omegaconf import DictConfig, open_dict
import os
from tqdm import tqdm
import sys
from contextlib import contextmanager

log = logging.getLogger(__name__)

@hydra.main(config_path="../configs/env", config_name="shelf_hook_panda_real", version_base=None)
def main(cfg: DictConfig):

    #evaluate_hook_feasibility(cfg, visualize=False)
    #evaluate_hook_lengths(cfg, file_name="hook_sweep_results.csv")
    collect_hook_data(cfg)

@contextmanager
def silence_logs():
    """Aggressive silencer that redirects C++ system-level streams."""
    # Open /dev/null
    devnull = os.open(os.devnull, os.O_WRONLY)
    # Save original file descriptors for stdout (1) and stderr (2)
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    try:
        # Redirect stdout and stderr file descriptors to devnull
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        # Restore original file descriptors
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        # Clean up
        os.close(devnull)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)

def collect_hook_data(cfg: DictConfig):
    env = gym.make("ShelfEnv-v1", q0= cfg.q0, botop=cfg.botop, on_real=cfg.on_real, rotate_panda_base=cfg.rotate_panda_base, real_table=cfg.real_table, img_type="RGB", robot_mode=cfg.robot_mode, end_effector="hook", path_mode=cfg.path_mode, shelf_pos_xyz=cfg.shelf_pos_xyz, shelf_quaternion=cfg.shelf_quaternion, shelf_floor_offsets=cfg.shelf_floor_offsets, camera_name=cfg.camera_name, simulate=True, seed=42, collect_data=True, box_size_ranges=cfg.box_size_ranges, allow_book_yaw=cfg.allow_book_yaw,  focal_length_range=cfg.focal_length_range, hook_base_length=cfg.hook_base_length, hook_tip_length=cfg.hook_tip_length, hook_width=cfg.get("hook_width", 0.02))
    env.reset()
    env.unwrapped.C.view(True)
    box_size = (cfg.box_size_ranges['x'][0], cfg.box_size_ranges['y'][0], cfg.box_size_ranges['z'][0])
    margins = {'x_min': 0.05, 'x_max': 0.05, 'y_min': 0.02, 'y_max': 0.02}

    uniform_samples = generate_uniform_box_params(env.unwrapped.shelf_dims_for_spawning, box_size=box_size, grid_size=(2, 2), margins=margins)
    print(f"Generated {len(uniform_samples)} uniform samples.")
    del env.unwrapped.bot

    RoboEnv = RobotEnviroment(env.unwrapped.C, sim=(not cfg.on_real), camera="cameraWrist", on_real=cfg.on_real)

    success_count = 0
    
    C2 = ry.Config()
    C2.addConfigurationCopy(env.unwrapped.C)
    C2.delFrame("target_book_0")
    

    for sample in uniform_samples:
        RoboEnv.bot.moveTo(env.unwrapped.q0)
        while RoboEnv.bot.getTimeToEnd() > 0:
            RoboEnv.bot.wait(env.unwrapped.C)
        env.unwrapped._spawn_book(sample[0])
        # maybe here some kind of gripper at book?
        env.unwrapped.C.view(True)
        book_pos = env.unwrapped.C.getFrame("target_book_0").getPosition()
        success = RoboEnv.hook_book("target_book_0")
        if success:
            success_count += 1
            C2.addFrame(f"book_{success_count}").setColor([0,1,0]).setPosition(book_pos).setShape(ry.ST.box, sample[0][:3] )
    

            demo_group = env.unwrapped.h5file.create_group(f"demo_{success_count}")

            if env.unwrapped.path_mode == "taskspace":
                pass
            elif env.unwrapped.path_mode == "jointspace":
                demo_group.create_dataset("path", data=RoboEnv.path)
            if env.unwrapped.img_type.upper() == "DEPTHRGB":
                demo_group.create_dataset(
                "depth", 
                data=RoboEnv.depth_image,
                compression="gzip",
                compression_opts=4
                )

                demo_group.create_dataset(
                "rgb", 
                data=RoboEnv.rgb_image,
                compression="gzip",
                compression_opts=4
                )
            elif env.unwrapped.img_type.upper() == "RGB":
                demo_group.create_dataset(
                "rgb", 
                data=RoboEnv.rgb_image,
                compression="gzip",
                compression_opts=4
                )

    if cfg.on_real:
        RoboEnv._rs_shutdown()
    env.unwrapped.h5file.close()

    print(f"Hooking success rate: {success_count}/{len(uniform_samples)}")
    C2.view(True)


def evaluate_hook_feasibility(cfg: DictConfig, visualize: bool = False):
    env = gym.make("ShelfEnv-v1", q0= cfg.q0, botop=False, on_real=False, rotate_panda_base=cfg.rotate_panda_base, real_table=cfg.real_table, img_type="DEPTH", robot_mode=cfg.robot_mode, end_effector="hook", path_mode=cfg.path_mode, shelf_pos_xyz=cfg.shelf_pos_xyz, shelf_quaternion=cfg.shelf_quaternion, shelf_floor_offsets=cfg.shelf_floor_offsets, camera_name=cfg.camera_name, simulate=True, seed=42, collect_data=False, box_size_ranges=cfg.box_size_ranges, allow_book_yaw=cfg.allow_book_yaw,  focal_length_range=cfg.focal_length_range, hook_base_length=cfg.hook_base_length, hook_tip_length=cfg.hook_tip_length, hook_width=cfg.get("hook_width", 0.02))
    env.reset()
    box_size = (cfg.box_size_ranges['x'][0], cfg.box_size_ranges['y'][0], cfg.box_size_ranges['z'][0])
    margins = {'x_min': 0.05, 'x_max': 0.05, 'y_min': 0.02, 'y_max': 0.02}

    uniform_samples = generate_uniform_box_params(env.unwrapped.shelf_dims_for_spawning, box_size=box_size, grid_size=(10, 10), margins=margins)
    print(f"Generated {len(uniform_samples)} uniform samples.")
    RoboEnv = RobotEnviroment(env.unwrapped.C, sim=True, camera="cameraWrist", on_real=False, visualize=visualize)

    success_count = 0

    for sample in uniform_samples:
        env.reset()
        env.unwrapped._spawn_book(sample[0])
        
        success = RoboEnv.hook_book("target_book_0")
        if success:
            success_count += 1
    
    print(f"Hooking success rate: {success_count}/{len(uniform_samples)}")

    del env
    return success_count


def evaluate_hook_lengths(cfg: DictConfig, file_name: str = "hook_sweep_results.csv"):
    orig_tip = cfg.hook_tip_length
    orig_base = cfg.hook_base_length

    # --- Setup Ranges ---
    tip_offsets = np.linspace(-0.06, 0.06, 13) # 0.01 steps
    base_offsets = np.linspace(-0.3, 0, 31)    # 0.005 steps
    
    results = []
    best_rate = 0.0

    print(f"🚀 Starting sweep (Total: {len(base_offsets) * len(tip_offsets)} sims)")
    print(f"Base: {orig_base} | Tip: {orig_tip}")

    total_iterations = len(base_offsets) * len(tip_offsets)
    # unit_scale=True makes tqdm handle large numbers gracefully
    pbar = tqdm(total=total_iterations, desc="Sweeping Hooks", unit="sim")

    for b_off in base_offsets:
        for t_off in tip_offsets:
            with open_dict(cfg):
                cfg.hook_base_length = float(orig_base + b_off)
                cfg.hook_tip_length = float(orig_tip + t_off)

            # --- SYSTEM-LEVEL SILENCER ---
            with silence_logs():
                successes = evaluate_hook_feasibility(cfg)
            
            # Update metrics (Assumes 100 samples per sim based on grid_size=10x10)
            current_rate = successes / 100.0 
            
            if current_rate > best_rate:
                best_rate = current_rate
                pbar.set_postfix({"best": f"{best_rate:.1%}"})

            results.append({
                "base_length": round(cfg.hook_base_length, 4),
                "tip_length": round(cfg.hook_tip_length, 4),
                "base_off": round(b_off, 4),
                "tip_off": round(t_off, 4),
                "success_count": successes,
                "success_rate": current_rate
            })
            pbar.update(1)
            
    pbar.close()

    # --- Data Processing & Summary ---
    df = pd.DataFrame(results)
    best_row = df.iloc[df['success_rate'].idxmax()]

    print("\n" + "="*50)
    print("📊 SWEEP SUMMARY (Top 10)")
    print("="*50)
    print(df.sort_values(by="success_rate", ascending=False).head(10).to_string(index=False))
    
    print("\n🏆 WINNER:")
    print(f"Base Length: {best_row['base_length']} (Offset: {best_row['base_off']})")
    print(f"Tip Length:  {best_row['tip_length']} (Offset: {best_row['tip_off']})")
    print(f"Success:     {best_row['success_rate']:.1%}")
    print("="*50)

    # Save to Hydra output dir
    df.to_csv(file_name, index=False)
    print(f"✅ CSV saved to: {os.getcwd()}/{file_name}")
    
    return best_row

if __name__ == "__main__":
    main()
