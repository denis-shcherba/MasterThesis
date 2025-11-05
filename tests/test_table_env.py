import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel
import robotic as ry
import gymnasium as gym
import envs  # noqa: F401  
from envs.utils import crop_or_rescale_img

env = gym.make("TableEnv-v0", img_type="DEPTH", robot_mode="taskspace", camera_name="cameraStatic", simulate=True, seed=42, collect_data=False)

env.reset()
rgb, depth = env.unwrapped.getImageDepth()

plt.imshow(rgb)
plt.show()

depth_scaled = crop_or_rescale_img(depth, False, True )

plt.imshow(depth_scaled, cmap='gray')
plt.show()

