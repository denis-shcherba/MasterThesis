import matplotlib.pyplot as plt
import gymnasium as gym
import envs  # noqa: F401  

env = gym.make("TableEnv-v0", img_type="DEPTH", robot_mode="normal", path_mode="pos3d", camera_name="cameraStaticTable", simulate=True, seed=42, collect_data=False)

env.reset()
rgb, depth = env.unwrapped.getImageDepth()

plt.imshow(rgb)
plt.show()

plt.imshow(depth, cmap='gray')
plt.show()

