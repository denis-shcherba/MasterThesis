import robotic as ry
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import envs.env  # noqa: F401  

env = gym.make("TableEnv-v0", img_type="DEPTH", robot_mode="taskspace", camera_name="cameraWrist", simulate=True, seed=42, collect_data=False)

env.reset()
depth = env.unwrapped.getImageDepth()
plt.imshow(depth)
plt.show()

C = ry.Config()
C.addFile(ry.raiPath("../rai-robotModels/scenarios/pandaSingle.g"))
C.view(True)


bot = ry.BotOp(C, True)
bot.home(C)
quit()
bot.moveTo([.0, .0, .0, -2., 0, 2., -.5])

while bot.getTimeToEnd()>0:
    bot.wait(C)

rgb, depth, pcl = bot.getImageDepthPcl("cameraWrist")
# plt.imshow(rgb)
# plt.show()
#   np.save("book.npy", depth)
plt.imshow(depth)
plt.show()
bot.home(C)
    