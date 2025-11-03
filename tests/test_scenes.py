import robotic as ry
import os

folder_path = "src/envs/scenes"
g_files = [f for f in os.listdir(folder_path) if f.endswith('.g')]

for g_file in g_files:
    C = ry.Config()
    C.addFile(os.path.join(folder_path, g_file))
    C.view(True)
    del C
