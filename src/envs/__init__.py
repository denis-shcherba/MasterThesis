# In my_custom_env/__init__.py
from gymnasium.envs.registration import register

register(
     id="ShelfEnv-v0",
     entry_point="envs.shelf_env_deprecated:ShelfEnv",
     #max_episode_steps=300, # As in the lerobot script TODO look into
)


register(
     id="TableEnv-v0",
     entry_point="envs.table_env:TableEnv",
)

register(
     id="ShelfEnv-v1",
     entry_point="envs.shelf_env:ShelfEnv",
)
