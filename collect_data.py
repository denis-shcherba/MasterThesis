# HARD TODO

import hydra
from omegaconf import DictConfig
from envs.create_env import ShelfPullDataCollector 
import logging

log = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="data_collection", version_base=None)
def main(cfg: DictConfig):

    log.info("Starting policy evaluation/inference...")

    collector = ShelfPullDataCollector(**cfg.env)
    collector.spawn_books_scene()
    collector.C.view(True) 
    run_data_collection(cfg.env)


def run_data_collection(config: dict):
    print(f"Running with config: {config}")

    collector = None  # Initialize for finally block
    try:
        print("Initializing ShelfPullDataCollector...")
        collector = ShelfPullDataCollector(**config)
        
        print("Initialization complete. Starting data collection experiment...")
        collector.run_experiment(view_simulation_steps=True) # Set to False for headless execution TODO
        
        print("Data collection experiment finished.")

    except Exception as e:
        print(f"An error occurred during the experiment: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if collector:
            print("Closing collector resources...")
            collector.close()
            print("Collector resources closed.")

if __name__ == "__main__":
    main()
