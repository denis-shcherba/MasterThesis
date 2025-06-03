# HARD TODO

import hydra
from omegaconf import DictConfig, OmegaConf
from envs.create_env import ShelfPullDataCollector 


@hydra.main(config_path="configs", config_name="inference", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # for debug

    experiment_config = OmegaConf.to_container(cfg, resolve=True)
    
    # Now you can use experiment_config like before
    run_data_collection(experiment_config)

def run_data_collection(config: dict):
    # your existing logic here
    print(f"Running with config: {config}")

    collector = None  # Initialize for finally block
    try:
        print("Initializing ShelfPullDataCollector...")
        collector = ShelfPullDataCollector(**config)
        
        print("Initialization complete. Starting data collection experiment...")
        #collector.run_experiment(view_simulation_steps=True) # Set to False for headless execution TODO
        
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
