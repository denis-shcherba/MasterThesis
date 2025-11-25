import argparse
import torch

from data_handling.waypoint_dataset import create_wp_dataloaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--obs", type=str, default="points")
    args = parser.parse_args()

    print("\n=== Testing WaypointDataset and DataLoaders ===\n")

    train_loader, val_loader = create_wp_dataloaders(
        h5_file_path=args.h5,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        observation_mode=args.obs,
        num_workers=0,  # debugging
    )

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    # --- Print demo info ---
    print("\n--- Demo information ---")
    print(f"Train demos kept (keys): {[m['demo_key'] for m in train_dataset.demo_meta]}")
    print(f"Val demos kept (keys):   {[m['demo_key'] for m in val_dataset.demo_meta]}")

    print(f"Train dataset length (windows): {len(train_dataset)}")
    print(f"Val dataset length (windows):   {len(val_dataset)}")

    print("\n--- Fetching one batch from train_loader ---")
    batch = next(iter(train_loader))

    if isinstance(batch, dict):
        for k, v in batch.items():
            if torch.is_tensor(v):
                print(f"{k}: {v.shape}")
            else:
                print(f"{k}: {v}")
    else:
        print("Non-dict batch:", batch)

    print("\n--- Test completed successfully. ---\n")


if __name__ == "__main__":
    main()
