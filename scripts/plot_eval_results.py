import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime
from typing import List, Tuple
import matplotlib.dates as mdates

# --- Config ---
EVAL_DIR = "eval_outputs"  # Path to evaluation results
TIME_FORMAT = "%Y-%m-%d_%H-%M-%S"


def load_data(
    eval_dir: str, start_time: datetime, end_time: datetime
) -> Tuple[List[datetime], List[float], List[float]]:
    """
    Load data from eval_dir within [start_time, end_time].

    Returns:
        timestamps: list of datetime objects
        avg_distances: list of average distances to goal
        success_rates: list of success rates (0-1)
    """
    timestamps, avg_distances, success_rates = [], [], []

    if not os.path.exists(eval_dir):
        print(f"Error: Evaluation directory not found at '{eval_dir}'")
        return timestamps, avg_distances, success_rates

    for date_folder in sorted(os.listdir(eval_dir)):
        date_path = os.path.join(eval_dir, date_folder)
        if not os.path.isdir(date_path):
            continue

        for time_folder in sorted(os.listdir(date_path)):
            time_path = os.path.join(date_path, time_folder)
            json_filepath = os.path.join(time_path, "data.json")

            if not os.path.isfile(json_filepath):
                continue

            try:
                file_timestamp_str = f"{date_folder}_{time_folder}"
                file_time = datetime.strptime(file_timestamp_str, TIME_FORMAT)

                if not (start_time <= file_time <= end_time):
                    continue

                with open(json_filepath, "r") as f:
                    data = json.load(f)

                distances = [item["distance_to_goal"] for item in data if "distance_to_goal" in item]
                successes = [item["success"] for item in data if "success" in item]

                if distances:
                    avg_distances.append(np.mean(distances))
                    timestamps.append(file_time)

                    if successes:
                        success_rate = np.mean(successes)
                        success_rates.append(success_rate)
                    else:
                        success_rates.append(np.nan)

            except (ValueError, KeyError, json.JSONDecodeError) as e:
                print(f"Could not process file in {time_path}: {e}")

    return timestamps, avg_distances, success_rates


def plot_avg_distance(timestamps: List[datetime], avg_distances: List[float], start_str: str, end_str: str):
    if not timestamps:
        print("No distance data to plot.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, avg_distances, marker="o", linestyle="-")
    plt.xlabel("Timestamp")
    plt.ylabel("Average Distance to Goal")
    plt.title(f"Average Goal Distance from {start_str} to {end_str}")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.xlim(min(timestamps), max(timestamps))
    plt.tight_layout()
    plt.show()


def plot_success_rate(timestamps, success_rates, start_str, end_str, use_bars=False):
    if not timestamps:
        print("No success data to plot.")
        return

    plt.figure(figsize=(12, 6))

    if use_bars:
        # Convert datetimes to numbers for uniform bar widths
        ts_nums = mdates.date2num(timestamps)
        bar_width = 0.02  # in days (~30 minutes)
        plt.bar(ts_nums, [sr * 100 for sr in success_rates],
                width=bar_width, align="center")
        plt.gca().xaxis_date()  # format x-axis back to dates
    else:
        # Line plot is usually better for success rates over time
        plt.plot(timestamps, [sr * 100 for sr in success_rates],
                 marker="o", linestyle="-")

    plt.xlabel("Timestamp")
    plt.ylabel("Success Rate (%)")
    plt.title(f"Success Rate from {start_str} to {end_str}")
    plt.grid(True, axis="y")
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    START_TIMESTAMP_STR = "2025-09-09_00-00-00"
    END_TIMESTAMP_STR = "2025-09-17_23-59-59"

    start_time = datetime.strptime(START_TIMESTAMP_STR, TIME_FORMAT)
    end_time = datetime.strptime(END_TIMESTAMP_STR, TIME_FORMAT)

    timestamps, avg_distances, success_rates = load_data(EVAL_DIR, start_time, end_time)

    plot_avg_distance(timestamps, avg_distances, START_TIMESTAMP_STR, END_TIMESTAMP_STR)
    plot_success_rate(timestamps, success_rates, START_TIMESTAMP_STR, END_TIMESTAMP_STR)
