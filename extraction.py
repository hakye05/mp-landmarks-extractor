import logging
import yaml
import numpy as np
import pandas as pd
import os

from src.engine import initialize_detector
from src.preprocessing import process_video, normalize_landmarks


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load configuration file
with open("config_extraction.yaml", "r") as f:
    config = yaml.safe_load(f)

hand_detector = initialize_detector(config['mediapipe_path'])
target_vector_size = config['target_vector_size']
target_frame_count = config['target_frame_count']
csv_path = config['csv_path']
videos_dir = config['videos_dir']
output_dir = config['output_dir']
normalizer = normalize_landmarks

# Prepare data and directories
df = pd.read_csv(csv_path, sep="\t")
df["file_path"] = videos_dir + df["attachment_id"] + ".mp4"
os.makedirs(output_dir, exist_ok=True)

# Extraction Loop
for i, row in df.iterrows():
    video_path = row["file_path"]
    attachment_id = row["attachment_id"]
    data = process_video(video_path, hand_detector, normalizer, target_vector_size, target_frame_count)

    if data is None:
        continue

    save_path = os.path.join(output_dir, f"{attachment_id}.npy")
    np.save(save_path, data)

print("Finished extraction")