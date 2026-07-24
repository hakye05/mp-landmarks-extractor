import yaml
import numpy as np
import pandas as pd
import os

from src.engine import initialize_detector
from src.preprocessing import normalize_hand_3d, normalize_hand_2d, process_video


with open("config_3d_data_extraction.yaml", "r") as f:
    config = yaml.safe_load(f)

hand_detector = initialize_detector(config['model_path'])
normalizer = normalize_hand_3d if config['dimension'] == 'xyz' else normalize_hand_2d
target_vector_size = config['target_vector_size']
target_frame_count = config['target_frame_count']

csv_path = config['csv_path']
videos_dir = config['videos_dir']
output_dir = config['output_dir']

os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv(csv_path, sep="\t")
df["file_path"] = videos_dir + df["attachment_id"] + ".mp4"

# Main loop for extraction
for i, row in df.iterrows():
    video_path = row["file_path"]
    attachment_id = row["attachment_id"]
    print(f"[{i+1}/{len(df)}] Processing {attachment_id}")
    data = process_video(video_path, hand_detector, normalizer, target_vector_size, target_frame_count)

    if data is None:
        continue

    save_path = os.path.join(output_dir, f"{attachment_id}.npy")
    np.save(save_path, data)

print("Finished extraction")