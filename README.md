# MediaPipe Landmark Data Extractor
This repository contains functionality for extracting, normalizing
and structuring hand landmark features from video datasets using `MediaPipe`
and `OpenCV`. Raw videos are processed into fixed length numpy arrays (`.npy`)
for model training.

## Key Features
This landmark extractor works one video at a time in succession until stopped or finished. It consists of the following features:
* **Dual-Hand Tracking:** Extracts both hands using `HandLandmarker` model. If hands are missing, they are kept as missing.
* **Location and Scale Normalization:** Hand landmark coordinates are normalized by pinning the wrist to `(0, 0, 0)` origin point and scaled based on the maximum absolute value.
* **Visibility Flags:** Encodes hand presence using a visibility flag (`1` if present, `0` otherwise) and followed by coordinate data of respective hand.
* **Uniform Sampling & Padding:** Uniformly samples frames to achieve target frame and pads missing frames if necessary.
* **Configurable Pipeline:** Dataset and extraction paths, target frame counts and target vector size are managed through YAML configuration file.

## Tech Stack
This project utilizes the following:
* Python
* MediaPipe
* OpenCV
* Numpy
* Pandas
* YAML

## Project Structure
```text
├── src/
│   ├── engine.py           # MediaPipe model initialization & detector setup
│   └── preprocessing.py    # Landmark normalization, hand assignment & video processing
├── extraction.py           # Main execution script for batch dataset feature extraction
├── config_extraction.yaml  # Configuration file for paths and parameters
└── README.md
```

## Prerequisites & Installation
Ensure your machine has Python installed. Clone the repository:
```bash
# Clone repository and move to project folder
git clone https://github.com/hakye05/mp-landmarks-extractor.git
cd mp-landmarks-extractor
```

Install required dependencies:
```bash
pip install opencv-python mediapipe numpy pandas pyyaml
```

Download the official MediaPipe Hand Landmarker model listed in vision tasks and place in project directory or `models/` folder. Additionally, prepare the dataset.

## Configuration (`config_extraction.yaml`)
Configure your file paths, annotations csv separator, frame targets and feature vector sizes (`2 + 3 * 21 * FRAME_COUNT`):
```yaml
mediapipe_path: models/hand_landmarker.task
csv_path: datasets/ANNOTATIONS_CSV
csv_separator: tabular
videos_dir: datasets/VIDEOS_FOLDER
output_dir: datasets/LANDMARKS_OUTPUT
target_frame_count: ENTER_INTEGER
target_vector_size: ENTER_INTEGER
```
## Usage
Run the extraction script to process your video dataset:
```bash
python extraction.py
```
This script will read the annotations csv file, locate the corresponding videos and process each into 2D numpy array.

## TO-DO NOTES
This repository lacks the pose landmarker implementation which is required for extracting the spatial information of hand landmarks due to project's location normalization algorithm.
* Add pose landmarker support
* Add face landmarker support
* Introduce more normalization options
* Add support for processing multiple files simultaneously