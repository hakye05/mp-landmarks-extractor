import cv2
import numpy as np
import mediapipe as mp


def normalize_landmarks(hand_landmarks):
    """Normalizes raw hand landmarks by pinning the wrist to the origin point
    and scaling coordinates uniformly based on the maximum absolute value.

    :param hand_landmarks: Raw hand landmarks containing x, y and z attributes.
    :return: normalized hand landmarks coordinates.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])

    # Pin wrist to origin and normalize scale
    coords -= coords[0]
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val

    return coords


def assign_hands(result):
    """Assigns detected hand landmarks to left/right categories based on MediaPipe's
    handedness classification.

    :param result: The results of MediaPipe's extracted hand landmarks
    :return: (current_left, current_right) assigned landmarks or None values.
    """
    current_left, current_right = None, None

    if not result.hand_landmarks or not result.handedness:
        return None, None

    hands = result.hand_landmarks
    handedness = result.handedness

    # Evaluate MediaPipe's prediction
    for hand_landmarks, handedness_info in zip(hands, handedness):
        label = handedness_info[0].category_name

        if label == "Left":
            current_left = hand_landmarks
        elif label == "Right":
            current_right = hand_landmarks

    return current_left, current_right


def build_frame_vector(hand_obj, target_vector_size, normalizer):
    """Builds a vector representation of hand landmarks. Contains a flag for visibility
    of the hand (1 if hand is present, 0 otherwise).

    :param hand_obj: The hand landmark or None value (if the hand is missing).
    :param target_vector_size: Expected length of the normalized coordinate features.
    :param normalizer: Function to normalize the hand landmarks.
    :return: A vector representation of the hand landmarks where presence flag is followed by the respective hand coordinates.
    """
    if hand_obj is None:
        return [0] + [0] * target_vector_size

    norm = normalizer(hand_obj)

    return [1] + norm.flatten().tolist()


def process_frame(frame, frame_idx, fps, last_processed_timestamp, hand_detector, normalizer, target_vector_size):
    """Processes a single frame to detect hands, assign hand labels, normalize data and construct a vector representation.

    :param frame: Frame image from OpenCv.
    :param frame_idx: Current frame index.
    :param fps: Frames per second of the video.
    :param last_processed_timestamp: Timestamp of the last frame being processed.
    :param hand_detector: MediaPipe's hand detection.
    :param normalizer: Function to normalize the hand landmarks.
    :param target_vector_size: Expected length of the normalized coordinate features.
    :return: A vector representation of the hand landmarks and currently processed timestamp.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp_ms = int(frame_idx * 1000 / fps)
    if timestamp_ms <= last_processed_timestamp:
        timestamp_ms = last_processed_timestamp + 1

    last_processed_timestamp = timestamp_ms

    result = hand_detector.detect_for_video(
        mp_image,
        timestamp_ms=timestamp_ms
    )

    c_l, c_r = assign_hands(result)

    l_vec = build_frame_vector(c_l, target_vector_size, normalizer)
    r_vec = build_frame_vector(c_r, target_vector_size, normalizer)

    frame_vec = l_vec + r_vec

    return frame_vec, last_processed_timestamp


def process_video(video_path, hand_detector, normalizer, target_vector_size, target_frames_count=24):
    """Samples frames uniformly from a video file, extracts hand landmarks, normalizes them and pads the sequence to target
    length.

    :param video_path: Path to video file.
    :param hand_detector: MediaPipe's hand detection.
    :param normalizer: Function to normalize the hand landmarks.
    :param target_vector_size: Expected length of the normalized coordinate features.
    :param target_frames_count: Number of frames to extract.
    :return: A 2D numpy array containing normalized hand landmarks across the sampled frames.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Empty video: {video_path}")
        return None

    frame_indices = np.linspace(0, total_frames - 1, target_frames_count, dtype=int)

    last_processed_timestamp = -1
    all_landmarks = []
    last_valid = None

    for t, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()

        if not success:
            if last_valid is not None:
                all_landmarks.append(last_valid.copy())
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_vec, last_processed_timestamp = process_frame(
            frame, frame_idx, fps, last_processed_timestamp, hand_detector, normalizer, target_vector_size
        )

        all_landmarks.append(frame_vec)
        last_valid = frame_vec

    cap.release()
    hand_detector.close()

    final_vector_size = (target_vector_size + 1) * 2

    # Pad to target frames
    while len(all_landmarks) < target_frames_count:
        if last_valid is not None:
            all_landmarks.append(last_valid.copy())
        else:
            all_landmarks.append([0] * final_vector_size)

    return np.array(all_landmarks)