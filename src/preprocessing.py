import cv2
import numpy as np
import mediapipe as mp


def normalize_landmarks(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])

    # Pin wrist to origin and normalize scale
    coords -= coords[0]
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val

    return coords


def assign_hands(result):
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
    if hand_obj is None:
        return [0] + [0] * target_vector_size

    norm = normalizer(hand_obj)

    return [1] + norm.flatten().tolist()


def process_frame(frame, frame_idx, fps, last_processed_timestamp, hand_detector, normalizer, target_vector_size):
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