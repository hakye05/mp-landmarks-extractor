import cv2
import numpy as np
import mediapipe as mp


# Hand landmark normalization method for 2D landmark
def normalize_hand_2d(hand_landmarks):
    coords = np.array([[lm.x, lm.y] for lm in hand_landmarks])

    # wrist -> origin
    coords -= coords[0]

    # scale normalize
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val

    return coords


# Hand landmark normalization method for 3D landmark
def normalize_hand_3d(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])

    # wrist -> origin
    coords -= coords[0]

    # scale normalize
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val

    return coords


# Hand label assigner
def assign_hands(result, prev_left, prev_right):
    current_left, current_right = None, None

    if not result.hand_landmarks:
        return None, None

    hands = result.hand_landmarks

    if prev_left is None and prev_right is None:
        for h in hands:
            if h[0].x < 0.5:
                current_left = h
            else:
                current_right = h
    else:
        for h in hands:
            wrist = h[0]

            d_l = (wrist.x - prev_left[0])**2 + (wrist.y - prev_left[1])**2 if prev_left else 1e9
            d_r = (wrist.x - prev_right[0])**2 + (wrist.y - prev_right[1])**2 if prev_right else 1e9

            if d_l < d_r:
                current_left = h
            else:
                current_right = h

    return current_left, current_right


# Frame vector builder
# For target_vector_size choose 42 for 2D and 63 for 3D landmarks
def build_frame_vector(hand_obj, target_vector_size, normalizer):
    if hand_obj is None:
        return [0] + [0] * target_vector_size, None

    norm = normalizer(hand_obj)
    wrist_pos = (hand_obj[0].x, hand_obj[0].y)

    return [1] + norm.flatten().tolist(), wrist_pos


# Process video
# For target_frames_count default is 24 frames
# target_vector_size use 42 to get final 86 for 2D and 63 to get final 128 for 3D landmarks
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
    prev_l, prev_r = None, None
    last_valid = None

    for t, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()

        if not success:
            if last_valid is not None:
                all_landmarks.append(last_valid.copy())
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        fps = cap.get(cv2.CAP_PROP_FPS)
        timestamp_ms = int(frame_idx * 1000 / fps)
        if timestamp_ms <= last_processed_timestamp:
            timestamp_ms = last_processed_timestamp + 1

        last_processed_timestamp = timestamp_ms

        result = hand_detector.detect_for_video(
            mp_image,
            timestamp_ms=timestamp_ms
        )

        c_l, c_r = assign_hands(result, prev_l, prev_r)

        l_vec, prev_l = build_frame_vector(c_l, target_vector_size, normalizer)
        r_vec, prev_r = build_frame_vector(c_r, target_vector_size, normalizer)

        frame_vec = l_vec + r_vec
        all_landmarks.append(frame_vec)
        last_valid = frame_vec

    cap.release()
    hand_detector.close()

    final_vector_size = (target_vector_size + 1) * 2
    # pad to target frames
    while len(all_landmarks) < target_frames_count:
        if last_valid is not None:
            all_landmarks.append(last_valid.copy())
        else:
            all_landmarks.append([0] * final_vector_size)

    return np.array(all_landmarks)