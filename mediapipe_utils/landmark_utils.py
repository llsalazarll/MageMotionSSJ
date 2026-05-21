import numpy as np


def normalize_landmarks(landmarks):

    """
    Normalize landmarks relative to wrist position.
    Input:
        landmarks -> flat list of 63 values
    Output:
        normalized numpy array
    """

    landmarks = np.array(landmarks).reshape(21, 3)

    # Wrist landmark
    wrist = landmarks[0]

    # Subtract wrist from all points
    normalized = landmarks - wrist

    return normalized.flatten()