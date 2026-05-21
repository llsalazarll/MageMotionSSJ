import cv2
import mediapipe as mp

from mediapipe_utils.landmark_utils import normalize_landmarks


class HandTracker:

    def __init__(
            self,
            max_num_hands=2,
            detection_confidence=0.7,
            tracking_confidence=0.7
    ):

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

    def process_frame(self, frame):

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb_frame)

        hands_data = {}

        if results.multi_hand_landmarks and results.multi_handedness:

            for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
            ):

                label = handedness.classification[0].label

                raw_hand = []

                for lm in hand_landmarks.landmark:
                    raw_hand.extend([lm.x, lm.y, lm.z])

                normalized_hand = normalize_landmarks(raw_hand)

                hands_data[label] = {
                    "raw": raw_hand,
                    "normalized": normalized_hand
                }

                # Draw skeleton
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

        return frame, hands_data