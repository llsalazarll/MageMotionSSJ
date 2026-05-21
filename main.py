import cv2
import numpy as np

from mediapipe_utils.hand_tracker import HandTracker
from realtime.gamepad_controller import GamepadController
from realtime.smoothing import EMAFilter

# =========================================
# Webcam Setup
# =========================================
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# =========================================
# Initialize Components
# =========================================
tracker = HandTracker()

controller = GamepadController()

steering_filter = EMAFilter(alpha=0.25)
throttle_filter = EMAFilter(alpha=0.2)
brake_filter = EMAFilter(alpha=0.25)

# Neutral steering position
center_x = 0.28


# =========================================
# Fist Detection
# =========================================
def is_fist(hand):

    fingertip_ids = [8, 12, 16, 20]
    mcp_ids = [5, 9, 13, 17]

    curled_fingers = 0

    for tip, mcp in zip(fingertip_ids, mcp_ids):

        fingertip_y = hand[tip][1]
        mcp_y = hand[mcp][1]

        if fingertip_y > mcp_y:
            curled_fingers += 1

    return curled_fingers >= 3


# =========================================
# Main Loop
# =========================================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    processed_frame, hands_data = tracker.process_frame(frame)

    steering = 0.0
    throttle = 0.0
    brake = 0.0

    # =========================================
    # LEFT HAND -> STEERING
    # =========================================
    if "Left" in hands_data:

        left_hand = np.array(
            hands_data["Left"]["raw"]
        ).reshape(21, 3)

        left_fist = is_fist(left_hand)

        if not left_fist:

            palm_x = np.mean([
                left_hand[0][0],
                left_hand[5][0],
                left_hand[9][0],
                left_hand[13][0],
                left_hand[17][0]
            ])

            steering = (palm_x - center_x) * 5

            steering = np.clip(steering, -1, 1)

            if abs(steering) < 0.08:
                steering = 0.0

            steering = steering_filter.update(steering)

        else:

            steering = 0.0

            cv2.putText(
                processed_frame,
                "STEERING LOCK",
                (420, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

    # =========================================
    # RIGHT HAND -> THROTTLE / BRAKE
    # =========================================
    if "Right" in hands_data:

        right_hand = np.array(
            hands_data["Right"]["raw"]
        ).reshape(21, 3)

        right_fist = is_fist(right_hand)

        # -------------------------------------
        # BRAKE MODE
        # -------------------------------------
        if right_fist:

            throttle = 0.0
            brake = 1.0

            brake = brake_filter.update(brake)

            cv2.putText(
                processed_frame,
                "BRAKE",
                (520, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )

        # -------------------------------------
        # THROTTLE MODE
        # -------------------------------------
        else:

            palm_y = np.mean([
                right_hand[0][1],
                right_hand[5][1],
                right_hand[9][1],
                right_hand[13][1],
                right_hand[17][1]
            ])

            throttle = np.interp(
                palm_y,
                [0.35, 0.65],
                [1.0, 0.0]
            )

            throttle = np.clip(throttle, 0, 1)

            throttle = throttle_filter.update(throttle)

    # =========================================
    # Send Controls
    # =========================================
    controller.update_controls(
        steering=steering,
        throttle=throttle,
        brake=brake
    )

    # =========================================
    # UI TEXT
    # =========================================
    cv2.putText(
        processed_frame,
        f"Steering: {steering:.2f}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        processed_frame,
        f"Throttle: {throttle:.2f}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.putText(
        processed_frame,
        f"Brake: {brake:.2f}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 100, 255),
        2
    )

    # =========================================
    # Steering Visualization
    # =========================================
    bar_center = 640
    steering_length = int(steering * 300)

    cv2.line(
        processed_frame,
        (bar_center, 200),
        (bar_center + steering_length, 200),
        (0, 255, 0),
        10
    )

    cv2.circle(
        processed_frame,
        (bar_center + steering_length, 200),
        15,
        (0, 0, 255),
        -1
    )

    # =========================================
    # Throttle Visualization
    # =========================================
    throttle_height = int(throttle * 250)

    cv2.rectangle(
        processed_frame,
        (50, 550 - throttle_height),
        (100, 550),
        (255, 255, 0),
        -1
    )

    # =========================================
    # Brake Visualization
    # =========================================
    brake_height = int(brake * 250)

    cv2.rectangle(
        processed_frame,
        (150, 550 - brake_height),
        (200, 550),
        (0, 100, 255),
        -1
    )

    cv2.imshow(
        "Neural Racing Controller",
        processed_frame
    )

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# =========================================
# Cleanup
# =========================================
cap.release()
cv2.destroyAllWindows()