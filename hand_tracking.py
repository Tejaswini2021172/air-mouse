import cv2
import mediapipe as mp
import pyautogui
import math
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def main():
    model_path = "hand_landmarker.task"

    base_options = python.BaseOptions(model_asset_path=model_path)

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    screen_width, screen_height = pyautogui.size()

    print("Universal AirMouse - Stage 6")
    print("Index finger = Mouse")
    print("Pinch = Left Click")
    print("Fist = Play/Pause")
    print("Two fingers = Scroll")
    print("Swipe = Next/Previous")
    print("Press Q to exit.")

    timestamp = 0

    last_click_time = 0
    last_media_time = 0
    last_swipe_time = 0

    click_cooldown = 0.7
    media_cooldown = 1.0
    swipe_cooldown = 1.0

    swipe_start_x = None
    swipe_start_time = None

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp += 1

        result = detector.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks:

            landmarks = result.hand_landmarks[0]

            index_finger = landmarks[8]
            thumb = landmarks[4]

            index_x = int(index_finger.x * frame.shape[1])
            index_y = int(index_finger.y * frame.shape[0])

            thumb_x = int(thumb.x * frame.shape[1])
            thumb_y = int(thumb.y * frame.shape[0])

            screen_x = int(index_finger.x * screen_width)
            screen_y = int(index_finger.y * screen_height)

            pyautogui.moveTo(screen_x, screen_y)

            current_time = time.time()

            if swipe_start_x is None:
                swipe_start_x = index_finger.x
                swipe_start_time = current_time

            swipe_distance = index_finger.x - swipe_start_x
            swipe_time = current_time - swipe_start_time

            if (
                swipe_time < 0.8
                and current_time - last_swipe_time > swipe_cooldown
            ):

                if swipe_distance > 0.25:

                    pyautogui.press("right")

                    last_swipe_time = current_time
                    swipe_start_x = index_finger.x
                    swipe_start_time = current_time

                    cv2.putText(
                        frame,
                        "SWIPE RIGHT - NEXT",
                        (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        3
                    )

                elif swipe_distance < -0.25:

                    pyautogui.press("left")

                    last_swipe_time = current_time
                    swipe_start_x = index_finger.x
                    swipe_start_time = current_time

                    cv2.putText(
                        frame,
                        "SWIPE LEFT - PREVIOUS",
                        (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        3
                    )

            if swipe_time >= 0.8:
                swipe_start_x = index_finger.x
                swipe_start_time = current_time

            distance = math.hypot(
                index_x - thumb_x,
                index_y - thumb_y
            )

            if distance < 40:

                if current_time - last_click_time > click_cooldown:

                    pyautogui.click()

                    last_click_time = current_time

                cv2.putText(
                    frame,
                    "CLICK",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

            finger_tips = [8, 12, 16, 20]
            finger_pips = [6, 10, 14, 18]

            folded_fingers = 0

            for tip, pip in zip(finger_tips, finger_pips):

                if landmarks[tip].y > landmarks[pip].y:
                    folded_fingers += 1

            if folded_fingers == 4:

                if current_time - last_media_time > media_cooldown:

                    pyautogui.press("playpause")

                    last_media_time = current_time

                cv2.putText(
                    frame,
                    "PLAY / PAUSE",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

            index_up = landmarks[8].y < landmarks[6].y
            middle_up = landmarks[12].y < landmarks[10].y
            ring_down = landmarks[16].y > landmarks[14].y
            pinky_down = landmarks[20].y > landmarks[18].y

            if index_up and middle_up and ring_down and pinky_down:

                pyautogui.scroll(2)

                cv2.putText(
                    frame,
                    "SCROLL UP",
                    (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

            cv2.circle(
                frame,
                (index_x, index_y),
                10,
                (0, 255, 0),
                -1
            )

            cv2.circle(
                frame,
                (thumb_x, thumb_y),
                10,
                (0, 255, 0),
                -1
            )

        else:
            swipe_start_x = None
            swipe_start_time = None

        cv2.imshow(
            "Universal AirMouse - Stage 6",
            frame
        )

        
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
          print("EMERGENCY STOP")
          break

        if key == ord("q"):
          break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()