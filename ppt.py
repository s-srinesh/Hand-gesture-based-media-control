import cv2
import mediapipe as mp
import keyboard
import math
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# OpenCV capture
cap = cv2.VideoCapture(0)

# Cooldown times (in seconds) for gestures
next_slide_cooldown = 1.0  # 1 second cooldown for "next slide"
prev_slide_cooldown = 1.0  # 1 second cooldown for "previous slide"

# Last trigger timestamps for gestures
last_next_slide_time = 0
last_prev_slide_time = 0

# Track finger state changes
prev_finger_states = [False, False]  # Only track thumb and index finger

def calculate_distance(landmark1, landmark2):
    """Calculate the Euclidean distance between two landmarks."""
    x1, y1 = landmark1.x, landmark1.y
    x2, y2 = landmark2.x, landmark2.y
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def get_finger_states(hand_landmarks):
    """
    Returns a list of True/False values for each of the thumb and index fingers,
    indicating whether the finger is up (True) or down (False).
    """
    # Thumb: Check if the thumb tip is above the thumb MCP (Metacarpophalangeal joint)
    thumb_up = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
    # Index finger: Check if the index tip is above the index DIP (Distal Interphalangeal joint)
    index_up = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
    
    return [thumb_up, index_up]

def detect_gesture(hand_landmarks, hand_label):
    global last_next_slide_time, last_prev_slide_time, prev_finger_states

    # Get finger states (True for up, False for down)
    finger_states = get_finger_states(hand_landmarks)

    current_time = time.time()

    if hand_label == "Right":  # Only trigger gestures for right hand
        # Check for "Next Slide" gesture (thumb and index fingers up)
        if finger_states[0] and finger_states[1]:  # Thumb and index up
            if not prev_finger_states[0] and not prev_finger_states[1]:
                if current_time - last_next_slide_time > next_slide_cooldown:
                    print("Right hand with thumb and index up. Triggering next slide.")
                    keyboard.send("right")  # Send right arrow key to simulate next slide
                    last_next_slide_time = current_time  # Update last trigger time
            prev_finger_states[0] = True
            prev_finger_states[1] = True
        else:
            if prev_finger_states[0] or prev_finger_states[1]:
                prev_finger_states[0] = False
                prev_finger_states[1] = False

    elif hand_label == "Left":  # Only trigger gestures for left hand
        # Check for "Previous Slide" gesture (thumb and index fingers up)
        if finger_states[0] and finger_states[1]:  # Thumb and index up
            if not prev_finger_states[0] and not prev_finger_states[1]:
                if current_time - last_prev_slide_time > prev_slide_cooldown:
                    print("Left hand with thumb and index up. Triggering previous slide.")
                    keyboard.send("left")  # Send left arrow key to simulate previous slide
                    last_prev_slide_time = current_time  # Update last trigger time
            prev_finger_states[0] = True
            prev_finger_states[1] = True
        else:
            if prev_finger_states[0] or prev_finger_states[1]:
                prev_finger_states[0] = False
                prev_finger_states[1] = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for natural interaction
    frame = cv2.flip(frame, 1)
    # Convert the frame color to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame with MediaPipe Hands
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            # Get the label of the hand (Left or Right)
            hand_label = hand_info.classification[0].label

            # Draw the hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect gesture and control media (slide controls here)
            detect_gesture(hand_landmarks, hand_label)

    # Display the frame
    cv2.imshow("Hand Gesture Google Slides Controller", frame)

    # Break with 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
