import cv2
import mediapipe as mp
import keyboard
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# OpenCV capture
cap = cv2.VideoCapture(0)

# Cooldown state for play/pause gesture
play_pause_triggered = False

def calculate_distance(landmark1, landmark2):
    """Calculate the Euclidean distance between two landmarks."""
    x1, y1 = landmark1.x, landmark1.y
    x2, y2 = landmark2.x, landmark2.y
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def detect_gesture(hand_landmarks):
    global play_pause_triggered

    # Get the thumb tip and index finger tip landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Calculate the distance between thumb tip and index tip
    distance = calculate_distance(thumb_tip, index_tip)

    # Threshold for detecting if thumb and index tip are touching
    distance_threshold = 0.04  # Adjust this value depending on your needs
    # print(f"Distance between thumb and index tip: {distance:.3f}")  # Debug print

    # Check for play/pause gesture (thumb and index finger tips touching)
    if distance < distance_threshold and not play_pause_triggered:
        print("Thumb and index finger tips are touching. Triggering play/pause.")  # Debug print
        keyboard.send("space")  # Send spacebar press to simulate play/pause
        play_pause_triggered = True  # Set the flag to prevent continuous triggering
    elif distance >= distance_threshold:  # Reset play/pause trigger when they are not touching
        play_pause_triggered = False

    # Detect Volume Up gesture (all fingers up)
    fingers = [
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y
    ]
    
    if all(fingers):
        print("All fingers are up. Triggering volume up.")
        keyboard.send("volume up")
    
    # Detect Volume Down gesture (thumb and pinky up, others down)
    elif fingers == [True, False, False, False, True]:
        print("Thumb and pinky are up. Triggering volume down.")
        keyboard.send("volume down")

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
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect gesture and control media
            detect_gesture(hand_landmarks)

    # Display the frame
    cv2.imshow("Hand Gesture Media Controller", frame)

    # Break with 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()