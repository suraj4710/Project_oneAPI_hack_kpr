import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
import threading

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Thresholds for behavior classification
shaky_hand_threshold = 15  # Example value, adjust as needed
slouching_angle_threshold = 170  # Angle threshold for slouching
shaky_hand_movement_threshold = 0.03  # Example threshold for larger vibrations in hand movement
hand_movement_buffer = []

# Function to capture video and detect emotion
def detect_emotion(cap):
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Analyze the frame for emotions
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']

            # Display the emotion on the frame
            cv2.putText(frame, dominant_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        except Exception as e:
            print(f"Error analyzing frame: {e}")

        # Display the video feed with emotion label
        cv2.imshow('Emotion Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def calculate_angle(p1, p2, p3):
    """
    Helper function to calculate the angle between three points.
    p1, p2, p3 are landmarks with x, y coordinates.
    """
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))

    if angle > 180.0:
        angle = 360 - angle

    return angle

def classify_posture(landmarks):
    """
    Classify if the person's posture is upright or slouched based on shoulder and hip positions.
    """
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

    # Calculate the angle between the shoulder, hip, and knee (to check for slouching)
    slouch_angle = calculate_angle(left_shoulder, left_hip, left_knee)

    if slouch_angle < slouching_angle_threshold:
        return "Slouched Posture"
    else:
        return "Upright Posture"

def classify_hand_movements(landmarks, prev_landmarks):
    """
    Classify if the hands are shaking based on hand movement between consecutive frames.
    """
    if prev_landmarks is None:
        return "Stable Hands"

    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    prev_left_wrist = prev_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

    # Calculate hand movement distance
    movement_distance = np.sqrt((left_wrist.x - prev_left_wrist.x) ** 2 + (left_wrist.y - prev_left_wrist.y) ** 2)

    # Add the movement to the buffer
    hand_movement_buffer.append(movement_distance)

    # Keep the buffer size limited to the last 10 frames
    if len(hand_movement_buffer) > 10:
        hand_movement_buffer.pop(0)

    # Check for frequent and rapid hand movements (shaking)
    if np.mean(hand_movement_buffer) > shaky_hand_movement_threshold:
        return "Shaky Hands - Nervous"
    else:
        return "Stable Hands"

def classify_steps(landmarks):
    """
    Classify if the person is stepping back or sideways (indicating avoidance).
    """
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Calculate the average position of the ankles and hips
    avg_ankle_x = (left_ankle.x + right_ankle.x) / 2
    avg_hip_x = (left_hip.x + right_hip.x) / 2

    if avg_hip_x < avg_ankle_x:
        return "Stepping Back - Avoidance"

    return "Stable Position"

def classify_shrugging(landmarks):
    """
    Classify if the person is shrugging their shoulders (indicating nervousness).
    """
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_lip = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value]

    # Calculate the distance between the lips and shoulders
    shoulder_distance = np.sqrt((left_shoulder.x - left_lip.x) ** 2 + (left_shoulder.y - left_lip.y) ** 2)
    relaxed_distance_threshold = 0.1  # Adjust this threshold as needed

    if shoulder_distance < relaxed_distance_threshold:
        return "Tensed Shoulders - Nervous"

    return "Relaxed Shoulders - Confident"

def classify_hand_position(landmarks):
    """
    Classify hand positioning (comfort or discomfort).
    """
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Calculate the distance of hands from hips
    left_hand_position = np.sqrt((left_wrist.x - left_hip.x) ** 2 + (left_wrist.y - left_hip.y) ** 2)
    right_hand_position = np.sqrt((right_wrist.x - right_hip.x) ** 2 + (right_wrist.y - right_hip.y) ** 2)

    if left_hand_position < 0.1 and right_hand_position < 0.1:
        return "Hands on Hips - Confident"
    elif left_hand_position > 0.1 and right_hand_position > 0.1:
        return "Fidgeting Hands - Discomfort"

    return "Neutral Hand Position"

def detect_body_language(cap):
    prev_landmarks = None  # To track the previous frame's landmarks for hand movement analysis

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform pose detection
        results = pose.process(image_rgb)

        # Extract landmarks if available
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Classify posture, hand movements, steps, shrugging, and hand positions
            posture_message = classify_posture(landmarks)
            hand_movement_message = classify_hand_movements(landmarks, prev_landmarks)
            stepping_message = classify_steps(landmarks)
            shrugging_message = classify_shrugging(landmarks)
            hand_position_message = classify_hand_position(landmarks)

            # Draw landmarks on the frame
            for landmark in landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

            # Add the classification messages to the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, posture_message, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, hand_movement_message, (50, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, stepping_message, (50, 150), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, shrugging_message, (50, 200), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, hand_position_message, (50, 250), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Update previous landmarks for hand movement detection in the next frame
            prev_landmarks = landmarks

        # Display the frame with all messages
        cv2.imshow('Body Language Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create and start threads for emotion detection and body language detection
emotion_thread = threading.Thread(target=detect_emotion, args=(cap,))
body_language_thread = threading.Thread(target=detect_body_language, args=(cap,))

emotion_thread.start()
body_language_thread.start()

emotion_thread.join()
body_language_thread.join()

# Release the capture once done
cap.release()
cv2.destroyAllWindows()