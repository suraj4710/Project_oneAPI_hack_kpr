import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace

# Initialize MediaPipe Pose for body language detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Start webcam capture
cap = cv2.VideoCapture(0)

# Threshold for body language classification
shaky_hand_threshold = 20
slouching_angle_threshold = 160

# Function to calculate the angle between three points
def calculate_angle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Function to classify posture
def classify_posture(landmarks):
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    
    slouch_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    
    if slouch_angle < slouching_angle_threshold:
        return "Slouched Posture"
    else:
        return "Upright Posture"

# Function to classify hand movements
def classify_hand_movements(landmarks, prev_landmarks):
    if prev_landmarks is None:
        return "Stable Hands"
    
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    prev_left_wrist = prev_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    
    movement_distance = np.sqrt((left_wrist.x - prev_left_wrist.x) ** 2 + (left_wrist.y - prev_left_wrist.y) ** 2)
    
    if movement_distance > shaky_hand_threshold:
        return "Shaky Hands"
    else:
        return "Stable Hands"

prev_landmarks = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for both MediaPipe and DeepFace
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ========== Emotion Detection using DeepFace ==========
    try:
        # Run DeepFace emotion analysis
        result = DeepFace.analyze(image_rgb, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result['dominant_emotion']
    except:
        dominant_emotion = "No face detected"

    # ========== Body Language Detection using MediaPipe Pose ==========
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Classify posture and hand movements
        posture_message = classify_posture(landmarks)
        hand_movement_message = classify_hand_movements(landmarks, prev_landmarks)

        # Draw landmarks on the frame
        for landmark in landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

        # Update previous landmarks for hand movement detection in the next frame
        prev_landmarks = landmarks

        # Add body language messages to the frame
        cv2.putText(frame, posture_message, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, hand_movement_message, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Add emotion detection result to the frame
    cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the combined frame with both emotion and body language results
    cv2.imshow('Emotion and Body Language Detection', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
