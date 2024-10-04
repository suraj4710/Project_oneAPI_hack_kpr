import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace

# Initialize MediaPipe Pose for body language detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Start webcam capture with higher resolution for better face detection
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Set width to 640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height to 480

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
# Add these variables at the beginning of your script
movement_history = []
frame_count = 0  # Count the number of frames for speed calculation
movement_threshold = 0.02  # Minimum movement distance to consider
speed_threshold = 0.05  # Speed threshold for determining shaky hands
shaky_hand_frames = 5  # Number of consecutive frames needed for shaky hands detection

# Function to classify hand movements
def classify_hand_movements(landmarks, prev_landmarks):
    global movement_history, frame_count
    if prev_landmarks is None:
        return "Stable Hands"

    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    prev_left_wrist = prev_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

    # Calculate movement distance
    movement_distance = np.sqrt((left_wrist.x - prev_left_wrist.x) ** 2 + (left_wrist.y - prev_left_wrist.y) ** 2)

    # Calculate speed of movement (distance over time)
    speed = movement_distance / (1 / 30)  # Assuming 30 FPS

    # Track the speed in movement history
    movement_history.append(speed)
    frame_count += 1

    # Remove old speeds if history exceeds the required frame count
    if frame_count > shaky_hand_frames:
        movement_history.pop(0)
        frame_count -= 1

    # Determine if hands are shaky based on the average speed over the history
    if len(movement_history) == shaky_hand_frames and np.mean(movement_history) > speed_threshold:
        return "Shaky Hands"
    
    return "Stable Hands"

prev_landmarks = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # =================== Emotion Detection ===================
    try:
        # Analyze the frame for emotions
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']

        # Display the emotion on the frame
        cv2.putText(frame, dominant_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    except Exception as e:
        print(f"Error analyzing frame: {e}")

    # =================== Body Language Detection ===================
    results = pose.process(frame)

    # Clone the original frame for body language display
    body_language_frame = frame.copy()

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Classify posture and hand movements
        posture_message = classify_posture(landmarks)
        hand_movement_message = classify_hand_movements(landmarks, prev_landmarks)

        # Draw landmarks on the body language frame
        for landmark in landmarks:
            x = int(landmark.x * body_language_frame.shape[1])
            y = int(landmark.y * body_language_frame.shape[0])
            cv2.circle(body_language_frame, (x, y), 5, (255, 0, 0), -1)

        # Update previous landmarks for hand movement detection in the next frame
        prev_landmarks = landmarks

        # Add body language messages to the body language frame
        cv2.putText(body_language_frame, posture_message, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(body_language_frame, hand_movement_message, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the video feed with emotion label and body language
    cv2.imshow('Emotion Detection', frame)
    cv2.imshow('Body Language Detection', body_language_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()