import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Start webcam capture
cap = cv2.VideoCapture(0)

# Threshold for classifying behaviors
shaky_hand_threshold = 20  # Example value, adjust as needed
slouching_angle_threshold = 160  # Angle threshold for slouching

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
    
    movement_distance = np.sqrt((left_wrist.x - prev_left_wrist.x) ** 2 + (left_wrist.y - prev_left_wrist.y) ** 2)
    
    if movement_distance > shaky_hand_threshold:
        return "Shaky Hands"
    else:
        return "Stable Hands"

prev_landmarks = None  # To track the previous frame's landmarks for hand movement analysis

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose detection
    results = pose.process(image_rgb)

    # Extract landmarks if available
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

        # Add the classification messages to the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, posture_message, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, hand_movement_message, (50, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Update previous landmarks for hand movement detection in the next frame
        prev_landmarks = landmarks

    # Display the frame with posture and hand movement messages
    cv2.imshow('Body Language Detection', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()