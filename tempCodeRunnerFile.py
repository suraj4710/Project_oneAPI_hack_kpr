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

def classify_hand_movements(landmarks, prev_landmarks, hand_movement_buffer):
    """
    Classify if the hands are shaking based on hand movement between consecutive frames.
    A buffer is used to track frequent and longer distance movements over time.
    """
    if prev_landmarks is None:
        return "Stable Hands", hand_movement_buffer

    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    prev_left_wrist = prev_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

    # Calculate the distance of hand movement between consecutive frames
    movement_distance = np.sqrt((left_wrist.x - prev_left_wrist.x) ** 2 + (left_wrist.y - prev_left_wrist.y) ** 2)

    # Store the movement distance in the buffer for checking frequent, rapid movements
    hand_movement_buffer.append(movement_distance)
    if len(hand_movement_buffer) > 10:  # Keep only the last 10 movements to check for consistent shaking
        hand_movement_buffer.pop(0)

    # Check if the hand has been moving frequently over longer distances
    frequent_movement = np.mean(hand_movement_buffer) > shaky_hand_threshold and np.max(hand_movement_buffer) > (shaky_hand_threshold * 1.5)

    if frequent_movement:
        return "Shaky Hands - Nervous", hand_movement_buffer
    else:
        return "Stable Hands", hand_movement_buffer

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
    Classify if the person is shrugging their shoulders (indicating nervousness or discomfort).
    """
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]

    shoulder_height = (left_shoulder.y + right_shoulder.y) / 2
    ear_height = (left_ear.y + right_ear.y) / 2

    # Stricter conditions for classifying shoulder positions
    if shoulder_height < ear_height:  # Shoulders very close to ears
        return "Shrugging Shoulders - Nervous"
    elif shoulder_height > (ear_height + 0.02) and shoulder_height < (ear_height + 0.05):  # Slightly away from relaxed position
        return "Shoulders Not Relaxed - Discomfort"
    else:  # Shoulders are in a relaxed position
        return "Relaxed Shoulders"

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
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()