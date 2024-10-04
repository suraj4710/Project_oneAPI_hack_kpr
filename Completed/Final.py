import sounddevice as sd
import numpy as np
import librosa
import speech_recognition as sr
import signal
import sys
import threading
import cv2
from deepface import DeepFace
import mediapipe as mp

# Initialize the recognizer
recognizer = sr.Recognizer()

# Define thresholds for categorization
PACING_THRESHOLDS = {'fast': 0.3, 'normal': 0.5, 'slow': 0.7}  # Adjusted for pacing (seconds per word)
PITCH_THRESHOLDS = {'low': 250, 'normal': 500, 'high': 700}  # Adjusted frequency ranges (Hz)
VOLUME_THRESHOLDS = {'loud': -5, 'normal': -15, 'feeble': -25}  # Adjusted dB levels

# Disfluency words and thresholds
disfluent_words = ['uh', 'um', 'so', 'you know', 'like']

def analyze_audio(audio):
    # Calculate volume in dB
    volume = 20 * np.log10(np.sqrt(np.mean(audio**2))) if np.any(audio) else -np.inf
    volume_category = categorize_volume(volume)

    # Analyze pitch
    pitches, magnitudes = librosa.piptrack(y=audio, sr=22050)
    pitch_freq = np.mean(pitches[magnitudes > 0]) if magnitudes.size > 0 else 0  # Get the average pitch frequency
    pitch_category = categorize_pitch(pitch_freq)

    # Analyze pacing
    pacing = analyze_pacing(audio)
    pacing_category = categorize_pacing(pacing)

    return volume_category, pitch_category, pacing_category

def categorize_volume(volume):
    if volume > VOLUME_THRESHOLDS['loud']:
        return 'loud'
    elif volume > VOLUME_THRESHOLDS['normal']:
        return 'normal'
    else:
        return 'feeble'

def categorize_pitch(pitch):
    if pitch < PITCH_THRESHOLDS['low']:
        return 'low'
    elif pitch > PITCH_THRESHOLDS['high']:
        return 'high'
    else:
        return 'normal'

def categorize_pacing(pacing):
    if pacing < PACING_THRESHOLDS['fast']:
        return 'fast'
    elif pacing < PACING_THRESHOLDS['normal']:
        return 'normal'
    else:
        return 'slow'

def analyze_pacing(audio):
    # Example logic for pacing: Count the number of words per second (mock implementation)
    words_count = len(audio) // 1000  # Mock word count
    return words_count  # Return the pacing in words per second

def check_disfluency(text):
    disfluency_count = 0
    words = text.split()
    for word in words:
        if word in disfluent_words:
            disfluency_count += 1
    return disfluency_count

def analyze_speech():
    microphone = sr.Microphone()

    with microphone as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)

        while True:
            print("Listening... Speak now!")
            try:
                # Listen for audio
                audio_data = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio_data)
                print(f"\nSpeech-to-Text: {text}")

                # Disfluency analysis from speech-to-text
                disfluency_count = check_disfluency(text)
                print(f"Disfluency Count: {disfluency_count}")

                # Convert audio_data to numpy array for analysis
                audio_numpy = np.frombuffer(audio_data.get_raw_data(), np.int16).astype(np.float32)

                # Analyze audio for pitch, volume, and pacing
                volume_category, pitch_category, pacing_category = analyze_audio(audio_numpy)
                
                # Print results separately for clarity
                print(f"Volume: {volume_category}")
                print(f"Pitch: {pitch_category}")
                print(f"Pacing: {pacing_category}")

            except sr.UnknownValueError:
                print("Sorry, I couldn't understand what you said.")
            except sr.RequestError:
                print("There seems to be an issue with the Google Web Speech API.")
            except sr.WaitTimeoutError:
                print("Listening timed out, no speech detected. Continuing to listen...")
            except KeyboardInterrupt:
                print("\nStopped listening.")
                break

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

            prev_landmarks = landmarks  # Store landmarks for next frame analysis

        # Display the video feed with body language analysis
        cv2.imshow('Body Language Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    # Start video capture
    cap = cv2.VideoCapture(0)

    # Create threads for emotion detection and body language detection
    emotion_thread = threading.Thread(target=detect_emotion, args=(cap,))
    body_language_thread = threading.Thread(target=detect_body_language, args=(cap,))

    # Start the threads
    emotion_thread.start()
    body_language_thread.start()

    # Start the speech analysis thread
    speech_thread = threading.Thread(target=analyze_speech)
    speech_thread.start()

    # Join threads
    emotion_thread.join()
    body_language_thread.join()
    speech_thread.join()

    # Release resources at the end
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
