Real-Time Emotion, Body Language, and Speech Analysis Tool
Overview

This project integrates real-time emotion detection, body language analysis, and speech recognition with audio feature extraction (pitch, volume, pacing) into a single application. The tool uses various machine learning and computer vision libraries to analyze facial expressions, body posture, hand movements, and speech features from live input via a webcam and microphone.

Features

    Emotion Detection: Recognizes facial emotions (e.g., happy, sad, angry) using DeepFace.
    Body Language Analysis: Detects body posture, hand movements, stepping behavior, and other non-verbal cues using MediaPipe's pose detection.
    Speech-to-Text and Audio Analysis: Converts speech to text using Google Web Speech API and analyzes speech features (pitch, volume, pacing, and disfluencies) using Librosa and SoundDevice.
    Real-Time Processing: All tasks (emotion detection, body language, and speech analysis) run concurrently using Python threading for real-time feedback.

Technologies Used

    Intel oneAPI: Optimized for performance on Intel hardware.
    Intel-Optimized Libraries: Utilizes Intel NumPy for enhanced performance in numerical computations.
    OpenCV: Used for video capture and displaying the live video feed.
    MediaPipe: Performs pose detection to analyze body language and classify postures, hand movements, and gestures.
    DeepFace: Detects emotions based on facial expressions.
    SpeechRecognition: Handles speech-to-text conversion using the Google Web Speech API.
    Librosa: Extracts audio features such as pitch and volume from recorded speech.
    SoundDevice: Captures live audio from the microphone.
    NumPy: Used for audio processing and numerical calculations.
    Threading: Allows concurrent processing of video and audio streams for real-time analysis.
    Signal: Ensures the program exits gracefully upon user interruption.

Setup and Installation
  Prerequisites
  
    Python 3.8+
    An internet connection (for Google Web Speech API)
    A webcam and microphone

  Install Dependencies
  Before running the project, install the necessary Python libraries. You can install them using pip:

    pip install opencv-python mediapipe deepface SpeechRecognition librosa sounddevice intel-numpy

  Clone the Repository
  To clone the repository:

    git clone https://github.com/suraj4710/Project_oneAPI_hack_kpr.git
    cd Project_oneAPI_hack_kpr.git

  Running the Project
  Once you have cloned the repository and installed the required dependencies, you can run the program with the intelpython3 interpreter using:

    python Final.py

  Exiting the Program
  The program can be interrupted by pressing Ctrl+C in the terminal, which will stop all the processes gracefully.


How It Works

    Emotion Detection: The webcam captures video frames, and DeepFace processes each frame to detect facial emotions. The detected emotion is displayed on the screen.
    Body Language Analysis: MediaPipe's pose detection identifies key points on the body (e.g., shoulders, hips, wrists). The system classifies behaviors such as slouching, stepping back, or shaky hands.
    Speech Analysis: The microphone captures speech, which is converted into text using the Google Web Speech API. Additionally, audio features like pitch, volume, and pacing are extracted using Librosa, and disfluencies are counted based on predefined keywords.

    
