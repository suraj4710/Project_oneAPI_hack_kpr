import sounddevice as sd
import numpy as np
import librosa
import signal
import sys
import speech_recognition as sr
import time

# Define thresholds for categorization
PACING_THRESHOLDS = {'fast': 4, 'normal': 6, 'slow': 8}  # Adjusted for words per minute
PITCH_THRESHOLDS = {'low': 300, 'normal': 500, 'high': 700}  # Example frequency ranges (Hz)
VOLUME_THRESHOLDS = {'loud': -10, 'normal': -20, 'feeble': -30}  # Example dB levels
DISFLUENCY_THRESHOLDS = {'frequent': 10, 'normal': 5, 'less': 0}  # Example counts for disfluencies

# Initialize counters
disfluency_count = 0
recognizer = sr.Recognizer()

# Initialize start time
start_time = time.time()

def analyze_audio(indata, frames, time, status):
    global disfluency_count, start_time
    audio = indata[:, 0]
    
    # Calculate volume in dB
    volume = 20 * np.log10(np.sqrt(np.mean(audio**2))) if np.any(audio) else -np.inf
    volume_category = categorize_volume(volume)

    # Analyze pitch
    pitches, magnitudes = librosa.piptrack(y=audio, sr=22050)
    pitch_freq = np.mean(pitches[magnitudes > 0]) if magnitudes.size > 0 else 0
    pitch_category = categorize_pitch(pitch_freq)

    # Convert audio to text
    audio_data = sr.AudioData(audio.tobytes(), 22050, 1)
    try:
        text = recognizer.recognize_google(audio_data)  # Using Google Web Speech API
        pacing_category = analyze_pacing(text)  # Calculate pacing based on text
        disfluencies = analyze_disfluencies(text)  # Count disfluencies in text
    except sr.UnknownValueError:
        text = ""
        pacing_category = 'normal'
        disfluencies = 0
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service")
        pacing_category = 'normal'
        disfluencies = 0

    # Update disfluency count
    disfluency_count += disfluencies
    disfluency_category = categorize_disfluencies(disfluency_count)

    print(f"Volume: {volume_category}, Pitch: {pitch_category}, Pacing: {pacing_category}, Disfluencies: {disfluency_category}")

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
    elif pitch < PITCH_THRESHOLDS['normal']:
        return 'normal'
    else:
        return 'high'

def categorize_pacing(pacing):
    if pacing < PACING_THRESHOLDS['fast']:
        return 'fast'
    elif pacing < PACING_THRESHOLDS['normal']:
        return 'normal'
    else:
        return 'slow'

def categorize_disfluencies(count):
    if count > DISFLUENCY_THRESHOLDS['frequent']:
        return 'frequent'
    elif count > DISFLUENCY_THRESHOLDS['normal']:
        return 'normal'
    else:
        return 'less'

def analyze_pacing(text):
    global start_time
    # Calculate pacing based on the number of words in the text
    words = text.split()
    word_count = len(words)
    
    # Calculate the time elapsed since the last audio chunk was processed
    elapsed_time = time.time() - start_time
    if elapsed_time < 1:  # Ensure at least 1 second has passed to avoid division by zero
        elapsed_time = 1.0

    pacing = (word_count / elapsed_time) * 60  # Words per minute
    start_time = time.time()  # Reset start time for the next analysis
    return categorize_pacing(pacing)

def analyze_disfluencies(text):
    # Count occurrences of common disfluencies in the text
    disfluency_words = ["um", "uh", "like", "you know"]
    return sum(text.lower().count(word) for word in disfluency_words)

def signal_handler(sig, frame):
    print("\nStopping the audio analysis.")
    sys.exit(0)

# Set up signal handler for graceful exit
signal.signal(signal.SIGINT, signal_handler)

# Set up audio input stream
with sd.InputStream(callback=analyze_audio, channels=1, samplerate=22050):
    print("Listening... Press Ctrl+C to stop.")
    while True:
        sd.sleep(1000)  # Keep the stream alive
