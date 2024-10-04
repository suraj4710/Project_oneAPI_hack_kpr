import sounddevice as sd
import numpy as np
import librosa
import speech_recognition as sr
import signal
import sys

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

def recognize_speech_and_analyze_audio():
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)
        
        while True:
            print("Listening... Speak now!")
            try:
                # Listen for speech
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

# Signal handler for graceful exit
def signal_handler(sig, frame):
    print("\nStopping the program.")
    sys.exit(0)

# Set up signal handler for graceful exit
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    try:
        recognize_speech_and_analyze_audio()
    except KeyboardInterrupt:
        print("\nProgram interrupted.")
