import sounddevice as sd
import numpy as np
import librosa
import signal
import sys

# Define thresholds for categorization
PACING_THRESHOLDS = {'fast': 0.3, 'normal': 0.5, 'slow': 0.7}  # Adjusted for pacing (seconds per word)
PITCH_THRESHOLDS = {'low': 250, 'normal': 500, 'high': 700}  # Adjusted frequency ranges (Hz)
VOLUME_THRESHOLDS = {'loud': -5, 'normal': -15, 'feeble': -25}  # Adjusted dB levels
DISFLUENCY_THRESHOLDS = {'frequent': 8, 'normal': 3, 'less': 0}  # Adjusted counts for disfluencies

# Initialize counters
disfluency_count = 0

def analyze_audio(indata, frames, time, status):
    global disfluency_count
    audio = indata[:, 0]

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

    # Analyze disfluencies
    disfluencies = analyze_disfluencies(audio)
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
    elif pitch > PITCH_THRESHOLDS['high']:
        return 'high'
    else:
        return 'normal'  # Default to normal if it's between low and high

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

def analyze_pacing(audio):
    # Example logic for pacing: Count the number of words per second (mock implementation)
    words_count = len(audio) // 1000  # Mock word count
    return words_count  # Return the pacing in words per second

def analyze_disfluencies(audio):
    # Mock implementation for disfluencies (e.g., "um", "uh")
    return audio.size // 1000  # Mock disfluency count

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
