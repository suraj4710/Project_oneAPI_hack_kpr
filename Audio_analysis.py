import numpy as np
import librosa
import sounddevice as sd

# Parameters
sample_rate = 22050  # Sampling rate
duration = 5000  # Duration to record in milliseconds
pacing_thresholds = {'fast': 0.5, 'normal': 1.5}  # Example thresholds for pacing (seconds)
volume_thresholds = {'loud': 0.1, 'normal': 0.05}  # Example thresholds for volume (RMS)

def categorize_pacing(word_intervals):
    pacing = []
    for interval in word_intervals:
        if interval < pacing_thresholds['fast']:
            pacing.append('fast')
        elif interval > pacing_thresholds['normal']:
            pacing.append('slow')
        else:
            pacing.append('normal')
    return pacing

def categorize_pitch(pitch):
    if pitch < 200:
        return 'low'
    elif 200 <= pitch <= 400:
        return 'normal'
    else:
        return 'high'

def categorize_volume(rms):
    if rms > volume_thresholds['loud']:
        return 'loud'
    elif rms < volume_thresholds['normal']:
        return 'feeble'
    else:
        return 'normal'

def process_audio(indata, frames, time, status):
    # Convert audio to array
    audio = indata[:, 0]
    
    # Analyze RMS
    rms = np.sqrt(np.mean(audio**2))
    
    # Analyze pitch (mean pitch estimation)
    pitches, _ = librosa.piptrack(y=audio, sr=sample_rate)
    mean_pitch = np.mean(pitches)

    # Here you would also implement word segmentation to get intervals for pacing
    # For simplicity, let's assume you have a list of word intervals
    word_intervals = [0.5, 1.0, 1.2]  # Example intervals (in seconds)

    pacing = categorize_pacing(word_intervals)
    pitch_category = categorize_pitch(mean_pitch)
    volume_category = categorize_volume(rms)
    
    # Combine results
    print(f"Pacing: {pacing}, Pitch: {pitch_category}, Volume: {volume_category}")

# Stream audio in real-time
with sd.InputStream(callback=process_audio, channels=1, samplerate=sample_rate):
    sd.sleep(duration)
