import sounddevice as sd
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from collections import deque
import time
import threading
import queue

# Audio parameters
DURATION = 3  # seconds
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_LEN = 200

# Model parameters
MODEL_PATH = "emotion_model_fixed_v3.keras"
EMOTIONS = np.load("label_encoder_classes.npy")
SMOOTHING = 5  # Increased smoothing window
CONFIDENCE_THRESHOLD = 0.15  # 15% threshold for showing secondary emotions

# Audio buffer for continuous recording
audio_buffer = queue.Queue()
recording = True

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    audio_buffer.put(indata.copy())

def extract_features(y):
    """Extract MFCC features matching the training preprocessing."""
    try:
        # Normalize audio
        y = y / (np.max(np.abs(y)) + 1e-9)
        
        # Extract MFCCs and deltas
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        feat = np.vstack([mfcc, delta, delta2]).T
        
        # Pad or trim to MAX_LEN
        if feat.shape[0] < MAX_LEN:
            feat = np.pad(feat, ((0, MAX_LEN - feat.shape[0]), (0, 0)), mode='constant')
        else:
            feat = feat[:MAX_LEN, :]
            
        # Normalize features
        feat = (feat - np.mean(feat, axis=0)) / (np.std(feat, axis=0) + 1e-9)
        return feat[np.newaxis, ...]
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        return None

# Load model
print("Loading model...")
model = load_model(MODEL_PATH)

# Setup visualization
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Bar plot for current prediction
bars = ax1.bar(EMOTIONS, [0]*len(EMOTIONS))
ax1.set_ylim(0, 1)
ax1.set_ylabel("Probability")
ax1.set_title("Current Emotion Probabilities")

# Line plot for emotion history
history_length = 50
emotion_history = {emotion: deque([0]*history_length, maxlen=history_length) for emotion in EMOTIONS}
lines = {}
for emotion in EMOTIONS:
    line, = ax2.plot(range(history_length), emotion_history[emotion], label=emotion)
    lines[emotion] = line
ax2.set_ylim(0, 1)
ax2.set_xlabel("Time")
ax2.set_ylabel("Probability")
ax2.legend()
ax2.set_title("Emotion Probability History")
plt.tight_layout()

# Initialize smoothing queue
recent_preds = deque(maxlen=SMOOTHING)

def update_plots(probs):
    # Update bar plot
    for bar, p in zip(bars, probs):
        bar.set_height(p)
    
    # Update line plot
    for i, emotion in enumerate(EMOTIONS):
        emotion_history[emotion].append(probs[i])
        lines[emotion].set_ydata(emotion_history[emotion])
    
    fig.canvas.draw()
    fig.canvas.flush_events()

print("\nStarting real-time emotion recognition...")
print("Available emotions:", EMOTIONS)
print("Press Ctrl+C to stop.\n")

try:
    with sd.InputStream(channels=1, 
                       samplerate=SAMPLE_RATE,
                       callback=audio_callback):
        while recording:
            # Get audio from buffer
            audio_chunks = []
            start_time = time.time()
            
            while time.time() - start_time < DURATION:
                if not audio_buffer.empty():
                    chunk = audio_buffer.get()
                    audio_chunks.append(chunk)
                    
            if not audio_chunks:
                continue
                
            # Combine chunks and flatten
            audio = np.concatenate(audio_chunks).flatten()
            
            # Skip if audio is too quiet
            if np.max(np.abs(audio)) < 0.01:
                print("No audio detected, skipping...")
                continue

            # Extract features
            features = extract_features(audio)
            if features is None:
                print("Failed to extract features, skipping...")
                continue
                
            # Predict emotion
            pred = model.predict(features, verbose=0)[0]
            recent_preds.append(pred)
            
            # Average predictions for smoothing
            smoothed = np.mean(recent_preds, axis=0)
            emotion_idx = np.argmax(smoothed)
            
            # Calculate confidence scores
            confidence = smoothed[emotion_idx] * 100
            
            # Print results
            print(f"\nPredicted emotion: {EMOTIONS[emotion_idx]} ({confidence:.1f}% confidence)")
            
            # Show all emotions above threshold
            for idx, prob in enumerate(smoothed):
                if prob > CONFIDENCE_THRESHOLD:
                    print(f"{EMOTIONS[idx]}: {prob*100:.1f}%")
                    
            # Update visualization
            update_plots(smoothed)
            
except KeyboardInterrupt:
    recording = False
    print("\nStopped.")
    plt.close()