import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Constants
DATA_PATH = r":C:\Users\awadt\OneDrive\Desktop\MP Final\Indian_Emotion_datasets"
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_LEN = 200

def extract_features(file_path):
    """Extract MFCC features from an audio file."""
    try:
        # Load and normalize audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        y = y / (np.max(np.abs(y)) + 1e-9)
        
        # Extract MFCCs and deltas
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        feat = np.vstack([mfcc, delta, delta2]).T  # (timesteps, 3*N_MFCC)
        
        # Pad or trim to MAX_LEN
        if feat.shape[0] < MAX_LEN:
            feat = np.pad(feat, ((0, MAX_LEN - feat.shape[0]), (0, 0)), mode='constant')
        else:
            feat = feat[:MAX_LEN, :]
            
        # Normalize features
        feat = (feat - np.mean(feat, axis=0)) / (np.std(feat, axis=0) + 1e-9)
        return feat
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def collect_data():
    """Collect features and labels from the dataset."""
    X, y = [], []
    print("Processing audio files...")
    
    for speaker_dir in os.listdir(DATA_PATH):
        speaker_path = os.path.join(DATA_PATH, speaker_dir)
        if not os.path.isdir(speaker_path):
            continue
            
        for emotion_dir in os.listdir(speaker_path):
            emotion_path = os.path.join(speaker_path, emotion_dir)
            if not os.path.isdir(emotion_path):
                continue
                
            print(f"Processing {speaker_dir}/{emotion_dir}...")
            for file in os.listdir(emotion_path):
                if not file.lower().endswith('.wav'):
                    continue
                    
                file_path = os.path.join(emotion_path, file)
                features = extract_features(file_path)
                
                if features is not None:
                    X.append(features)
                    y.append(emotion_dir)

    return np.array(X), np.array(y)

# Collect and preprocess data
print("Starting data collection...")
X, y = collect_data()

# Convert labels to one-hot encoding
print("\nEncoding labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = to_categorical(y_encoded)

print("\nEmotion mapping:")
for i, emotion in enumerate(le.classes_):
    print(f"{emotion}: {i}")

# Split data
print("\nSplitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded
)

# Save processed data
print("\nSaving processed data...")
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
np.save("label_encoder_classes.npy", le.classes_)

print("\n✅ Preprocessing complete!")
print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print("\nLabel distribution:")
for emotion, count in zip(le.classes_, np.sum(y_onehot, axis=0)):
    print(f"{emotion}: {count}")