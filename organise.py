import os
import shutil

# Update these paths appropriately
DATASET_PATH = r"D:\Varun's documents\ISEC\Indian Emotional Speech Corpora (IESC)"
OUTPUT_PATH = r"D:\Varun's documents\ISEC\Organized"

os.makedirs(OUTPUT_PATH, exist_ok=True)

def organize_file(file_path, emotion_label):
    emotion_folder = os.path.join(OUTPUT_PATH, emotion_label)
    os.makedirs(emotion_folder, exist_ok=True)
    base_name = os.path.basename(file_path)
    dest_path = os.path.join(emotion_folder, base_name)
    count = 1
    while os.path.exists(dest_path):
        name, ext = os.path.splitext(base_name)
        dest_path = os.path.join(emotion_folder, f"{name}_{count}{ext}")
        count += 1
    shutil.copy(file_path, dest_path)

# A map of emotion keywords in filenames or folder names to emotion labels
emotion_map = {
    "A": "Angry",
    "H": "Happy",
    "S": "Sad",
    "N": "Neutral",
    "F": "Fear",
}

# Walk through dataset directory recursively
for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith(".wav"):
            file_path = os.path.join(root, file)
            # Infer emotion label from folder name or filename
            emotion_label = None
            # Check folder names from bottom up for known emotion labels
            for emotion_keyword in emotion_map:
                if emotion_keyword in root.lower():
                    emotion_label = emotion_map[emotion_keyword]
                    break
            # If not found in folder path, check filename
            if emotion_label is None:
                for emotion_keyword in emotion_map:
                    if emotion_keyword in file.lower():
                        emotion_label = emotion_map[emotion_keyword]
                        break
            # Default to 'Unknown' if no label found
            if emotion_label is None:
                emotion_label = "Unknown"
            organize_file(file_path, emotion_label)

print("✅ Dataset organization complete!")
