# 🎤 Speech Emotion System using ML & IoT

A machine-learning powered Speech Emotion Recognition (SER) system capable of detecting eight human emotions—Happy, Sad, Angry, Fear, Disgust, Surprise, Calm, and Neutral—from audio input and displaying the results through an IoT-enabled real-time monitoring setup.
The system integrates signal processing, CNN–LSTM deep learning, and Raspberry Pi–based IoT hardware to create a complete end-to-end prototype.

⭐ Overview

Speech-based emotion recognition identifies the emotional state of a speaker by analyzing vocal features such as pitch, tone, and rhythm.
This project uses MFCC audio features and a hybrid CNN + LSTM model to classify emotions with high accuracy.
A Raspberry Pi is used for real-time emotion detection, visualization on a 16×2 LCD, and LED indication for each emotion.

🎯 Features

1. Detects 8 emotions: Happy, Sad, Angry, Fear, Disgust, Surprise, Calm, Neutral

2. Real-time audio capture using a microphone

3. CNN–LSTM deep learning model

4. IoT integration for remote monitoring

5. Output through LED indicators and 16×2 LCD

6. Raspberry Pi–based deployment

7. Achieved 97% model accuracy during testing

🧠 Machine Learning Model

- Feature Extraction: MFCC (Mel-Frequency Cepstral Coefficients)

- Model Architecture: CNN for spatial feature learning + LSTM for temporal pattern recognition

- Training Performance:

    - Accuracy: 97%

    - Stable training/validation convergence
      
    - Low loss and minimal overfitting

    - Strong classification performance in confusion matrix

🛠️ Hardware & Components

- Raspberry Pi 4

- USB / Digital Microphone

- 16×2 LCD Display

- LED indicators for emotion output

- Connecting wires, breadboard, power supply

🧩 System Workflow

1. Speech signal captured via microphone

2. Audio pre-processing & MFCC extraction

3. MFCC fed into trained CNN–LSTM model

4. Model predicts one of the 8 emotions

5. Output displayed:

    - Emotion label on LCD

    - Corresponding LED lights up

    - Optionally sent to IoT cloud dashboard (MQTT / HTTP)

📈 Results

- Successfully classified all 8 emotions in real time

- High accuracy during model training (97%)

- Raspberry Pi prototype responded instantly to speech input

- Reliable performance across multiple voice samples

- Clear display of results via LCD and LEDs

📌 Applications

- Smart home voice assistants

- Driver alertness / safety systems

- Healthcare and emotional monitoring

- Call-center analytics

- Human–robot interaction

- Intelligent IoT voice-based devices
