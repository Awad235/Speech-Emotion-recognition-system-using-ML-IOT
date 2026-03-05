import sounddevice as sd
import numpy as np
import wave
import speech_recognition as sr
import google.generativeai as genai
from sys import argv

import RPi.GPIO as GPIO
import time
import os

# ---------------- EMOTION LABELS ----------------
label_conversion = {
    0: 'neutral',
    1: 'calm',
    2: 'happy',
    3: 'sad',
    4: 'angry',
    5: 'fearful',
    6: 'disgust',
    7: 'surprised'
}

# --------------- CONFIGURE GEMINI -----------------
genai.configure(api_key="YOUR_API_KEY_HERE")
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

############ LCD #################
LCD_RS = 21
LCD_E  = 20
LCD_D4 = 6
LCD_D5 = 13
LCD_D6 = 19
LCD_D7 = 26

LCD_WIDTH = 20        # Max characters per line
LCD_CHR = True
LCD_CMD = False

LCD_LINE_1 = 0x80     # 1st line
LCD_LINE_2 = 0xC0     # 2nd line
LCD_LINE_3 = 0x94     # 3rd line
LCD_LINE_4 = 0xD4     # 4th line

E_PULSE = 0.0005
E_DELAY = 0.0005

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

GPIO.setup(LCD_E, GPIO.OUT)
GPIO.setup(LCD_RS, GPIO.OUT)
GPIO.setup(LCD_D4, GPIO.OUT)
GPIO.setup(LCD_D5, GPIO.OUT)
GPIO.setup(LCD_D6, GPIO.OUT)
GPIO.setup(LCD_D7, GPIO.OUT)

def lcd_toggle_enable():
    time.sleep(E_DELAY)
    GPIO.output(LCD_E, True)
    time.sleep(E_PULSE)
    GPIO.output(LCD_E, False)
    time.sleep(E_DELAY)

def lcd_byte(bits, mode):
    GPIO.output(LCD_RS, mode)

    GPIO.output(LCD_D4, bool(bits & 0x10))
    GPIO.output(LCD_D5, bool(bits & 0x20))
    GPIO.output(LCD_D6, bool(bits & 0x40))
    GPIO.output(LCD_D7, bool(bits & 0x80))
    lcd_toggle_enable()

    GPIO.output(LCD_D4, bool(bits & 0x01))
    GPIO.output(LCD_D5, bool(bits & 0x02))
    GPIO.output(LCD_D6, bool(bits & 0x04))
    GPIO.output(LCD_D7, bool(bits & 0x08))
    lcd_toggle_enable()

def lcd_init():
    lcd_byte(0x33, LCD_CMD)
    lcd_byte(0x32, LCD_CMD)
    lcd_byte(0x06, LCD_CMD)
    lcd_byte(0x0C, LCD_CMD)
    lcd_byte(0x28, LCD_CMD)
    lcd_byte(0x01, LCD_CMD)
    time.sleep(E_DELAY)

def lcd_string(message, line):
    message = message.ljust(LCD_WIDTH, " ")
    lcd_byte(line, LCD_CMD)
    for i in range(LCD_WIDTH):
        lcd_byte(ord(message[i]), LCD_CHR)

########### LED SETTINGS ###########
LED1 = 2
LED2 = 17
LED3 = 14
LED4 = 27
LED5 = 22
LED6 = 5
LED7 = 25
LED8 = 16

for led in [LED1,LED2,LED3,LED4,LED5,LED6,LED7,LED8]:
    GPIO.setup(led, GPIO.OUT)
    GPIO.output(led, False)


# -------- SAFE GEMINI TEXT EXTRACTOR ----------
def extract_gemini_text(resp):
    try:
        if hasattr(resp, "text") and resp.text:
            return resp.text
    except:
        pass
    try:
        if hasattr(resp, "candidates"):
            result = ""
            for c in resp.candidates:
                if hasattr(c, "content") and hasattr(c.content, "parts"):
                    for p in c.content.parts:
                        if hasattr(p, "text"):
                            result += p.text
            return result.strip()
    except:
        pass
    return ""

# ---------- RECORDING SETTINGS ----------
duration = 10
fs = 22050
channels = 1
filename = "input.wav"

recognizer = sr.Recognizer()
lcd_init()
lcd_byte(0x01, LCD_CMD)
lcd_string("speech emotion ", LCD_LINE_1)
lcd_string("recognition system", LCD_LINE_2)
time.sleep(1)

print("\nSpeak now...")
audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype=np.int16)
sd.wait()
print("Recording complete.")

with wave.open(filename, 'wb') as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(2)
    wf.setframerate(fs)
    wf.writeframes(audio_data.tobytes())

# ---------- SPEECH TO TEXT ----------
with sr.AudioFile(filename) as source:
    audio = recognizer.record(source)
try:
    transcript_text = recognizer.recognize_google(audio)
except:
    transcript_text = ""

print("\nYou said: >>>>>>>>>>>> {}".format(transcript_text))

# ---------- SEND TEXT TO GEMINI ----------
prompt = f"""
You are an AI that analyzes text and returns exactly one emotion label number
0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad', 4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'
Return only the number corresponding to the correct emotion. Text: "{transcript_text}"
"""

gemini_response = gemini_model.generate_content(prompt)
gemini_text = extract_gemini_text(gemini_response).strip()

if gemini_text.isdigit() and int(gemini_text) in label_conversion:
    label_number = int(gemini_text)
    label_name = label_conversion[label_number]
else:
    label_name = "unknown"

print("\nPredicted Emotion Label:")
print(label_name)
lcd_byte(0x01, LCD_CMD)
lcd_string("Detected emotion :", LCD_LINE_1)
lcd_string(label_name, LCD_LINE_2)

# ------------ LED INDICATOR ------------
if label_name == "calm":
    GPIO.output(LED4, True); time.sleep(1); GPIO.output(LED4, False)
elif label_name == "neutral":
    GPIO.output(LED3, True); time.sleep(1); GPIO.output(LED3, False)
elif label_name == "happy":
    GPIO.output(LED2, True); time.sleep(1); GPIO.output(LED2, False)
elif label_name == "angry":
    GPIO.output(LED1, True); time.sleep(1); GPIO.output(LED1, False)
elif label_name == "fearful":
    GPIO.output(LED8, True); time.sleep(1); GPIO.output(LED8, False)
elif label_name == "surprised":
    GPIO.output(LED7, True); time.sleep(1); GPIO.output(LED7, False)
elif label_name == "disgust":
    GPIO.output(LED6, True); time.sleep(1); GPIO.output(LED6, False)
elif label_name == "sad":
    GPIO.output(LED5, True); time.sleep(1); GPIO.output(LED5, False)

time.sleep(4)
