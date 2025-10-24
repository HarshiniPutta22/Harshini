# %pip install vosk
# %pip install whisper
# %pip install sounddevice
# %pip install numpy
# %pip install scipy
# %pip install jiwer

!apt-get install -y ffmpeg

# Commented out IPython magic to ensure Python compatibility.
# %pip install vosk

# Commented out IPython magic to ensure Python compatibility.
# %pip install whisper

# Commented out IPython magic to ensure Python compatibility.
# %pip install sounddevice

# Commented out IPython magic to ensure Python compatibility.
# %pip install numpy

# Commented out IPython magic to ensure Python compatibility.
# %pip install scipy

# Commented out IPython magic to ensure Python compatibility.
# %pip install jiwer
# Commented out IPython magic to ensure Python compatibility.
# %%markdown
# ## Model Comparison: Vosk vs. Whisper
# 
# | Feature          | Vosk                                  | Whisper                                     |
# |------------------|---------------------------------------|---------------------------------------------|
# | Model Size       | Smaller models available (e.g., 50MB) | Larger models available (e.g., 1.5GB)       |
# | Accuracy         | Generally good, can vary by language  | High accuracy, multilingual                 |
# | Required Resources| CPU-friendly, less resource intensive | Can be CPU or GPU intensive, benefits from GPU |
# | Ease of Use      | Relatively easy to set up and use     | Easy to use with pre-trained models         |
# 
# **Preliminary Decision:**
# 
# Based on the comparison, **Vosk** appears to be more suitable for a real-time speech-to-text system on a typical user's machine with likely computational constraints. Its smaller model size and lower resource requirements make it more practical for real-time processing without dedicated high-end hardware like a powerful GPU. While Whisper offers higher accuracy and multilingual support, its larger size and potential for higher resource consumption might be a bottleneck for real-time performance on less powerful machines. We will proceed with evaluating Vosk first, and potentially explore Whisper if performance with Vosk is not satisfactory and resources allow.

# Commented out IPython magic to ensure Python compatibility.
# %pip install vosk soundfile

!wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
!unzip vosk-model-small-en-us-0.15.zip

from google.colab import files
uploaded = files.upload()

import os

audio_file = '/content/audio.wav.mp3'
if os.path.exists(audio_file):
    if audio_file.lower().endswith('.wav'):
        print(f"The audio file '{audio_file}' is already in WAV format.")
    elif audio_file.lower().endswith('.mp3'):
        print(f"The audio file '{audio_file}' is in MP3 format and needs to be converted to WAV.")
    else:
        print(f"The audio file '{audio_file}' is in an unknown format.")
else:
    print(f"The audio file '{audio_file}' does not exist.")

!apt-get install -y ffmpeg
!ffmpeg -i /content/audio.mp3 -ac 1 -ar 16000 /content/output.wav

import wave
import json
from vosk import Model, KaldiRecognizer

# Load the model
model = Model("vosk-model-small-en-us-0.15")

# Open the uploaded file (change filename if needed)
wf = wave.open("/content/output.wav", "rb")

# Setup recognizer
rec = KaldiRecognizer(model, wf.getframerate())

# Read and recognize
results = []
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        results.append(result.get("text", ""))

# Print the final transcript
print("Transcription:")
print(" ".join(results))
