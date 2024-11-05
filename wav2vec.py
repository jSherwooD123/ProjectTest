import torch
import pyaudio
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load pre-trained model and processor
model_name = "facebook/wav2vec2-large-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Audio recording parameters
RATE = 16000  # wav2vec2 requires 16kHz sampling rate
CHUNK = 1024  # Size of audio buffer
FORMAT = pyaudio.paInt16
CHANNELS = 1
DURATION = 5  # Duration of each recording in seconds

# Initialize PyAudio
audio = pyaudio.PyAudio()

def record_audio():
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []

    for _ in range(int(RATE / CHUNK * DURATION)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording complete.")
    stream.stop_stream()
    stream.close()

    # Convert to NumPy array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0  # Normalize between -1 and 1
    return audio_data

def transcribe(audio_data):
    # Process audio data with the processor
    inputs = processor(audio_data, sampling_rate=RATE, return_tensors="pt", padding=True)
    input_values = inputs.input_values

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

try:
    while True:
        audio_data = record_audio()
        transcription = transcribe(audio_data)
        print("Transcription:", transcription)
except KeyboardInterrupt:
    print("Exiting...")
finally:
    audio.terminate()

