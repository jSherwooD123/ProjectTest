import torch
import pyaudio
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load pre-trained model and processor
model_name = "facebook/wav2vec2-large-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Audio recording parameters
RATE = 16000  # Sampling rate required by wav2vec
CHUNK_DURATION = 1  # Duration of each audio chunk in seconds (smaller = lower latency)
CHUNK = int(RATE * CHUNK_DURATION)  # Size of each audio chunk in samples
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Starting real-time transcription. Press Ctrl+C to stop.")

def transcribe_chunk(audio_chunk):
    # Convert to a NumPy array and normalize
    audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

    # Process with the Wav2Vec2 processor
    inputs = processor(audio_data, sampling_rate=RATE, return_tensors="pt", padding=True)
    input_values = inputs.input_values

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the logits to get the transcription
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

try:
    while True:
        audio_chunk = stream.read(CHUNK)  # Read a chunk of audio
        transcription = transcribe_chunk(audio_chunk)
        print("Transcription:", transcription)

except KeyboardInterrupt:
    print("\nStopping transcription...")

finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()


