import whisper
import av
import numpy as np

def load_audio(path, target_sr=16000):
    container = av.open(path)
    resampler = av.audio.resampler.AudioResampler(
        format="fltp", layout="mono", rate=target_sr
    )
    samples = []
    for frame in container.decode(audio=0):
        for resampled in resampler.resample(frame):
            samples.append(resampled.to_ndarray().flatten())
    for resampled in resampler.resample(None):
        samples.append(resampled.to_ndarray().flatten())
    return np.concatenate(samples).astype(np.float32)

model = whisper.load_model("tiny")
audio = load_audio(r"C:\Users\rakesh.balla\Documents\Sound Recordings\test1.m4a")
result = model.transcribe(audio, language="en")
print(result["text"])
