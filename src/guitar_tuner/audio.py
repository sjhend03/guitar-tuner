import numpy as np
import sounddevice as sd

def record_audio(duration: float = 3, fs: int = 44100) -> np.ndarray:
    """pip
    duration: how many seconds to record
    fs: sampling frequency e.g. 44100, the signal gets measured 44100 a second
    """
    print("Starting to record")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float64") # Total samples are now duration * fs
    sd.wait() # Wait until all samples are recorded
    print("Done recording")
    return audio[:, 0]
