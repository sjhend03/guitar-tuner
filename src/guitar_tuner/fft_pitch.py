import numpy as np
from scipy.signal import windows

def estimate_frequency(signal: np.ndarray, fs: int) -> float:
    """
    signal: recorded audio sample
    fs: sample rate (times per second)
    """
    if len(signal) == 0:
        return 0.0
    
    windowed = signal * windows.hann(len(signal)) # Windows keep everything equal, sometimes the recorded sample is not able to be periodically cut up perfectly
    spectrum = np.fft.rfft(windowed) # Real fft transform (its what I use in research so I assume it may be best here)

    freqs = np.fft.rfftfreq(len(windowed), d=1 / fs) # Builds frequency axis, d is the spacing between the samples
    magnitude = np.abs(spectrum) # We do not care about phase, just magnitude

    mask = (freqs >= 70) & (freqs <= 400) # mask freqs to be only the ones in range of the guitar strings
    # Low E = 82 Hz, High E = 330 Hz

    if not np.any(mask):
        return 0.0
    
    masked_freqs = freqs[mask]
    masked_magnitudes = magnitude[mask]

    peak_idx = np.argmax(masked_magnitudes) # Find biggest magnitude e.g. the note being played

    return float(masked_freqs[peak_idx])

