import numpy as np
import os
from scipy import signal
from scipy.signal import windows
import matplotlib.pyplot as plt

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

    #produce time axis for plotting (seconds)
    time = np.arange(len(signal)) / fs

    #plot aplitude and time graph and save it to Plots/amplitude_time.png
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal)
    plt.title("Time Spectrum")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend(["Audio Signal"])

    folder_path = "Plots"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(os.path.join(folder_path, "amplitude_time.png"))
    plt.close()

    #plot frequency and magnitude graph with peaks and save it to Plots/frequency_magnitude.png
    plt.figure(figsize=(10, 4))
    plt.plot(masked_freqs, masked_magnitudes)
    plt.scatter(masked_freqs[peak_idx], masked_magnitudes[peak_idx], color='red', s=100, label='Detected Peak')
    plt.title("Magnitude Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend(["Magnitude Spectrum", "Detected Peak"])
    plt.savefig(os.path.join(folder_path, "frequency_magnitude.png"))
    plt.close()

    return float(masked_freqs[peak_idx])

