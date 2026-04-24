import numpy as np
import os
from scipy import signal
from scipy.signal import windows
import matplotlib.pyplot as plt

def get_data(signal: np.ndarray, fs: int):
    """
    This function takes in the recoded audio signal and processes the data to get the 
    frequencies and magnitudes of the signal. It also applies a mask to only keep frequencies 
    in the range of the guitar strings (70-400 Hz). 
    It returns the masked frequencies, masked magnitudes, index of the peak frequency, and time axis for plotting.
    signal: recorded audio sample
    fs: sample rate (times per second)
    """
    if len(signal) == 0:
        return None
    
    windowed = signal * windows.hann(len(signal)) # Windows keep everything equal, sometimes the recorded sample is not able to be periodically cut up perfectly
    spectrum = np.fft.rfft(windowed) # Real fft transform (its what I use in research so I assume it may be best here)

    freqs = np.fft.rfftfreq(len(windowed), d=1 / fs) # Builds frequency axis, d is the spacing between the samples
    magnitude = np.abs(spectrum) # We do not care about phase, just magnitude

    mask = (freqs >= 70) & (freqs <= 400) # mask freqs to be only the ones in range of the guitar strings
    # Low E = 82 Hz, High E = 330 Hz

    if not np.any(mask):
        return None
    
    masked_freqs = freqs[mask]
    masked_magnitudes = magnitude[mask]
    
    peak_idx = int(np.argmax(masked_magnitudes)) # Find biggest magnitude e.g. the note being played

    #produce time axis for plotting (seconds)
    time = np.arange(len(signal)) / fs

    return masked_freqs, masked_magnitudes, peak_idx, time

def estimate_frequency(signal: np.ndarray, masked_freqs: np.ndarray, masked_magnitudes: np.ndarray, peak_idx: int, time: np.ndarray)-> float:
    """"
    This function plots the data obtained from get_data and saves the plots to the Plots folder. 
    It then returns the frequency of the detected note.
    """
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

