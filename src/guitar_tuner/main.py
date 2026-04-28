from guitar_tuner.audio import record_audio
from guitar_tuner.fft_pitch import (estimate_frequency, get_data)
from guitar_tuner.notes import (
    frequency_to_note,
    note_to_frequency,
    note_to_name,
    amount_off,
)
from spectral_fractals.fractal_test import map_domains, produce_grid
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

FS = 44100 # Sample rate HZ
BLOCK_SIZE = 1024 # Number of samples in data block used to display the data and do FFT on
DETECT_EVERY = 1
FRACTAL_EVERY = 1
MIN_RMS = 0.02 # Minimal root mean square to cut out silence/noise

latest = {"signal": np.zeros(BLOCK_SIZE), "freq": 0.0, "note": ""} # Shared buffer of the latest data

BUFFER_SECONDS = 2 # How long of a buffer to use in order to collect enough data to get a good fft
BUFFER_SIZE = int(FS * BUFFER_SECONDS)
audio_buffer = np.zeros(BUFFER_SIZE) # Actual buffer data
frame_count = {"n": 0}

# Plot setup
fig, axes = plt.subplots(3, 1, figsize=(10, 11))
fig.tight_layout(pad=3)

ax_wave, ax_fft, ax_fractal = axes

line_wave, = ax_wave.plot([], [], lw=0.8, color="steelblue")
line_fft,  = ax_fft.plot([], [], lw=1, color="darkorange")
peak_dot_fft, = ax_fft.plot([], [], 'ro', ms=8)

ax_wave.set_title("Waveform")
ax_wave.set_xlabel("Time (s)")
ax_wave.set_ylabel("Amplitude")

ax_fft.set_title("FFT Magnitude")
ax_fft.set_xlabel("Frequency (Hz)")
ax_fft.set_ylabel("Magnitude")
ax_fft.set_xlim(70, 400)

ax_fractal.set_title("Live Fractal")
ax_fractal.axis("off")
fractal_im = ax_fractal.imshow(
    np.zeros((128, 128)), cmap="turbo", animated=True,
    aspect="auto", vmin=0, vmax=1
)

GUITAR_NOTES = {
    "E2": 82.4, "A2": 110.0, "D3": 146.8, "G3": 196.0, "B3": 246.9, "E4": 329.6,
}
for name, freq in GUITAR_NOTES.items():
    ax_fft.axvline(x=freq, color="gray", linestyle="--", lw=0.8, alpha=0.6)
    ax_fft.text(freq, 0, name, fontsize=7, color="gray", ha="center", va="bottom")

note_text  = ax_wave.text(0.02, 0.90, "", transform=ax_wave.transAxes, fontsize=14, fontweight="bold")
freq_text  = ax_wave.text(0.02, 0.75, "", transform=ax_wave.transAxes, fontsize=11)
cents_text = ax_wave.text(0.02, 0.60, "", transform=ax_wave.transAxes, fontsize=11)

def audio_callback(indata, frames, time, status):
    # callback for matplotlib animation
    global audio_buffer
    sig = indata[:, 0]
    if np.sqrt(np.mean(sig**2)) >= MIN_RMS: # Make sure its not silence and update audio buffer
        latest["signal"] = sig.copy()
        audio_buffer = np.roll(audio_buffer, -len(sig))
        audio_buffer[-len(sig):] = sig

def update(frame):
    frame_count["n"] += 1
    sig = latest["signal"]

    # Always update waveform
    t = np.arange(len(sig)) / FS
    line_wave.set_data(t, sig)
    ax_wave.set_xlim(t[0], t[-1])
    ax_wave.set_ylim(-1, 1)

    if frame_count["n"] % DETECT_EVERY != 0:
        return

    rms = np.sqrt(np.mean(audio_buffer**2))
    if rms < MIN_RMS:
        return

    data = get_data(audio_buffer, FS)
    if data is None:
        return
    mfreqs, mmag, peak_idx, t = data
    freq, mfreqs, mmag, peak_idx = estimate_frequency(audio_buffer, mfreqs, mmag, peak_idx, t)

    if freq <= 0 or len(mfreqs) == 0:
        return

    note   = frequency_to_note(freq)
    target = note_to_frequency(note)
    name   = note_to_name(note)
    error  = amount_off(freq, target)

    note_text.set_text(f"Note: {name}")
    freq_text.set_text(f"Freq: {freq:.1f} Hz")
    cents_text.set_text(f"Cents: {error:+.1f}")

    line_fft.set_data(mfreqs, mmag)
    ax_fft.set_ylim(0, mmag.max() * 1.1 or 1)
    peak_dot_fft.set_data([mfreqs[peak_idx]], [mmag[peak_idx]])

    if frame_count["n"] % FRACTAL_EVERY == 0 and freq > 0:
        ampl_2d = map_domains((mfreqs, mmag, peak_idx, t))
        if ampl_2d is not None:
            # shape of ampl_2d is (gridpoints, gridpoints)
            R = produce_grid(ampl_2d.shape[0])
            R_shifted = np.fft.fftshift(R)
            R_shifted[R_shifted==0] = 1e-8 # handle division by zero
            fractal_scaling = ampl_2d * (1 / R_shifted**2.5)
            img = np.fft.ifft2(fractal_scaling)
            img_abs = np.abs(img)
            # normalize
            img_norm = (img_abs - img_abs.min()) / (img_abs.max() - img_abs.min())
            fractal_im.set_data(img_norm)

def main():
    # Make a stream in order to get data
    stream = sd.InputStream(samplerate=FS, channels=1, blocksize=BLOCK_SIZE, dtype="float32", callback=audio_callback)
    # Connect it with a matplotlib animation
    with stream:
        ani = animation.FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False)
        plt.show()

if __name__ == "__main__":
    main()