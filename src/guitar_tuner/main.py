from guitar_tuner.audio import record_audio
from guitar_tuner.fft_pitch import estimate_frequency
from guitar_tuner.notes import (
    frequency_to_note,
    
    note_to_frequency,
    note_to_name,
    amount_off,
)

FS = 44100 # Sample frequency

def main():
    print("Play a note after prompt")

    signal = record_audio(duration=3, fs=FS)

    freq = estimate_frequency(signal, FS)

    if freq <= 0:
        print("No note played")
        return
    
    note = frequency_to_note(freq)

    target = note_to_frequency(note)

    name = note_to_name(note)

    error = amount_off(freq, target)

    print(f"Detected freq: {freq} Hz")
    print(f"Nearest note: {name}")
    print(f"Target: {target} Hz")
    print(f"Amount off: {error}")

if __name__ == "__main__":
    main()