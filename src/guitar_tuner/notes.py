import math

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def frequency_to_note(freq: float) -> int:
    """
    Converts a frequency to a note
    Each octave doubles frequency so we need a logarithmic formula for calculating it
    it compares the frequency to A4 and then calculates how many semitones it is away
    and adds that to note 69, if the frequency is 440 (A4) then it will return 69
    """
    return round(69 + 12 * math.log2(freq / 440.0))

def note_to_frequency(note: int) -> float:
    """
    Converts note to a frequency, formula is jsut opposite of frequency_to_note function
    """
    return 440.0 * (2 ** ((note - 69) / 12))

def note_to_name(note: int) -> str:
    """
    Converts note to its note name, % 12 comes from the amount of semitones in an octave
    """
    note_name = NOTES[note % 12]
    octave = (note // 12) - 1
    return f"{note_name}{octave}"

def amount_off(freq: float, target_freq: float) -> float:
    """
    Returns how many cents off the freq is from tarrget_freq.
    cent -> 1/100 of a semitone
    """
    return 1200 * math.log2(freq / target_freq)