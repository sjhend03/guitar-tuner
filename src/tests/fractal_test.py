from guitar_tuner.audio import record_audio
from guitar_tuner.fft_pitch import get_data
import numpy as np

FS = 44100 # Sample frequency

def process_data():
    """
    Process audio data to extract pitch information.
    """
    signal = record_audio(duration=3, fs=FS)

    #Properly unpack the data from get_data and handle the case where it returns None
    data = get_data(signal, FS)
    if data is None:
        print("No note played")
        return
    
    masked_freqs, masked_magnitudes, peak_idx, time = data
    return data

def produce_grid(grid_points=256):
    """
    This function produces n*n frequency space grid of the frequencies and magnitudes. 
    The grid points are determined by the grid_points parameter.
    """
    #Produce grid in frequency space
    nx = np.fft.fftfreq(grid_points)
    ny = np.fft.fftfreq(grid_points)
    NX, NY = np.meshgrid(nx, ny) # [-0.5,0.5)

    #Define metric in frequency space 
    R = np.sqrt(NX**2 + NY**2) #[1e-8,0.707)
    #handle DivisionByZero error   
    R[0,0]=1e-8

    return R


def map_domains(data):
    """
    This function maps the rescaled domains to the processed data.
    Need to map freq. (1D) to (2D) grid.
    """
    #handle case where data is None
    if data is None:
        print("No audio data was processed.")
        return None
    
    masked_freqs, masked_magnitudes, peak_idx, time = data #unpack data

    R = produce_grid(grid_points=256)

    #handle case where R_scaled is None
    if R is None:
        print("No rescaled grid available.")
        return None
    
    #normalize magnitudes to be between 0 and 1
    norm_mask_magnitudes = masked_magnitudes / masked_magnitudes.max()

    #linear scaling of R_scaled to match the range of the normalized frequencies
    f_min = masked_freqs.min()
    f_max = masked_freqs.max()

    R_scaled = f_min + (R / R.max()) * (f_max - f_min)

    #to interpolate, masked freq need to be sorted 
    idx = np.argsort(masked_freqs)
    masked_freqs_sorted = masked_freqs[idx]

    ampl_2d = np.interp(
        R_scaled.flatten(),
        masked_freqs_sorted,
        norm_mask_magnitudes
    ).reshape(R_scaled.shape)
    
    return ampl_2d

if __name__ == "__main__":
    print("Testing produce_grid()...")
    grid = produce_grid()

    print("Testing process_data()...")
    print("Note: This will attempt to record audio for 3 seconds.")
    
    # Run the audio processing
    try:
        result = process_data()
        print("Process data executed")
    except Exception as e:
        print("An error occurred during audio processing")