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
    This function produces 2*2 frequency space grid of the frequencies and magnitudes. 
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

def rescale_domains():
    """
    This function rescales the domains of the processed data.
    """
    data = process_data()


    if data is None:
        print("No audio data was processed.")
        return None
    
    R = produce_grid()
    R_scaled = (R*(len(data[0])-1))/R.max() #Scale the grid to match the frequency range of the data
    return


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