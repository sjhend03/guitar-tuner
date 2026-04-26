from guitar_tuner.audio import record_audio
from guitar_tuner.fft_pitch import get_data
from guitar_tuner.fft_pitch import estimate_frequency
import numpy as np
import matplotlib.pyplot as plt
import os

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
    
    #call function to plot amplitude and frequency graphs and save them to Plots folder
    ampl_and_freq_plot = estimate_frequency(signal, data[0], data[1], data[2], data[3])
    
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

    #we need to sort magnitudes in the same order as frequencies for interpolation
    norm_mask_magnitudes_sorted = norm_mask_magnitudes[idx]

    ampl_2d = np.interp(
        R_scaled.flatten(),
        masked_freqs_sorted,
        norm_mask_magnitudes_sorted
    ).reshape(R_scaled.shape)
    
    return ampl_2d

def visualize_domain_mapping(ampl_2d):
    """
    Visualize the mapping of the rescaled domains to the processed data.
    2D interpolation of a 1D spectrum onto a radial frequency field.
    """
    if ampl_2d is None:
        print("No amplitude data to visualize.")
        return
    
    plt.imshow(ampl_2d, cmap='viridis')
    plt.colorbar()
    plt.title("Mapped Amplitude in Frequency Space")
    plt.xlabel("Frequency (normalized)")
    plt.ylabel("Frequency (normalized)")
    
    #save the plot to tests/Plots/domain_mapping.png
    folder_path = "tests/Plots"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(os.path.join(folder_path, "domain_mapping.png"))
    plt.close()


# def generate_fractal(ampl_2d, beta=1):
#     R = produce_grid(ampl_2d.shape[0])

#     # fractal scaling
#     amplitude = ampl_2d / (R ** beta)

#     # random phase
#     phase = np.exp(2j * np.pi * np.random.rand(*amplitude.shape))

#     # complex spectrum
#     F = amplitude * phase

#     # inverse FFT
#     img = np.fft.ifft2(F).real

#     # normalize
#     img = (img - img.min()) / (img.max() - img.min())

#     plt.imshow(img, cmap='viridis')
#     plt.colorbar()
#     plt.title("Audio-driven fractal")
#     plt.show()

#     return img


if __name__ == "__main__":
    result = visualize_domain_mapping(map_domains(process_data()))