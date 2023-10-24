import os
import sys
import pickle
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
import variables


# Function to process a single audio file
def process_audio_file(file_path, window_size, window_durations, model_cutoff, output_folder):
    initial_window_size = window_size
    sample_rate, X = wavfile.read(file_path)
    X = X.astype(np.float64)
    X = np.mean(X, axis=1)
    duration_s = X.shape[0] / sample_rate

    if duration_s > window_durations[-2]:
        window_durations[-1] = duration_s

    # For cropping out lower frequencies like drums
    f_max = 4000
    f_min = 0

    # For restricting maximum width / more drastically reducing time resolution with longer windows
    _, T_base, _ = signal.spectrogram(X, sample_rate, scaling="spectrum", mode="magnitude",
                                      window=signal.windows.tukey(initial_window_size, alpha=0.4),
                                      noverlap=np.ceil(initial_window_size * 0.0))
    base_time_bins = T_base.shape[0]
    max_time_bins = 5 * base_time_bins

    # Iterate over each window duration and save the corresponding spectrogram
    for secs_per_img in window_durations:
        window_size = initial_window_size
        if duration_s < secs_per_img:
            break

        # Adjust frequency and time resolution based on window duration
        if secs_per_img >= model_cutoff:
            window_size *= 2
            # TODO: filter out loud frequencies

        overlap_multiplier = 2.0 / secs_per_img * 0.667
        if secs_per_img >= model_cutoff:
            overlap_multiplier *= 2
        N_overlap_adjusted = np.ceil(window_size * (1 - overlap_multiplier))
        if secs_per_img >= model_cutoff:
            N_overlap_adjusted *= 0.8
        window = signal.windows.tukey(window_size, alpha=0.4)

        F, T_adjusted, S_mag_adjusted = signal.spectrogram(X, sample_rate, scaling="spectrum", mode="magnitude",
                                                           window=window, noverlap=N_overlap_adjusted)
        num_time_bins = S_mag_adjusted.shape[1]
        if num_time_bins > max_time_bins:
            #print(f"Bins before: {num_time_bins}")
            # Reduce time resolution to fit within max_time_bins
            decimation_factor = int(np.ceil(num_time_bins / max_time_bins))
            S_mag_adjusted = signal.decimate(S_mag_adjusted, decimation_factor, axis=1, ftype='fir')
            num_time_bins = S_mag_adjusted.shape[1]

        # Crop frequency and select relevant time bins
        f_max_idx = np.argmin(np.abs(f_max - F)) + 1
        f_min_idx = np.argmin(np.abs(f_min - F))
        S_mag_crop = S_mag_adjusted[f_min_idx:f_max_idx + f_min_idx, :]
        F_crop = F[f_min_idx:f_max_idx + f_min_idx]
        img_width = min(np.argmin(np.abs(secs_per_img - T_adjusted)) + 1, 5 * int(2.0 / secs_per_img))
        #img_height = S_mag_crop.shape[0]

        # Ensure even dimensions for the decoder network
        if img_width % 2 == 1:
            img_width += 1

        # Save the spectrogram as a pickled object
        output_file_name = f"{os.path.basename(file_path).split('.')[0]}_{secs_per_img}s.pkl"
        output_file_path = os.path.join(output_folder, output_file_name)
        with open(output_file_path, "wb") as f:
            pickle.dump(S_mag_crop, f)

        num_freq_bins = S_mag_crop.shape[0]
        #print(f"Duration (secs_per_img): {secs_per_img}, Time bins: {num_time_bins}, Freq bins: {num_freq_bins}, Tot: {num_time_bins * num_freq_bins}")
        # Plot the spectrogram
        if single_file:
            # Optionally, you can apply a scaling factor to limit the range
            # c = 13  # If you want to keep the scaling factor
            # num_time_bins = min(num_time_bins, img_width * c)  # Apply scaling for the x-axis

            plt.figure(figsize=(20, 7))
            ax = plt.axes()
            plt.pcolormesh(T_adjusted[:num_time_bins], F_crop[:num_freq_bins], S_mag_crop[:, :num_time_bins],
                           shading='nearest', cmap='inferno')

            ax.set_yscale('linear')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')

            cbar = plt.colorbar(ax=ax)
            cbar.set_label('Amplitude (dB)')
            plt.show()

# Load variables
vars = variables.load_variables_from_json()
window_durations = vars["window_durations"]
model_cutoff = vars["model_cutoff"]
window_size = vars["window_size"]

# Determine whether to process a single file or all files in the directory
single_file = False
if len(sys.argv) > 1:
    single_file = True
    # Single file specified via command line argument
    audio_file_path = sys.argv[1]
    audio_files = [audio_file_path]
    output_folder = os.path.join(os.path.dirname(audio_file_path), os.path.basename(audio_file_path).split('.')[0])
    os.makedirs(output_folder, exist_ok=True)
else:
    # No file specified, process all .wav files in the 'Audio/train' directory
    audio_files_directory = 'Audio/train'
    audio_files = [os.path.join(audio_files_directory, f) for f in os.listdir(audio_files_directory) if f.endswith('.wav')]
    output_folder = os.path.join(audio_files_directory, 'spectrograms')
    os.makedirs(output_folder, exist_ok=True)

# Process each audio file
for audio_file in audio_files:
    process_audio_file(audio_file, window_size, window_durations, model_cutoff, output_folder)

variables.write_variables_to_json(vars)
