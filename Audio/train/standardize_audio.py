import os
import soundfile as sf
import numpy as np
import resampy


def standardize_audio(file_path, target_sample_rate=44100):
    # Load the WAV file with SoundFile
    data, current_sample_rate = sf.read(file_path, dtype='int16')

    # Convert to mono if it's stereo
    if len(data.shape) > 1 and data.shape[1] == 2:
        #data = np.mean(data, axis=1).astype(np.int16)
        data = data[:, 0]

    # Resample the audio if needed
    if current_sample_rate != target_sample_rate:
        data = resampy.resample(data, current_sample_rate, target_sample_rate, filter='kaiser_best')

    # Save the standardized audio back to the file
    sf.write(file_path, data, target_sample_rate, subtype='PCM_16')


def standardize_directory(directory_path):
    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        # Check if the file is a WAV file
        if filename.endswith(".wav"):
            file_path = os.path.join(directory_path, filename)
            print(f"Standardizing {file_path}...")
            standardize_audio(file_path)
    print("Standardization complete!")


if __name__ == "__main__":
    directory_path = os.getcwd()  # Gets the current directory
    standardize_directory(directory_path)
