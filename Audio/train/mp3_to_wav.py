import os
from pydub import AudioSegment


def convert_mp3_to_wav(folder_path="."):
    """Convert all MP3 files in the given folder to WAV format."""
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".mp3"):
            mp3_path = os.path.join(folder_path, file_name)
            wav_path = os.path.splitext(mp3_path)[0] + ".wav"

            audio = AudioSegment.from_mp3(mp3_path)
            audio.export(wav_path, format="wav")
            print(f"Converted {file_name} to WAV format.")


if __name__ == "__main__":
    convert_mp3_to_wav()

