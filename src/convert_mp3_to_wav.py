import librosa
import soundfile as sf
import os

input_dir = "Data/mp3/cv-valid-dev"
output_dir = "Data/wav/cv-valid-dev"
os.makedirs(output_dir, exist_ok=True)

for file_name in os.listdir(input_dir):
    if file_name.endswith(".mp3"):
        mp3_path = os.path.join(input_dir, file_name)
        wav_path = os.path.join(output_dir, file_name.replace(".mp3", ".wav"))

        y, sr = librosa.load(mp3_path, sr=16000)  # Load and resample to 16kHz
        sf.write(wav_path, y, sr)
