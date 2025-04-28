import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd

# Paths
input_audio_dir = "D:/Real-Time-Speech-Recognition-and-Translation/Data/wav"
transcript_csv_path = "D:/Real-Time-Speech-Recognition-and-Translation/Data/txt/cv-other-dev.csv"
output_audio_dir = "D:/Real-Time-Speech-Recognition-and-Translation/Data/processed_audio"
output_feature_dir = "D:/Real-Time-Speech-Recognition-and-Translation/Data/featured_extracted"
os.makedirs(output_audio_dir, exist_ok=True)
os.makedirs(output_feature_dir, exist_ok=True)

# Load transcript data
df = pd.read_csv(transcript_csv_path)
processed_data = []

# Max feature length (128 frames)
max_sequence_length = 128

for idx, row in df.iterrows():
    filename = row['filename']
    transcript = row['text']
    audio_path = os.path.join(input_audio_dir, filename)
    processed_audio_path = os.path.join(output_audio_dir, filename)
    mel_path = os.path.join(output_feature_dir, filename.replace('.wav', '.npy'))

    try:
        # Load and process audio
        y, sr = librosa.load(audio_path, sr=16000)
        y, _ = librosa.effects.trim(y)
        y = y / np.max(np.abs(y))  # Normalize

        # Save cleaned audio (optional)
        sf.write(processed_audio_path, y, sr)

        # Extract Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB scale

        # Truncate or pad the Mel spectrogram to match the desired sequence length (128)
        if mel_db.shape[1] > max_sequence_length:
            mel_db = mel_db[:, :max_sequence_length]  # Truncate to 128 frames
        elif mel_db.shape[1] < max_sequence_length:
            padding = np.zeros((mel_db.shape[0], max_sequence_length - mel_db.shape[1]))
            mel_db = np.concatenate((mel_db, padding), axis=1)  # Pad to 128 frames

        # Save the processed Mel spectrogram
        np.save(mel_path, mel_db)

        # Append metadata
        processed_data.append([mel_path, transcript])

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Save metadata linking features to transcripts
processed_df = pd.DataFrame(processed_data, columns=["mel_path", "transcript"])
processed_df.to_csv("mel_processed_dataset.csv", index=False)

print("✅ Mel spectrogram extraction complete!")
