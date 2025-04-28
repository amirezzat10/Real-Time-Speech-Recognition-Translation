import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from fpdf import FPDF

# --- Load CSV ---
csv_path = "mel_processed_dataset.csv"  # Update if needed
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
print("📋 Cleaned columns:", df.columns.tolist())

# Detect path column
path_col = [col for col in df.columns if 'path' in col.lower()][0]
print(f"✅ Using path column: {path_col}")

# --- Stats containers ---
durations = []
n_mels = []
transcript_lengths = []

# --- Collect statistics ---
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        mel_path = row[path_col]
        transcript = row["transcript"]
        
        mel_spec = np.load(mel_path)
        time_steps = mel_spec.shape[1]
        mel_bands = mel_spec.shape[0]
        
        duration_sec = time_steps * 0.01  # Assuming 10ms per frame

        durations.append(duration_sec)
        n_mels.append(mel_bands)
        transcript_lengths.append(len(transcript.split()))
    except Exception as e:
        print(f"❌ Error loading {mel_path}: {e}")

# --- Plotting ---
plt.figure(figsize=(10, 5))
plt.hist(durations, bins=40, color='skyblue', edgecolor='black')
plt.title("Audio Duration Distribution (seconds)")
plt.xlabel("Duration (s)")
plt.ylabel("Count")
plt.savefig("duration_hist.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.hist(transcript_lengths, bins=30, color='lightcoral', edgecolor='black')
plt.title("Transcript Length Distribution (words)")
plt.xlabel("Number of Words")
plt.ylabel("Count")
plt.savefig("transcript_length_hist.png")
plt.close()

# --- PDF Report ---
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "Data Preprocessing Report", ln=True)

pdf.set_font("Arial", size=12)
pdf.ln(10)
pdf.cell(0, 10, f"Total Samples: {len(durations)}", ln=True)
pdf.cell(0, 10, f"Average Duration: {np.mean(durations):.2f} sec", ln=True)
pdf.cell(0, 10, f"Min Duration: {np.min(durations):.2f} sec", ln=True)
pdf.cell(0, 10, f"Max Duration: {np.max(durations):.2f} sec", ln=True)
pdf.cell(0, 10, f"Mel Bands: {n_mels[0] if n_mels else 'N/A'}", ln=True)
pdf.cell(0, 10, f"Average Transcript Length: {np.mean(transcript_lengths):.2f} words", ln=True)

pdf.ln(10)
pdf.cell(0, 10, "Duration Histogram", ln=True)
pdf.image("duration_hist.png", w=180)

pdf.ln(10)
pdf.cell(0, 10, "Transcript Length Histogram", ln=True)
pdf.image("transcript_length_hist.png", w=180)

# Save PDF
report_name = "preprocessing_report.pdf"
pdf.output(report_name)
print(f"✅ Report generated: {report_name}")
