# Real-Time-Speech-Recognition-Translation

This project focuses on building a Real-Time Speech Recognition and Translation system using a fine-tuned version of the Whisper model from Hugging Face.
It enables users to upload audio files (WAV, MP3, etc.) and receive accurate text transcriptions in real time.

## Project Structure

- Model: Fine-tuned Whisper model (whisper-small) for speech recognition.

- Backend: FastAPI server to handle file uploads and model inference.

- Deployment: Easy-to-run local API server using Uvicorn.

## How to Run the Project

*Clone the repository:*

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

*Install the required packages:*

```
pip install -r requirements.txt
```

*Start the FastAPI server:*

```
python -m uvicorn main:app
```
*Access the API: Open your browser and navigate to:*
```
http://127.0.0.1:8000
```
*You can also visit:*
```
http://127.0.0.1:8000/docs
```
to use the interactive Swagger UI for testing the API.

## Dataset and Checkpoints

You can find the training data and model checkpoints on Google Drive through the following link:

📂 [Access the Data and Checkpoints](https://drive.google.com/drive/folders/19n5h-e92ZoP0IslAuYKqo-0Kd-A4zO2X?usp=drive_link)

**Data Folder: Contains the preprocessed audio (.wav) files and transcripts (.csv files).**

**Checkpoints Folder: Contains the saved fine-tuned Whisper model.**

## API Usage
**Endpoint:** POST /predict/

**Request:**

- Upload an audio file (.wav, .mp3, etc.).

**Response:**

- JSON response containing the transcribed text.

Example:
```
{
  "transcription": "Hello, welcome to the real-time speech recognition demo.",
  "trasnlation": "Bonjour, bienvenue à la démonstration de reconnaissance vocale en temps réel."
}
```

## Project Highlights

- Real-time audio file transcription.
- Fine-tuned multilingual speech recognition.
- Simple deployment using FastAPI.
- Extendable to real-time translation in the future.

## Future Improvements

- Add real-time audio streaming support.
- Expand to multilingual translation after transcription.
- Enhance model robustness for noisy audio.


