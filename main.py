from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
import torch
import os
import numpy as np
from pydub import AudioSegment
import traceback
import shutil

# Initialize FastAPI
app = FastAPI()

# Define the allowed upload directory
UPLOAD_DIR = "upload"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load the trained model and processor
model = WhisperForConditionalGeneration.from_pretrained("whisper-small-hi")
processor = WhisperProcessor.from_pretrained("whisper-small-hi")

# Modify the generation config directly
model.generation_config.forced_decoder_ids = None

# Load a translation pipeline (English ➔ French)
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

# Convert audio file to the format suitable for prediction (16kHz)
def convert_audio(file_path: str):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000)
    audio_array = np.array(audio.get_array_of_samples())
    return audio_array

# New prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Check if file saved properly
        if not os.path.isfile(file_location):
            raise HTTPException(status_code=400, detail="File upload failed.")

        audio_array = convert_audio(file_location)

        input_features = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        )

        # Generate English transcription
        predicted_ids = model.generate(input_features.input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        translation = ""
        if transcription.strip():
            try:
                translation_output = translator(transcription)
                if translation_output and isinstance(translation_output, list):
                    translation = translation_output[0]['translation_text']
            except Exception as translation_error:
                print(f"Translation failed: {str(translation_error)}")
                translation = ""

        return {
            "transcription": transcription,
            "translation": translation
        }

    except Exception as e:
        error_message = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)