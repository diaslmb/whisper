import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import pipeline
import logging
import uvicorn

# --- App and Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Whisper ASR Service",
    description="A FastAPI service to serve the OpenAI Whisper Large v3 model.",
    version="1.0.0",
)

# --- Model Loading ---
try:
    logger.info("Loading Whisper model 'openai/whisper-large-v3'...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    logger.info("Whisper model loaded successfully. 🚀")
except Exception as e:
    logger.error(f"Fatal: Failed to load the Whisper model. Error: {e}")
    pipe = None

# --- API Endpoints ---
@app.get("/")
def health_check():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Whisper ASR service is running."}

# MODIFIED LINE: The endpoint route is updated here
@app.post("/audio/transcriptions")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribes an uploaded audio file.

    Accepts an audio file (e.g., .wav, .mp3) and returns the transcription.
    """
    if pipe is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Please check server logs."
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file uploaded.")

    try:
        logger.info(f"Transcribing file: {file.filename}")
        result = pipe(audio_bytes)
        transcription = result["text"].strip()
        logger.info("Transcription successful.")
        return {"filename": file.filename, "transcription": transcription}
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during transcription: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
