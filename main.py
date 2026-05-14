import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
import asyncio
import uuid
import whisper
import os


app = FastAPI()
SUPPORTED_EXTENSIONS = [".wav", ".mp3", ".ogg", ".flac", ".mp4"]
JOB_DATABASE = {}

@app.get("/")
async def get():
    with open("index.html", "r", encoding="utf-8") as file:
        return HTMLResponse(file.read())


# Check if your GPU is visible to Python, otherwise fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Whisper model on {device}...")

# Add the device parameter
model = whisper.load_model("tiny", device=device)
print("Model loaded!")


def process_audio_in_background(job_id: str, file_path: str):
    JOB_DATABASE[job_id]["status"] = "processing"
    
    try:
        # Run real Whisper transcription
        result = model.transcribe(file_path)
        
        JOB_DATABASE[job_id]["status"] = "completed"
        JOB_DATABASE[job_id]["transcript"] = result["text"]
    except Exception as e:
        JOB_DATABASE[job_id]["status"] = "failed"
        JOB_DATABASE[job_id]["transcript"] = f"Error: {str(e)}"
    finally:
        # Clean up the saved file to save space
        if os.path.exists(file_path):
            os.remove(file_path)

# --- BATCH DOOR (Updated to save file to disk) ---
@app.post("/api/v1/asr/batch")
async def handle_batch_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    filename_lower = file.filename.lower()
    if not any(filename_lower.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        raise HTTPException(status_code=422, detail="ASR-005: Invalid format.")

    job_id = f"job_asr_{uuid.uuid4().hex[:8]}"
    
    # Save uploaded file temporarily to disk for Whisper
    temp_file_path = f"temp_{job_id}_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    JOB_DATABASE[job_id] = {"status": "queued", "filename": file.filename}

    # Pass the file path to the background worker
    background_tasks.add_task(process_audio_in_background, job_id, temp_file_path)
    
    return {"job_id": job_id, "status": "queued", "message": "File queued for AI."}



# --- 4. THE STATUS CHECKER DOOR (New SRS Requirement) ---
# The front-end will call this every 1 second to check on the job.
@app.get("/api/v1/asr/jobs/{job_id}")
async def check_job_status(job_id: str):
    if job_id not in JOB_DATABASE:
        raise HTTPException(status_code=404, detail="ASR-201: Job not found")
    
    return JOB_DATABASE[job_id]


# --- THE STREAMING DOOR (Unchanged) ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = bytearray() 
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)
            if len(audio_buffer) >= 5000: 
                await asyncio.sleep(0.5) # Fake model
                await websocket.send_text(f"Processed {len(audio_buffer)} bytes.")
                audio_buffer.clear()
    except WebSocketDisconnect:
        pass