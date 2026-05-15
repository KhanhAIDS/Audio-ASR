import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
import asyncio
import uuid
import whisper
import os
from pyannote.audio import Pipeline
import subprocess
import torchaudio

app = FastAPI()
SUPPORTED_EXTENSIONS = [".wav", ".mp3", ".ogg", ".flac", ".mp4"]
JOB_DATABASE = {}

@app.get("/")
async def get():
    with open("index.html", "r", encoding="utf-8") as file:
        return HTMLResponse(file.read())

# --- VRAM SAFETY LIMITER ---
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    # 20GB is approx 15% (0.15) of a 128GB GPU. 
    # This prevents your app from crashing your teammates' programs.
    torch.cuda.set_per_process_memory_fraction(0.15, device=0)
    print("VRAM explicitly limited to 15% (~20GB).")
    
print(f"Loading Whisper model on {device}...")
# UPGRADED TO SRS SPECIFICATION: large-v3
model = whisper.load_model("large-v3", device=device) 
print("Whisper loaded!")

# --- NEW: LOAD REAL DIARIZATION MODEL ---
print("Loading Pyannote Diarization...")
# REPLACE 'YOUR_HF_TOKEN' with your actual Hugging Face token
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token="HF_KEY"
)
diarization_pipeline.to(torch.device(device))
print("Diarization loaded!")
# ----------------------------------------


# --- HELPER FUNCTION: Match Whisper timestamps to Pyannote speakers ---
def get_dominant_speaker(start_time, end_time, diarization_result):
    overlap_durations = {}
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        # Calculate how much the speaker's time overlaps with the word's time
        overlap = max(0, min(end_time, turn.end) - max(start_time, turn.start))
        if overlap > 0:
            overlap_durations[speaker] = overlap_durations.get(speaker, 0) + overlap
    
    if overlap_durations:
        # Return the speaker who talked the most during this specific timeframe
        return max(overlap_durations, key=overlap_durations.get)
    return "UNKNOWN"


# --- UPDATED BACKGROUND WORKER ---
def process_audio_in_background(job_id: str, file_path: str):
    JOB_DATABASE[job_id]["status"] = "processing"
    wav_path = f"{file_path}.wav" # Đường dẫn file chuẩn sẽ được tạo ra
    
    try:
        # 1. ÉP KIỂU FILE: Dùng ffmpeg tĩnh để ép mọi file thành WAV 16kHz chuẩn AI
        subprocess.run([
            "ffmpeg", "-y", "-i", file_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 2. Chạy Whisper trên file WAV chuẩn
        whisper_result = model.transcribe(wav_path, word_timestamps=True)
        
        # 3. LÁCH LUẬT ĐỌC ÂM THANH: Đọc file WAV trực tiếp vào RAM bằng torchaudio + soundfile
        waveform, sample_rate = torchaudio.load(wav_path, backend="soundfile")
        diarization_input = {"waveform": waveform, "sample_rate": sample_rate}
        
        # 4. Truyền thẳng RAM vào Pyannote (Không dùng đường dẫn file nữa)
        diarization_result = diarization_pipeline(diarization_input)
        
        # 5. Nối dữ liệu (Giữ nguyên logic cũ)
        segments_data = []
        for segment in whisper_result["segments"]:
            segment_speaker = get_dominant_speaker(segment["start"], segment["end"], diarization_result)
            
            words = []
            for word in segment["words"]:
                words.append({
                    "word": word["word"].strip(),
                    "start_ms": int(word["start"] * 1000),
                    "end_ms": int(word["end"] * 1000),
                    "speaker": get_dominant_speaker(word["start"], word["end"], diarization_result)
                })
            
            segments_data.append({
                "text": segment["text"].strip(),
                "start_ms": int(segment["start"] * 1000),
                "end_ms": int(segment["end"] * 1000),
                "speaker": segment_speaker,
                "words": words
            })
            
        JOB_DATABASE[job_id]["status"] = "completed"
        JOB_DATABASE[job_id]["transcript"] = whisper_result["text"]
        JOB_DATABASE[job_id]["segments"] = segments_data 
        
    except Exception as e:
        JOB_DATABASE[job_id]["status"] = "failed"
        JOB_DATABASE[job_id]["transcript"] = f"Error: {str(e)}"
    finally:
        # Dọn dẹp cả file gốc và file WAV tạm để bảo vệ System RAM
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)

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
    try:
        while True:
            # 1. Receive the fully formed 3-second audio file
            data = await websocket.receive_bytes()
            
            temp_stream_path = f"temp_stream_{uuid.uuid4().hex[:8]}.webm"
            with open(temp_stream_path, "wb") as f:
                f.write(data)
            
            # 2. Transcribe
            try:
                result = await asyncio.to_thread(model.transcribe, temp_stream_path)
                text = result["text"].strip()
                
                # 3. NEXT LEAP: SRS 3.3.2 Structured JSON Output
                if text:
                    response = {
                        "type": "segment",
                        "text": text,
                        "language": result["language"],
                        "status": "success"
                    }
                    # Send JSON instead of plain text
                    await websocket.send_json(response)
                    
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})
            
            # Clean up
            if os.path.exists(temp_stream_path):
                os.remove(temp_stream_path)
                
    except WebSocketDisconnect:
        pass