import modal
import os
import uuid
import subprocess
import wave
import numpy as np
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager 

# 1. DEFINE THE CLOUD ENVIRONMENT
image = (
    modal.Image.debian_slim(python_version="3.11") 
    .apt_install("ffmpeg", "cargo", "rustc")
    .pip_install(
        "fastapi", 
        "uvicorn", 
        "python-multipart", 
        "openai-whisper", 
        "pyannote.audio", 
        "torch==2.3.1",       
        "torchaudio==2.3.1",  
        "numpy",
        "deepfilternet",
        "huggingface_hub==0.25.2"  # THE FINAL FIX: Prevent the hub from crashing Pyannote
    )
    .add_local_file("index.html", remote_path="/root/index.html") 
)

app = modal.App("asr-weekend-mvp")
JOB_DATABASE = modal.Dict.from_name("asr-job-db", create_if_missing=True)

# 2. ISOLATE HEAVY ML IMPORTS
# Only these libraries stay in the "danger zone" so they don't break your local laptop
with image.imports():
    import torch
    import whisper
    from pyannote.audio import Pipeline

model = None
diarization_pipeline = None

def load_ai_models():
    global model, diarization_pipeline
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper large-v3 on {device}...")
        model = whisper.load_model("large-v3", device=device)
        
        print("Loading Pyannote...")
        hf_token = os.environ.get("HF_TOKEN") 
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        ).to(torch.device(device))
        print("All models loaded successfully!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_ai_models() 
    yield

# 3. WEB APP INITIALIZATION (Safely outside the imports block!)
web_app = FastAPI(lifespan=lifespan)
SUPPORTED_EXTENSIONS = [".wav", ".mp3", ".ogg", ".flac", ".mp4"]

def get_dominant_speaker(start_time, end_time, diarization_result):
    overlap_durations = {}
    for turn, speaker in diarization_result.speaker_diarization:
        overlap = max(0, min(end_time, turn.end) - max(start_time, turn.start))
        if overlap > 0:
            overlap_durations[speaker] = overlap_durations.get(speaker, 0) + overlap
    if overlap_durations:
        return max(overlap_durations, key=overlap_durations.get)
    return "UNKNOWN"


# --- THE ISOLATED BACKGROUND WORKER ---
@app.function(
    image=image, 
    gpu="A10G",
    secrets=[modal.Secret.from_name("HF_TOKEN")],
    timeout=1800 
)
def process_audio_in_background(job_id: str, file_bytes: bytes, filename: str):
    JOB_DATABASE[job_id] = {"status": "processing", "transcript": ""}
    
    file_path = f"/tmp/{job_id}_{filename}"
    wav_path = f"/tmp/{job_id}_raw.wav" 
    clean_wav_path = f"/tmp/{job_id}_raw_DeepFilterNet3.wav" 
    
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    
    try:
        load_ai_models() 
        
        # 1. Format to 16kHz WAV
        subprocess.run([
            "ffmpeg", "-y", "-i", file_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 2. NOISE SUPPRESSION: Clean the audio using DeepFilterNet before sending to Whisper
        print("Running DeepFilterNet Noise Suppression...")
        subprocess.run(["deepFilter", wav_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 3. ACCURACY FIX: Force Vietnamese language and pass an initial prompt
        print("Transcribing with Whisper...")
        whisper_result = model.transcribe(
            clean_wav_path, 
            word_timestamps=True,
            language="vi",
            task="transcribe",
            initial_prompt="Đây là một cuộc hội thoại thực tế bằng tiếng Việt, có chứa các từ tiếng Anh và tên riêng như ProtonTech."
        )
        
        # 4. Diarization (Using the cleaned audio)
        with wave.open(clean_wav_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            audio_bytes = wf.readframes(wf.getnframes())
            
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).copy()
        audio_tensor = torch.from_numpy(audio_array).float() / 32768.0
        audio_tensor = audio_tensor.unsqueeze(0) 
        
        diarization_input = {"waveform": audio_tensor, "sample_rate": sample_rate}
        diarization_result = diarization_pipeline(diarization_input)
        
        # 5. Merge Results
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
            
        JOB_DATABASE[job_id] = {
            "status": "completed", 
            "transcript": whisper_result["text"],
            "segments": segments_data
        }
        
    except Exception as e:
        JOB_DATABASE[job_id] = {"status": "failed", "transcript": f"Error: {str(e)}"}
    finally:
        for p in [file_path, wav_path, clean_wav_path]:
            if os.path.exists(p): os.remove(p)


# --- WEB ROUTES ---
@web_app.get("/")
async def get():
    with open("/root/index.html", "r", encoding="utf-8") as file:
        return HTMLResponse(file.read())

@web_app.post("/api/v1/asr/batch")
async def handle_batch_upload(file: UploadFile = File(...)):
    filename_lower = file.filename.lower()
    if not any(filename_lower.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        raise HTTPException(status_code=422, detail="ASR-005: Invalid format.")
    
    file_bytes = await file.read()
    job_id = f"job_asr_{uuid.uuid4().hex[:8]}"
    
    await JOB_DATABASE.put.aio(job_id, {"status": "queued", "filename": file.filename})
    process_audio_in_background.spawn(job_id, file_bytes, file.filename)
    
    return {"job_id": job_id, "status": "queued"}

@web_app.get("/api/v1/asr/jobs/{job_id}")
async def check_job_status(job_id: str):
    has_job = await JOB_DATABASE.contains.aio(job_id)
    if not has_job:
        raise HTTPException(status_code=404, detail="Job not found")
    return await JOB_DATABASE.get.aio(job_id)

@web_app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            temp_stream_path = f"/tmp/stream_{uuid.uuid4().hex[:8]}.webm"
            with open(temp_stream_path, "wb") as f:
                f.write(data)
            try:
                result = await asyncio.to_thread(
                    model.transcribe, 
                    temp_stream_path,
                    language="vi",
                    task="transcribe",
                    initial_prompt="Đây là cuộc hội thoại tiếng Việt."
                )
                text = result["text"].strip()
                if text:
                    await websocket.send_json({"type": "segment", "text": text, "language": result.get("language", "unknown"), "status": "success"})
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})
            if os.path.exists(temp_stream_path):
                os.remove(temp_stream_path)
    except WebSocketDisconnect:
        pass

@app.function(
    image=image, 
    gpu="A10G",
    secrets=[modal.Secret.from_name("HF_TOKEN")],
    timeout=600
)
@modal.asgi_app()
def fastapi_app():
    return web_app