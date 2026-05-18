from fastapi.middleware.cors import CORSMiddleware
import json
from kafka import KafkaProducer
import threading # <--- NEW: Import threading for the GPU Lock

# Initialize Kafka Producer
try:
    producer = KafkaProducer(
        bootstrap_servers=['192.168.40.96:9092'], 
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    print("Kafka Producer connected successfully.")
except Exception as e:
    print(f"Warning: Kafka not connected. {e}")
    producer = None

import os
import uuid
import subprocess
import wave
import numpy as np
import asyncio
import torch
import torchaudio
import whisper
from pyannote.audio import Pipeline
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager

# --- STANDARD IN-MEMORY DATABASE ---
JOB_DATABASE = {}
SUPPORTED_EXTENSIONS = [".wav", ".mp3", ".ogg", ".flac", ".mp4"]

model = None
diarization_pipeline = None

# --- NEW: THE GPU TRAFFIC LIGHT ---
gpu_lock = threading.Lock() 

def load_ai_models():
    global model, diarization_pipeline
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.15, device=0)
            print("VRAM explicitly limited to 15%.")
            
        print(f"Loading Whisper large-v3 on {device}...")
        model = whisper.load_model("large-v3", device=device)
        
        print("Loading Pyannote...")
        hf_token = os.environ.get("HF_TOKEN") 
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        ).to(torch.device(device))
        print("All models loaded successfully!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_ai_models() 
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_dominant_speaker(start_time, end_time, diarization_result):
    overlap_durations = {}
    for turn, speaker in diarization_result.speaker_diarization:
        overlap = max(0, min(end_time, turn.end) - max(start_time, turn.start))
        if overlap > 0:
            overlap_durations[speaker] = overlap_durations.get(speaker, 0) + overlap
    if overlap_durations:
        return max(overlap_durations, key=overlap_durations.get)
    return "UNKNOWN"

def process_audio_in_background(job_id: str, file_path: str):
    JOB_DATABASE[job_id]["status"] = "processing"
    
    wav_path = f"{file_path}.wav" 
    clean_wav_path = f"{file_path}_DeepFilterNet3.wav" 
    
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", file_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("Running DeepFilterNet Noise Suppression...")
        try:
            output_dir = os.path.dirname(os.path.abspath(wav_path))
            subprocess.run(
                ["deepFilter", wav_path, "-o", output_dir], 
                capture_output=True, text=True, check=True
            )
            final_audio_path = clean_wav_path 
        except subprocess.CalledProcessError as e:
            print(f"DeepFilterNet crashed: {e.stderr}")
            final_audio_path = wav_path
            
        print("Transcribing with Whisper...")
        # --- FIX: LOCK THE GPU ---
        with gpu_lock:
            whisper_result = model.transcribe(
                final_audio_path, 
                word_timestamps=True,
                task="transcribe",
                initial_prompt="Dạ vâng, hello mọi người, trong buổi meeting hôm nay chúng ta sẽ discuss các vấn đề, update tiến độ, và confirm lại với team nhé."
            )
        
        with wave.open(final_audio_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            audio_bytes = wf.readframes(wf.getnframes())
            
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).copy()
        audio_tensor = torch.from_numpy(audio_array).float() / 32768.0
        audio_tensor = audio_tensor.unsqueeze(0) 
        
        diarization_input = {"waveform": audio_tensor, "sample_rate": sample_rate}
        
        print("Diarizing...")
        # --- FIX: LOCK THE GPU ---
        with gpu_lock:
            diarization_result = diarization_pipeline(diarization_input)
        
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
        
        if producer:
            print(f"Publishing Job {job_id} to Kafka...")
            event_payload = {
                "event_type": "ASR_JOB_COMPLETED",
                "job_id": job_id,
                "full_transcript": whisper_result["text"],
                "segments": segments_data 
            }
            producer.send('asr_completed_events', value=event_payload)
            producer.flush()
            
    except Exception as e:
        JOB_DATABASE[job_id]["status"] = "failed"
        JOB_DATABASE[job_id]["transcript"] = f"Error: {str(e)}"
        if producer:
            producer.send('asr_failed_events', value={"job_id": job_id, "error": str(e)})
            producer.flush()
    finally:
        for p in [file_path, wav_path, clean_wav_path]:
            if os.path.exists(p): os.remove(p)


@app.get("/")
async def get():
    with open("index.html", "r", encoding="utf-8") as file:
        return HTMLResponse(file.read())

@app.post("/api/v1/asr/batch")
async def handle_batch_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    filename_lower = file.filename.lower()
    if not any(filename_lower.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        raise HTTPException(status_code=422, detail="ASR-005: Invalid format.")
    
    job_id = f"job_asr_{uuid.uuid4().hex[:8]}"
    temp_file_path = f"temp_{job_id}_{file.filename}"
    
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    JOB_DATABASE[job_id] = {"status": "queued", "filename": file.filename}
    background_tasks.add_task(process_audio_in_background, job_id, temp_file_path)
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/api/v1/asr/jobs/{job_id}")
async def check_job_status(job_id: str):
    if job_id not in JOB_DATABASE:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOB_DATABASE[job_id]

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    chunk_index = 0 # <--- NEW: Track which 3-second chunk we are on
    try:
        while True:
            data = await websocket.receive_bytes()
            
            # --- NEW: Calculate Timestamp for this exact chunk ---
            start_time_s = chunk_index * 3
            end_time_s = (chunk_index + 1) * 3
            chunk_index += 1
            
            temp_stream_path = f"temp_stream_{uuid.uuid4().hex[:8]}.webm"
            with open(temp_stream_path, "wb") as f:
                f.write(data)
            try:
                # --- FIX: LOCK THE GPU ---
                def stream_transcribe_locked(path):
                    with gpu_lock:
                        return model.transcribe(
                            path,
                            task="transcribe",
                            initial_prompt="Dạ vâng, hello mọi người, trong buổi meeting hôm nay chúng ta sẽ discuss các vấn đề, update tiến độ, và confirm lại với team nhé."
                        )
                
                result = await asyncio.to_thread(stream_transcribe_locked, temp_stream_path)
                text = result["text"].strip()
                if text:
                    await websocket.send_json({
                        "type": "segment", 
                        "text": text, 
                        "language": result.get("language", "unknown"),
                        "start_time": start_time_s, # Send time to browser
                        "end_time": end_time_s,     # Send time to browser
                        "status": "success"
                    })
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})
            if os.path.exists(temp_stream_path):
                os.remove(temp_stream_path)
    except WebSocketDisconnect:
        pass