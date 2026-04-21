import os
import uuid
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import yt_dlp
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="VidTranscribe")

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set")
groq_client = Groq(api_key=groq_api_key)

# Create temp directory on startup
TEMP_DIR = Path("/tmp/vidtranscribe")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Setup Instagram cookies from environment variable
COOKIES_FILE = TEMP_DIR / "cookies.txt"

def setup_cookies():
    """Write cookies from env var to a Netscape cookies file for yt-dlp."""
    cookies_data = os.getenv("INSTAGRAM_COOKIES")
    if cookies_data:
        COOKIES_FILE.write_text(cookies_data)
        return True
    return False

HAS_COOKIES = setup_cookies()

# Request models
class TranscribeRequest(BaseModel):
    url: str


class TranslateRequest(BaseModel):
    text: str


class SummarizeRequest(BaseModel):
    text: str


# Response models
class TranscribeResponse(BaseModel):
    transcript: str
    duration_seconds: float
    language: str


class TranslateResponse(BaseModel):
    original_text: str
    translated_text: str
    language: str


@app.on_event("startup")
async def startup_event():
    """Ensure temp directory exists on startup."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/transcribe")
async def transcribe(request: TranscribeRequest):
    """
    Main transcription endpoint.
    Accepts a TikTok or Instagram Reels URL and returns the transcript.
    """
    try:
        url = request.url.strip()
        
        # Validate URL
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        if not ("tiktok.com" in url or "instagram.com" in url):
            return {"error": "Only TikTok and Instagram Reels URLs are supported"}
        
        # Create unique filename for this request
        request_id = str(uuid.uuid4())
        video_path = TEMP_DIR / f"{request_id}.mp4"
        audio_path = TEMP_DIR / f"{request_id}.mp3"
        
        try:
            # Step 1: Download video using yt-dlp
            ydl_opts = {
                "format": "best[ext=mp4]",
                "outtmpl": str(video_path.with_suffix("")),
                "quiet": True,
                "no_warnings": True,
                "no_playlist": True,
            }
            
            # Add cookies for Instagram if available
            if "instagram.com" in url and HAS_COOKIES:
                ydl_opts["cookiefile"] = str(COOKIES_FILE)
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Check if video was downloaded
            if not video_path.exists():
                # Try to find the downloaded file (yt-dlp might save with different extension)
                possible_files = list(TEMP_DIR.glob(f"{request_id}*"))
                if possible_files:
                    video_path = possible_files[0]
                else:
                    raise FileNotFoundError("Video download failed")
            
            # Step 2: Extract audio using ffmpeg
            ffmpeg_cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-ar", "16000",
                "-ac", "1",
                "-b:a", "32k",
                "-y",
                str(audio_path),
            ]
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            # Step 3: Check audio file size
            audio_size_mb = audio_path.stat().st_size / (1024 * 1024)
            if audio_size_mb > 40:
                return {"error": "Video too long, please use a shorter clip"}
            
            # Step 4: Call Groq Whisper API
            with open(audio_path, "rb") as audio_file:
                transcript_response = groq_client.audio.transcriptions.create(
                    file=(audio_path.name, audio_file, "audio/mpeg"),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                )
            
            # Extract transcript and metadata
            transcript_text = transcript_response.text
            duration = transcript_response.duration
            language = getattr(transcript_response, "language", "Unknown")
            
            return {
                "transcript": transcript_text,
                "duration_seconds": duration,
                "language": language,
            }
        
        finally:
            # Clean up temp files
            for temp_file in [video_path, audio_path]:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except Exception as e:
                    print(f"Failed to delete {temp_file}: {e}")
            
            # Also clean up any related files (in case yt-dlp saved with different extension)
            try:
                for f in TEMP_DIR.glob(f"{request_id}*"):
                    f.unlink()
            except Exception as e:
                print(f"Failed to clean up related files: {e}")
    
    except Exception as e:
        # Global exception handler
        error_msg = str(e)
        if "Video too long" in error_msg:
            return {"error": error_msg}
        elif "Only TikTok and Instagram" in error_msg:
            return {"error": error_msg}
        else:
            # Generic error message
            return {"error": "Failed to transcribe video. Please try again."}


@app.post("/translate")
async def translate(request: TranslateRequest):
    """
    Translation endpoint.
    Translates the provided text to Indonesian using Groq chat completion.
    """
    try:
        text = request.text.strip()
        
        if not text:
            return {"error": "No text provided for translation"}
        
        # Use Groq chat completion for translation
        message = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional translator. Translate the following text to Indonesian (Bahasa Indonesia). Provide only the translation, nothing else.",
                },
                {
                    "role": "user",
                    "content": f"Translate this to Indonesian:\n\n{text}",
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=2048,
        )
        
        translated_text = message.choices[0].message.content.strip()
        
        return {
            "original_text": text,
            "translated_text": translated_text,
            "language": "Indonesian",
        }
    
    except Exception as e:
        # Global exception handler
        error_msg = str(e)
        return {"error": "Failed to translate text. Please try again."}


@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    """
    Summarize endpoint.
    Summarizes the provided transcript text using Groq chat completion.
    """
    try:
        text = request.text.strip()
        
        if not text:
            return {"error": "No text provided for summarization"}
        
        message = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional content summarizer. Summarize the following transcript into clear, concise bullet points. Keep the key information and main ideas. Write the summary in the same language as the input text. Format each point on a new line starting with a bullet point (•).",
                },
                {
                    "role": "user",
                    "content": f"Summarize this transcript:\n\n{text}",
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=2048,
        )
        
        summary_text = message.choices[0].message.content.strip()
        
        return {
            "original_text": text,
            "summary": summary_text,
        }
    
    except Exception as e:
        return {"error": "Failed to summarize text. Please try again."}


# Serve the frontend
@app.get("/")
async def serve_frontend():
    """Serve the index.html frontend."""
    return FileResponse("index.html", media_type="text/html")


@app.get("/index.html")
async def serve_index():
    """Serve the index.html frontend."""
    return FileResponse("index.html", media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
