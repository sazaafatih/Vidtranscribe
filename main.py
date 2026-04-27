import os
import uuid
import json
import subprocess
import tempfile
import math
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

# Max audio chunk size for Groq Whisper (in MB)
MAX_CHUNK_MB = 24

# Supported platforms
SUPPORTED_PLATFORMS = ["tiktok.com", "instagram.com", "youtube.com", "youtu.be"]


def is_youtube_url(url):
    return "youtube.com" in url or "youtu.be" in url


def get_youtube_subtitles(url):
    """Try to extract subtitles/captions from YouTube video."""
    try:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "id", "en-orig"],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            # Get video duration
            duration = info.get("duration", 0)
            
            # Try to get subtitles
            subtitles = info.get("subtitles", {})
            auto_captions = info.get("automatic_captions", {})
            
            # Prefer manual subtitles, then auto captions
            all_subs = {**auto_captions, **subtitles}
            
            # Try languages in order: en, id, then any available
            for lang in ["en", "id", "en-orig"]:
                if lang in all_subs:
                    for fmt in all_subs[lang]:
                        if fmt.get("ext") == "json3":
                            # Download and parse json3 subtitle
                            sub_url = fmt.get("url")
                            if sub_url:
                                import urllib.request
                                with urllib.request.urlopen(sub_url) as resp:
                                    sub_data = json.loads(resp.read().decode())
                                    events = sub_data.get("events", [])
                                    texts = []
                                    for event in events:
                                        segs = event.get("segs", [])
                                        for seg in segs:
                                            t = seg.get("utf8", "").strip()
                                            if t and t != "\n":
                                                texts.append(t)
                                    if texts:
                                        transcript = " ".join(texts)
                                        return transcript, duration, lang
            
            # Try vtt format as fallback
            for lang in ["en", "id", "en-orig"]:
                if lang in all_subs:
                    for fmt in all_subs[lang]:
                        if fmt.get("ext") == "vtt":
                            sub_url = fmt.get("url")
                            if sub_url:
                                import urllib.request
                                with urllib.request.urlopen(sub_url) as resp:
                                    vtt_text = resp.read().decode()
                                    # Simple VTT parser
                                    lines = vtt_text.split("\n")
                                    texts = []
                                    for line in lines:
                                        line = line.strip()
                                        if not line:
                                            continue
                                        if "-->" in line:
                                            continue
                                        if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
                                            continue
                                        if line.isdigit():
                                            continue
                                        # Remove HTML tags
                                        import re
                                        clean = re.sub(r"<[^>]+>", "", line)
                                        if clean.strip():
                                            texts.append(clean.strip())
                                    if texts:
                                        # Remove consecutive duplicates
                                        deduped = [texts[0]]
                                        for t in texts[1:]:
                                            if t != deduped[-1]:
                                                deduped.append(t)
                                        transcript = " ".join(deduped)
                                        return transcript, duration, lang
            
            return None, duration, None
    except Exception as e:
        print(f"Subtitle extraction failed: {e}")
        return None, 0, None


def split_audio(audio_path, max_size_mb=MAX_CHUNK_MB):
    """Split audio file into chunks smaller than max_size_mb."""
    audio_size_mb = audio_path.stat().st_size / (1024 * 1024)
    
    if audio_size_mb <= max_size_mb:
        return [audio_path]
    
    # Get audio duration using ffprobe
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
    total_duration = float(result.stdout.strip())
    
    # Calculate number of chunks needed
    num_chunks = math.ceil(audio_size_mb / max_size_mb)
    chunk_duration = total_duration / num_chunks
    
    chunks = []
    for i in range(num_chunks):
        start_time = i * chunk_duration
        chunk_path = audio_path.parent / f"{audio_path.stem}_chunk{i}.mp3"
        
        split_cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-ss", str(start_time),
            "-t", str(chunk_duration),
            "-q:a", "9",
            str(chunk_path)
        ]
        
        subprocess.run(split_cmd, capture_output=True, text=True, timeout=300)
        
        if chunk_path.exists():
            chunks.append(chunk_path)
    
    return chunks


def transcribe_audio_chunks(chunks):
    """Transcribe multiple audio chunks and combine results."""
    all_text = []
    total_duration = 0
    language = "Unknown"
    
    for chunk_path in chunks:
        try:
            with open(chunk_path, "rb") as audio_file:
                response = groq_client.audio.transcriptions.create(
                    file=(chunk_path.name, audio_file, "audio/mpeg"),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                )
            all_text.append(response.text)
            total_duration += response.duration
            language = getattr(response, "language", language)
        except Exception as e:
            print(f"Failed to transcribe chunk {chunk_path}: {e}")
            all_text.append("[transcription failed for this segment]")
    
    return " ".join(all_text), total_duration, language


# Request models
class TranscribeRequest(BaseModel):
    url: str


class TranslateRequest(BaseModel):
    text: str


class SummarizeRequest(BaseModel):
    text: str


@app.on_event("startup")
async def startup_event():
    """Ensure temp directory exists on startup."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/transcribe")
async def transcribe(request: TranscribeRequest):
    """
    Main transcription endpoint.
    Accepts a TikTok, Instagram Reels, or YouTube URL and returns the transcript.
    """
    try:
        url = request.url.strip()
        
        # Validate URL
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        if not any(platform in url for platform in SUPPORTED_PLATFORMS):
            return {"error": "Supported platforms: TikTok, Instagram Reels, YouTube"}
        
        # For YouTube: try subtitles first (fast, free, no Groq usage)
        if is_youtube_url(url):
            subtitle_text, yt_duration, sub_lang = get_youtube_subtitles(url)
            if subtitle_text:
                return {
                    "transcript": subtitle_text,
                    "duration_seconds": yt_duration,
                    "language": sub_lang or "Unknown",
                    "source": "subtitles",
                }
        
        # Fallback: download and transcribe with Whisper
        request_id = str(uuid.uuid4())
        video_path = TEMP_DIR / f"{request_id}.mp4"
        audio_path = TEMP_DIR / f"{request_id}.mp3"
        chunk_paths = []
        
        try:
            # Step 1: Download video/audio using yt-dlp
            if is_youtube_url(url):
                # For YouTube, download audio only (much smaller)
                ydl_opts = {
                    "format": "bestaudio/best",
                    "outtmpl": str(audio_path.with_suffix("")),
                    "postprocessors": [{
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "64",
                    }],
                    "quiet": True,
                    "no_warnings": True,
                    "no_playlist": True,
                }
            else:
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
            
            # For non-YouTube: extract audio from video
            if not is_youtube_url(url):
                if not video_path.exists():
                    possible_files = list(TEMP_DIR.glob(f"{request_id}*"))
                    if possible_files:
                        video_path = possible_files[0]
                    else:
                        raise FileNotFoundError("Video download failed")
                
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-q:a", "9",
                    str(audio_path),
                ]
                
                result = subprocess.run(
                    ffmpeg_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            # Check if audio file exists
            if not audio_path.exists():
                possible_audio = list(TEMP_DIR.glob(f"{request_id}*.mp3"))
                if possible_audio:
                    audio_path = possible_audio[0]
                else:
                    raise FileNotFoundError("Audio extraction failed")
            
            # Split audio if too large for Groq
            chunks = split_audio(audio_path)
            chunk_paths = [c for c in chunks if c != audio_path]
            
            # Transcribe
            transcript_text, duration, language = transcribe_audio_chunks(chunks)
            
            return {
                "transcript": transcript_text,
                "duration_seconds": duration,
                "language": language,
                "source": "whisper",
            }
        
        finally:
            # Clean up all temp files
            for temp_file in [video_path, audio_path] + chunk_paths:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except Exception as e:
                    print(f"Failed to delete {temp_file}: {e}")
            
            try:
                for f in TEMP_DIR.glob(f"{request_id}*"):
                    f.unlink()
            except Exception as e:
                print(f"Failed to clean up related files: {e}")
    
    except Exception as e:
        error_msg = str(e)
        if "Video too long" in error_msg:
            return {"error": error_msg}
        elif "Supported platforms" in error_msg:
            return {"error": error_msg}
        else:
            print(f"Transcribe error: {e}")
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
                    "content": "You are a professional content summarizer. Summarize the following transcript into clear, concise bullet points. Keep the key information and main ideas. Write the summary in the same language as the input text. Format each point on a new line starting with a bullet point (\u2022).",
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
