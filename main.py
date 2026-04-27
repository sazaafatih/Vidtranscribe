import os
import re
import uuid
import json
import subprocess
import math
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import yt_dlp
from groq import Groq
from youtube_transcript_api import YouTubeTranscriptApi

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
    cookies_data = os.getenv("INSTAGRAM_COOKIES")
    if cookies_data:
        COOKIES_FILE.write_text(cookies_data)
        return True
    return False


HAS_COOKIES = setup_cookies()

MAX_CHUNK_MB = 24
SUPPORTED_PLATFORMS = ["tiktok.com", "instagram.com", "youtube.com", "youtu.be"]


def is_youtube_url(url):
    return "youtube.com" in url or "youtu.be" in url


def extract_video_id(url):
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:embed/)([a-zA-Z0-9_-]{11})',
        r'(?:shorts/)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_youtube_transcript(url):
    """Get YouTube transcript using youtube-transcript-api (no cookies needed)."""
    try:
        video_id = extract_video_id(url)
        if not video_id:
            return None, 0, None

        # Try to fetch transcript - try multiple languages
        transcript_list = None
        language = "en"

        try:
            # Try English first, then Indonesian, then any available
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
            language = "en"
        except Exception:
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['id'])
                language = "id"
            except Exception:
                try:
                    # Get any available transcript
                    available = YouTubeTranscriptApi.list_transcripts(video_id)
                    for transcript in available:
                        transcript_list = transcript.fetch()
                        language = transcript.language_code
                        break
                except Exception:
                    return None, 0, None

        if not transcript_list:
            return None, 0, None

        # Combine all text segments
        texts = [entry['text'] for entry in transcript_list if entry.get('text', '').strip()]
        full_text = " ".join(texts)

        # Calculate duration from last entry
        if transcript_list:
            last_entry = transcript_list[-1]
            duration = last_entry.get('start', 0) + last_entry.get('duration', 0)
        else:
            duration = 0

        return full_text, duration, language

    except Exception as e:
        print(f"YouTube transcript extraction failed: {e}")
        return None, 0, None


def split_audio(audio_path, max_size_mb=MAX_CHUNK_MB):
    audio_size_mb = audio_path.stat().st_size / (1024 * 1024)
    if audio_size_mb <= max_size_mb:
        return [audio_path]

    probe_cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
    total_duration = float(result.stdout.strip())

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


def chunk_text(text, max_chars=4000):
    """Split text into chunks at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_chars:
            chunks.append(text)
            break
        cut = text[:max_chars].rfind(". ")
        if cut == -1:
            cut = text[:max_chars].rfind(" ")
        if cut == -1:
            cut = max_chars
        else:
            cut += 1
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    return chunks


# Request models
class TranscribeRequest(BaseModel):
    url: str


class TranslateRequest(BaseModel):
    text: str


class SummarizeRequest(BaseModel):
    text: str


@app.on_event("startup")
async def startup_event():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/transcribe")
async def transcribe(request: TranscribeRequest):
    try:
        url = request.url.strip()

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        if not any(platform in url for platform in SUPPORTED_PLATFORMS):
            return {"error": "Supported platforms: TikTok, Instagram Reels, YouTube"}

        # For YouTube: use youtube-transcript-api (no cookies needed!)
        if is_youtube_url(url):
            transcript_text, yt_duration, sub_lang = get_youtube_transcript(url)
            if transcript_text:
                return {
                    "transcript": transcript_text,
                    "duration_seconds": yt_duration,
                    "language": sub_lang or "Unknown",
                    "source": "subtitles",
                }
            else:
                return {"error": "No subtitles available for this YouTube video. Only videos with captions (auto-generated or manual) are supported."}

        # For TikTok/Instagram: download and transcribe with Whisper
        request_id = str(uuid.uuid4())
        video_path = TEMP_DIR / f"{request_id}.mp4"
        audio_path = TEMP_DIR / f"{request_id}.mp3"
        chunk_paths = []

        try:
            ydl_opts = {
                "format": "best[ext=mp4]",
                "outtmpl": str(video_path.with_suffix("")),
                "quiet": True,
                "no_warnings": True,
                "no_playlist": True,
            }

            if "instagram.com" in url and HAS_COOKIES:
                ydl_opts["cookiefile"] = str(COOKIES_FILE)

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

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

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")

            if not audio_path.exists():
                raise FileNotFoundError("Audio extraction failed")

            chunks = split_audio(audio_path)
            chunk_paths = [c for c in chunks if c != audio_path]

            transcript_text, duration, language = transcribe_audio_chunks(chunks)

            return {
                "transcript": transcript_text,
                "duration_seconds": duration,
                "language": language,
                "source": "whisper",
            }

        finally:
            for temp_file in [video_path, audio_path] + chunk_paths:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except Exception:
                    pass
            try:
                for f in TEMP_DIR.glob(f"{request_id}*"):
                    f.unlink()
            except Exception:
                pass

    except Exception as e:
        print(f"Transcribe error: {e}")
        return {"error": "Failed to transcribe video. Please try again."}


@app.post("/translate")
async def translate(request: TranslateRequest):
    try:
        text = request.text.strip()
        if not text:
            return {"error": "No text provided for translation"}

        chunks = chunk_text(text, max_chars=4000)
        translated_parts = []

        for chunk in chunks:
            message = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional translator. Translate the following text to Indonesian (Bahasa Indonesia). Provide only the translation, nothing else.",
                    },
                    {
                        "role": "user",
                        "content": f"Translate this to Indonesian:\n\n{chunk}",
                    }
                ],
                model="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=8000,
            )
            translated_parts.append(message.choices[0].message.content.strip())

        translated_text = " ".join(translated_parts)

        return {
            "original_text": text,
            "translated_text": translated_text,
            "language": "Indonesian",
        }

    except Exception as e:
        print(f"Translate error: {e}")
        return {"error": "Failed to translate text. Please try again."}


@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    try:
        text = request.text.strip()
        if not text:
            return {"error": "No text provided for summarization"}

        chunks = chunk_text(text, max_chars=6000)

        if len(chunks) == 1:
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
                max_tokens=4000,
            )
            summary_text = message.choices[0].message.content.strip()
        else:
            partial_summaries = []
            for i, chunk in enumerate(chunks):
                message = groq_client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are a professional content summarizer. Summarize part {i+1} of {len(chunks)} of this transcript into concise bullet points. Keep key information. Write in the same language as the input. Format each point starting with \u2022.",
                        },
                        {
                            "role": "user",
                            "content": f"Summarize this:\n\n{chunk}",
                        }
                    ],
                    model="llama-3.1-8b-instant",
                    temperature=0.3,
                    max_tokens=2000,
                )
                partial_summaries.append(message.choices[0].message.content.strip())

            combined = "\n".join(partial_summaries)
            message = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional content summarizer. Combine these partial summaries into one clean, organized summary with bullet points (\u2022). Remove duplicates and keep only the most important points. Write in the same language as the input.",
                    },
                    {
                        "role": "user",
                        "content": f"Combine these summaries into one:\n\n{combined}",
                    }
                ],
                model="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=4000,
            )
            summary_text = message.choices[0].message.content.strip()

        return {
            "original_text": text,
            "summary": summary_text,
        }

    except Exception as e:
        print(f"Summarize error: {e}")
        return {"error": "Failed to summarize text. Please try again."}


@app.get("/")
async def serve_frontend():
    return FileResponse("index.html", media_type="text/html")


@app.get("/index.html")
async def serve_index():
    return FileResponse("index.html", media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
