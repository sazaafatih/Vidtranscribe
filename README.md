# VidTranscribe

A production-ready web application that transcribes Instagram Reels and TikTok videos from a pasted URL. Simply paste a link, and get the transcript instantly.

## Features

- **Instant Transcription**: Paste a TikTok or Instagram Reels URL and get the transcript in seconds
- **Language Detection**: Automatically detects the language of the video
- **Mobile-Friendly**: Fully responsive design works on all devices
- **Dark Theme**: Easy on the eyes with a modern dark interface
- **Copy to Clipboard**: One-click copying of transcripts
- **Free**: Uses the free tier of Groq's Whisper API

## Tech Stack

- **Backend**: Python, FastAPI
- **Frontend**: Vanilla HTML, CSS, and JavaScript (single file)
- **Video Download**: yt-dlp
- **Audio Extraction**: FFmpeg
- **Transcription**: Groq Whisper API (whisper-large-v3 model)
- **Deployment**: Railway.app (with nixpacks)

## Prerequisites

- Python 3.10 or higher
- FFmpeg (installed on your system)
- A free Groq API key

## Getting a Free Groq API Key

1. Visit [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to the API keys section
4. Create a new API key
5. Copy the key and save it for later

## Local Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd vidtranscribe
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 5. Run the Application

```bash
python main.py
```

The app will be available at `http://localhost:8000`

## Usage

1. Open the app in your browser
2. Paste a TikTok or Instagram Reels URL in the input field
3. Click "Transcribe" or press Enter
4. Wait for the transcription to complete
5. Copy the transcript using the "Copy to Clipboard" button

## Supported URLs

- TikTok: `https://www.tiktok.com/@username/video/...`
- Instagram Reels: `https://www.instagram.com/reel/...`

## Deployment to Railway

### Prerequisites

- GitHub account with the repository pushed
- Railway.app account

### Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Connect to Railway**
   - Go to [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Authorize and select your repository

3. **Add Environment Variables**
   - In the Railway dashboard, go to your project
   - Click on the service
   - Go to "Variables"
   - Add `GROQ_API_KEY` with your actual API key

4. **Deploy**
   - Railway will automatically detect the Python project
   - It will use `nixpacks` to install FFmpeg and dependencies
   - The app will start using the command in `railway.toml`
   - Your app will be live at the provided Railway URL

## API Endpoints

### POST /transcribe

Transcribes a video from a given URL.

**Request:**
```json
{
  "url": "https://www.tiktok.com/@username/video/123456789"
}
```

**Success Response (200):**
```json
{
  "transcript": "The transcribed text from the video...",
  "duration_seconds": 45.5,
  "language": "Indonesian"
}
```

**Error Response (400):**
```json
{
  "error": "Error message describing what went wrong"
}
```

## Error Handling

The app provides clear error messages for common issues:

- **Invalid URL**: "Only TikTok and Instagram Reels URLs are supported"
- **Video Too Long**: "Video too long, please use a shorter clip" (if audio > 24MB)
- **Network Error**: "Failed to transcribe video. Please try again."

## Limitations

- Maximum video length: Limited by the 24MB audio file size (typically 10-15 minutes depending on quality)
- Supported platforms: TikTok and Instagram Reels only
- Requires internet connection for video download and transcription

## Troubleshooting

### FFmpeg Not Found

Make sure FFmpeg is installed on your system:

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### Groq API Key Error

Ensure your `GROQ_API_KEY` is correctly set in the `.env` file and that you have a valid API key from [console.groq.com](https://console.groq.com)

### Video Download Fails

- Check your internet connection
- Ensure the URL is correct and publicly accessible
- Try a different video if the issue persists

## License

This project is open source and available under the MIT License.

## Support

For issues or questions, please open an issue on GitHub or contact support.
