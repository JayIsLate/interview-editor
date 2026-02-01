# Interview Auto-Editor

Automatically detect and remove pauses, filler words, and dead air from longform interview videos while keeping the raw, conversational feel. Export to Premiere Pro.

## Quick Start (Mac)

```bash
# 1. Clone the repo
git clone https://github.com/JayIsLate/interview-editor.git
cd interview-editor

# 2. Install FFmpeg
brew install ffmpeg

# 3. Install Python packages
pip3 install -r requirements.txt

# 4. Run the app
python3 -m streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

## Usage

1. **Paste your video file path** (e.g. `/Users/you/Movies/interview.mov`)
2. **Click Analyze** — transcribes and detects cuts
3. **Review cuts** — approve or reject each one
4. **Export** — download Premiere Pro XML

## Free vs Paid Mode

Toggle in the sidebar:

| Local Whisper (default) | OpenAI API |
|------------------------|------------|
| Free | ~$0.006/min |
| Slower | Faster |
| Runs on your CPU | Runs on OpenAI servers |

## Features

- Silence detection
- Filler word detection (um, uh, like, you know...)
- Off-mic speech detection
- False start detection
- Export to Premiere Pro XML, EDL, CSV
- YouTube thumbnail generator

## Requirements

- macOS
- Python 3.8+
- FFmpeg
- 8GB+ RAM recommended for local Whisper

## Troubleshooting

**FFmpeg not found:** Run `brew install ffmpeg`

**Browser crashes on upload:** Use the file path input instead of drag-and-drop for large videos

**Slow transcription:** Use a smaller Whisper model (tiny or base) in the sidebar
