# Interview Auto-Editor

A web-based tool that automatically detects pauses, filler words, and dead space in interview videos, lets you review/approve cuts, then exports to Premiere Pro.

## Features

- **Automatic transcription** using OpenAI Whisper (runs locally, no API keys needed)
- **Silence detection** - finds gaps in audio
- **Filler word detection** - catches "um", "uh", "like", "you know", etc.
- **Long pause detection** - identifies mid-sentence hesitations
- **Review interface** - approve or reject each proposed cut
- **Timeline visualization** - see cuts on a visual timeline
- **Premiere Pro export** - generates XML that Premiere imports natively

## Requirements

- Python 3.8+
- FFmpeg (must be installed on your system)

## Installation

### 1. Install FFmpeg

**macOS (with Homebrew):**
```bash
brew install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH.

**Linux:**
```bash
sudo apt install ffmpeg
```

### 2. Install Python dependencies

```bash
cd interview-editor
pip install -r requirements.txt
```

Note: The first time you run the app, it will download the Whisper model (about 140MB for the "base" model).

## Usage

### Run the app

```bash
streamlit run app.py
```

This opens the app in your browser at http://localhost:8501

### Workflow

1. **Upload** - Drag and drop your video file (MP4, MOV, AVI, MKV, WebM)
2. **Analyze** - Click "Detect Cuts" to process the video
3. **Review** - Approve or reject each proposed cut
4. **Export** - Download the Premiere-compatible XML

### Import into Premiere Pro

1. In Premiere, go to File > Import
2. Select the downloaded XML file
3. The sequence with edit points will appear in your project
4. Review and adjust cuts as needed
5. Export your final video

## Configuration

Adjust settings in the sidebar:

| Setting | Default | Description |
|---------|---------|-------------|
| Minimum silence | 0.8 sec | Silence duration to flag |
| Silence threshold | -40 dB | Audio level for silence |
| Filler words | um, uh, like... | Words to detect |
| Padding | 0.1 sec | Buffer around cuts |
| Whisper model | base | Accuracy vs speed |
| Frame rate | 29.97 | Match your video |

You can also edit `config.yaml` directly.

## Whisper Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | 39M | Fastest | Basic |
| base | 74M | Fast | Good |
| small | 244M | Medium | Better |
| medium | 769M | Slow | Great |
| large | 1550M | Slowest | Best |

For most interviews, "base" or "small" works well.

## Export Formats

### Edited Sequence (XML)
Creates a Premiere sequence with cuts already applied. Each segment is a separate clip on the timeline.

### Markers Only (XML)
Adds markers at cut points without removing anything. Good for manual review.

### EDL
Edit Decision List format, compatible with many video editors.

## Troubleshooting

### "FFmpeg not found"
Make sure FFmpeg is installed and in your system PATH.

### Slow processing
- Use a smaller Whisper model (tiny or base)
- Process shorter video clips
- Ensure you have enough RAM (8GB+ recommended)

### Cuts not accurate
- Adjust silence threshold if silence isn't being detected
- Add/remove filler words in the settings
- Use a larger Whisper model for better transcription

## Project Structure

```
interview-editor/
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
├── config.yaml            # Default settings
├── src/
│   ├── transcriber.py     # Whisper speech-to-text
│   ├── silence_detector.py # Find audio gaps
│   ├── filler_detector.py  # Find filler words
│   ├── cut_manager.py      # Merge and manage cuts
│   └── premiere_export.py  # Generate Premiere XML
└── README.md
```

## License

MIT License
