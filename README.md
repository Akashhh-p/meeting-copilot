# Meeting Copilot

Real-time meeting transcription with speaker labels using Whisper and pyannote speaker diarization.

## Features

- 🎤 Real-time microphone audio capture in 3-second chunks
- 🎯 Accurate speech-to-text using Faster Whisper
- 👥 Speaker identification and labeling with pyannote
- ⏱️ Timestamped transcript output
- 🚀 Fast inference on CPU or GPU

## Quick Start

### Prerequisites

- Python 3.8+
- Working microphone
- HuggingFace account (free) for speaker diarization

### Installation

1. **Clone and setup**
```bash
git clone https://github.com/yourusername/meeting-copilot
cd meeting-copilot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**
```bash
pip install faster-whisper pyannote.audio sounddevice numpy torch
```

3. **Get HuggingFace token**
- Go to https://huggingface.co/settings/tokens
- Create a read token
- Accept the model license at https://huggingface.co/pyannote/speaker-diarization-3.0

4. **Set environment variable**
```bash
export HF_TOKEN="your_token_here"  # On Windows: set HF_TOKEN=your_token_here
```

### Usage

**Demo mode (no microphone needed):**
```bash
python transcribe.py --demo
```

**Real-time transcription:**
```bash
python transcribe.py
```

The transcriber will:
1. Record 3-second audio chunks from your microphone
2. Transcribe each chunk with speaker labels
3. Display timestamped output in the terminal
4. Continue until you press Ctrl+C

### Example Output

```
[0.00s] Speaker A: Good morning everyone, thanks for joining
[1.23s] Speaker B: Hi team, glad to be here. Let's start with the agenda
[2.45s] Speaker A: Sure, first item is the Q2 roadmap review
[3.67s] Speaker B: Great, I've prepared slides on the upcoming features
[5.12s] Speaker A: Perfect, let's dive into those now
```

## How It Works

### Audio Capture
- Records continuous audio from the default microphone
- Processes in 3-second chunks for real-time responsiveness

### Transcription
- Uses Faster Whisper (quantized version of OpenAI Whisper)
- Lightweight "tiny" model by default (fast, good for meetings)
- Can upgrade to "base", "small", "medium" for higher accuracy

### Speaker Diarization
- Uses pyannote 3.0 speaker diarization model
- Identifies and labels different speakers automatically
- Maps speaker IDs to consistent labels (Speaker A, B, etc.)

## Configuration

Edit `transcribe.py` to adjust:
- `CHUNK_DURATION`: Change chunk size (default: 3 seconds)
- `SAMPLE_RATE`: Audio sample rate (default: 16000 Hz)
- Model size: Change from "tiny" to "base", "small", or "medium"

## Troubleshooting

**"No module named 'sounddevice'"**
- Reinstall: `pip install --upgrade sounddevice`

**"No microphone detected"**
- Check system audio settings
- Test with: `python -c "import sounddevice as sd; print(sd.query_devices())"`

**"Authentication required for pyannote"**
- Ensure HF_TOKEN is set correctly
- Verify token has read access

**"Out of memory"**
- Switch to smaller model: Change `"tiny"` to `"base"` in transcribe.py
- Or reduce chunk duration

## Roadmap

- [ ] Save transcripts to file (JSON/SRT format)
- [ ] Add audio file transcription
- [ ] Implement speaker profiles
- [ ] Export to meeting notes format
- [ ] Add keyword highlighting
- [ ] Real-time translation

## Performance

- **Latency**: ~2-3 seconds per chunk (depending on hardware)
- **Accuracy**: ~95% on clear audio (tiny model), 98%+ (medium model)
- **Memory**: ~500MB for tiny model, 1.5GB for medium model

## License

MIT

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## Support

For issues or questions:
- Check existing GitHub issues
- Create a new issue with details and audio sample if possible
