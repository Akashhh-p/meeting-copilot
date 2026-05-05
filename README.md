# Meeting Copilot

Real-time meeting transcription with automatic transcript saving. Captures audio from your microphone and transcribes it using OpenAI's Whisper model.

## Features

✅ **Real-time Transcription** - Transcribe audio as you record  
✅ **Speaker Diarization** - Identifies and labels different speakers (Speaker A, Speaker B, etc.)  
✅ **Automatic Session Management** - Each session saved in a timestamped folder  
✅ **Text & JSON Output** - Save transcripts in both human-readable and structured formats  
✅ **Audio Device Selection** - Choose which microphone to use  
✅ **Speech Detection** - Only processes chunks that contain actual speech  
✅ **Multiple Model Sizes** - Tiny, base, small, medium, or large Whisper models  
✅ **CPU-based** - Works without CUDA/GPU (with timeout protection)  

## Installation

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate it:**
   - Windows: `venv\Scripts\activate.ps1`
   - Linux/Mac: `source venv/bin/activate`

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Get HuggingFace token (optional but recommended for speaker diarization):**
   - Go to https://huggingface.co/settings/tokens
   - Create a **read-only** token
   - Accept the model license at https://huggingface.co/pyannote/speaker-diarization-3.0
   - Set environment variable (Windows PowerShell):
     ```powershell
     $env:HF_TOKEN = "your_token_here"
     ```
   - Or Linux/Mac:
     ```bash
     export HF_TOKEN="your_token_here"
     ```

## Usage

### Basic Usage (Default Device)
```bash
python transcribe.py
```

### Select Audio Device Interactively
```bash
python transcribe.py --select-device
```
This will show all available microphones and let you choose one.

### Choose a Specific Device
```bash
python transcribe.py --device 0
```

### Use Different Model Size
```bash
python transcribe.py --model small
```

Model options: `tiny` (fastest), `base` (recommended), `small`, `medium`, `large` (most accurate)

### Combined Example
```bash
python transcribe.py --select-device --model medium --hf-token your_token_here
```

### With Environment Variable (Recommended)
```powershell
$env:HF_TOKEN = "your_huggingface_token"
python transcribe.py --select-device --model base
```

The token can also be passed via command line:
```bash
python transcribe.py --hf-token "your_token_here"
```

## How It Works

1. **Recording** - Captures 5-second chunks of audio
2. **Speech Detection** - Analyzes audio energy to detect if speech is present
3. **Transcription** - Sends speech chunks to Whisper for transcription
4. **Speaker Diarization** - Identifies different speakers (requires HuggingFace token)
5. **Storage** - Saves transcripts in real-time and on exit
6. **Output** - Creates a timestamped folder with transcript files

### Speaker Diarization Details
- Uses **pyannote 3.0** speaker diarization model
- Automatically identifies and labels different speakers
- Maps speaker IDs to consistent labels (Speaker A, Speaker B, etc.)
- **Timeout protection:** 10-second timeout prevents freezes on CPU
- **Non-blocking:** Runs in background thread without blocking transcription
- **Graceful fallback:** If diarization fails, continues with timestamps only

## Output Files

Transcripts are saved in `meeting_transcripts/YYYYMMDD_HHMMSS/`:
- `transcript.txt` - Human-readable format with timestamps
- `transcript.json` - Structured JSON format for parsing

Example:
```
meeting_transcripts/
└── 20260505_100932/
    ├── transcript.txt
    └── transcript.json
```

### Example Transcript Format

**transcript.txt:**
```
MEETING TRANSCRIPT
Started: 2026-05-05 10:09:32
Ended: 2026-05-05 10:10:15
Speakers: Speaker A, Speaker B
============================================================

[0.00s - 1.25s] Speaker A: Good morning everyone, let's start the meeting
[1.50s - 3.20s] Speaker B: Today we'll discuss the new product roadmap
[3.45s - 5.10s] Speaker A: First, let me share some updates on Q2 plans
```

**transcript.json:**
```json
{
  "session_start": "2026-05-05T10:09:32.123456",
  "session_end": "2026-05-05T10:10:15.654321",
  "speakers": ["Speaker A", "Speaker B"],
  "chunks": [
    {
      "chunk_num": 1,
      "timestamp": "2026-05-05T10:09:35.234567",
      "segments": [
        {
          "start": 0.0,
          "end": 1.25,
          "speaker": "Speaker A",
          "text": "Good morning everyone, let's start the meeting"
        }
      ]
    }
  ]
}
```

## Troubleshooting

### No Audio is Being Recorded
- Run with `--select-device` to list and choose the correct microphone
- Check your system volume is not muted
- Ensure microphone permissions are granted

### Speaker Diarization Not Working
- **Missing HuggingFace token:** Get one from https://huggingface.co/settings/tokens
- **Invalid token:** Make sure to create a READ token, not a write token
- **Model license not accepted:** Accept the license at https://huggingface.co/pyannote/speaker-diarization-3.0
- **Timeout issues:** Diarization has a 10-second timeout on CPU. If it times out, it continues without speaker labels (non-fatal)

### Wrong Audio Device Being Used
```bash
python transcribe.py --select-device
# Choose the correct device number
```

### Slow Transcription
Use the smaller model:
```bash
python transcribe.py --model tiny
```

### Missing Dependencies
Reinstall requirements:
```bash
pip install -r requirements.txt --force-reinstall
```

## Tips for Best Results

1. **Speak clearly** - Whisper works better with clear speech
2. **Minimal background noise** - Reduce ambient noise for better accuracy
3. **Use `base` or `small` model** - Good balance between speed and accuracy
4. **Shorter chunks** - Current 2-second chunks work well for real-time transcription

## Controls

- **Ctrl+C** - Stop recording and save transcript
- The app will save your transcript automatically

## Model Comparison

| Model | Speed | Accuracy | Memory | Recommended For |
|-------|-------|----------|--------|-----------------|
| tiny | ⚡⚡⚡ | ⭐ | ✓ | Testing, demos |
| base | ⚡⚡ | ⭐⭐⭐ | ✓ | **Default choice** |
| small | ⚡ | ⭐⭐⭐⭐ | ✓✓ | Better accuracy |
| medium | - | ⭐⭐⭐⭐⭐ | ✓✓✓ | High accuracy |
| large | - | ⭐⭐⭐⭐⭐ | ✓✓✓✓ | Maximum accuracy |

## Requirements

- Python 3.8+
- Microphone
- ~500MB disk space for base model (more for larger models)

## License

MIT

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
