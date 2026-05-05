# Meeting Co-Pilot

Meeting Co-Pilot is a local-first AI meeting assistant that captures live audio, streams real-time transcripts, improves speaker labels in the background, answers questions over the transcript, and generates structured meeting summaries.

It is designed for real meetings where transcription must keep running even while diarization, indexing, Q&A, or summary generation is busy.

## Highlights

- Real-time transcription with `faster-whisper`
- Microphone, system audio, and mixed audio capture
- Rolling audio chunks with overlap to reduce word cut-offs
- Non-blocking worker pipeline for long-running meetings
- Background speaker diarization with `pyannote.audio`
- ChromaDB-backed RAG over transcript chunks
- Local Q&A and summary generation through Ollama
- Live browser UI with audio level meter and speaker colors
- Debug tools for audio devices, RMS, peak level, and queue health
- Tauri desktop packaging scaffold for Windows builds

## Product Flow

```text
Audio Capture
  -> audio_queue
  -> Transcription Worker
  -> transcript events
  -> UI Stream + File Save + RAG Index

Background:
  Diarization -> speaker_update events
  Q&A         -> transcript-grounded answers
  Summary     -> structured meeting notes
```

The transcription path is intentionally protected. Slow work is moved to separate queues/workers so audio capture and transcription do not stop.

## Features

| Area | Details |
| --- | --- |
| Live transcription | 4 second rolling windows, 0.5 second overlap, Whisper loaded once |
| Audio sources | `mic`, `system`, `mixed`, `debug` |
| Low-volume handling | input gain, soft VAD, force-transcribe, RMS/peak logging |
| Speaker diarization | pyannote background worker, stable `Speaker A/B/C` labels, fallback to `Speaker Unknown` |
| RAG Q&A | ChromaDB vector store, cached embeddings, transcript-only answers |
| Summary | meeting summary, decisions, action items, open questions, next steps |
| Observability | `/health`, `/metrics`, `logs/app.log`, `logs/audio_debug.log` |
| Desktop app | Tauri scaffold with Python backend sidecar |

## Repository Structure

```text
meeting_copilot/
  server.py                  FastAPI app, live pipeline, WebSocket streaming
  rag.py                     ChromaDB RAG, Q&A, summary prompts
  transcribe.py              CLI audio diagnostics and fallback recorder
  benchmark.py               Runtime latency/resource/WER helper
  requirements.txt           Python dependencies
  .env.example               Environment configuration example
  frontend/
    index.html               Browser UI
  desktop/                   Tauri desktop app scaffold
  scripts/
    build_windows.ps1        Windows packaging script
  outputs/                   Transcript, JSONL, summaries
  logs/                      App and audio debug logs
```

## Requirements

- Windows 10/11 recommended for current audio capture work
- Python 3.10+
- Ollama installed and running
- Microphone for `mic` mode
- Windows Stereo Mix or `soundcard` loopback support for `system` mode
- HuggingFace token for speaker diarization
- Optional: CUDA GPU for lower latency

## Installation

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Install dependencies:

```powershell
.\venv\Scripts\pip.exe install -r requirements.txt
```

Pull local Ollama models:

```powershell
ollama pull llama3
ollama pull nomic-embed-text
```

Copy the environment template if desired:

```powershell
Copy-Item .env.example .env
```

## Configuration

Minimum recommended PowerShell environment:

```powershell
$env:HF_TOKEN="your_huggingface_token"
$env:WHISPER_MODEL="small.en"
$env:AUDIO_SOURCE="mic"
$env:AUDIO_CHUNK_SECONDS="4"
$env:AUDIO_OVERLAP_SECONDS="0.5"
$env:VAD_MODE="soft"
$env:PORT="8012"
```

Important variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | empty | Enables pyannote speaker diarization |
| `WHISPER_MODEL` | `small.en` | faster-whisper model |
| `AUDIO_SOURCE` | `mic` | `mic`, `system`, `mixed`, or `debug` |
| `AUDIO_DEVICE_INDEX` | `auto` | Selected microphone device |
| `AUDIO_SYSTEM_DEVICE_INDEX` | `auto` | Selected system/Stereo Mix device |
| `AUDIO_INPUT_GAIN` | `1.5` | Gain applied before VAD/transcription |
| `VAD_MODE` | `soft` | `off`, `soft`, or `normal` |
| `FORCE_TRANSCRIBE` | `false` | Transcribe chunks even when VAD thinks they are quiet |
| `MIN_RMS_THRESHOLD` | `0.0015` | Silence threshold |
| `LOW_AUDIO_WARNING_THRESHOLD` | `0.003` | UI/log warning threshold |
| `OLLAMA_MODEL` | `llama3` | Local LLM for Q&A and summary |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Local embedding model |

If `HF_TOKEN` is missing, transcription still works and diarization is disabled with a clear warning.

## Running the App

Start the FastAPI server:

```powershell
.\venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8012
```

Open the UI:

```text
http://127.0.0.1:8012
```

The UI includes:

- Start/Stop recording
- Audio source selector
- Device selector
- VAD mode selector
- Force-transcribe toggle
- Live audio level meter
- Live transcript
- Q&A panel
- Summary panel
- Speaker list

## Audio Modes

### Laptop microphone

```powershell
$env:AUDIO_SOURCE="mic"
$env:VAD_MODE="soft"
```

### Mobile phone speaker near laptop mic

Use this when the phone speaker is quiet or far from the laptop:

```powershell
$env:AUDIO_SOURCE="mic"
$env:VAD_MODE="off"
$env:FORCE_TRANSCRIBE="true"
$env:AUDIO_INPUT_GAIN="2.0"
```

### YouTube, Zoom, Meet, Teams audio on the same PC

```powershell
$env:AUDIO_SOURCE="system"
```

System audio capture uses `soundcard` loopback when available and falls back to Windows Stereo Mix when possible.

### Mic plus system audio

```powershell
$env:AUDIO_SOURCE="mixed"
```

Use this for meetings where your voice comes from the mic and remote speakers come through PC audio.

## Audio Debugging

List all devices:

```powershell
.\venv\Scripts\python.exe transcribe.py --list-devices
```

Print RMS/peak levels every second:

```powershell
.\venv\Scripts\python.exe transcribe.py --debug-audio --debug-seconds 30
```

Useful backend endpoints:

```powershell
Invoke-RestMethod http://127.0.0.1:8012/health
Invoke-RestMethod http://127.0.0.1:8012/devices
Invoke-RestMethod http://127.0.0.1:8012/metrics
```

Logs:

```text
logs/app.log
logs/audio_debug.log
```

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Backend, model, queue, output status |
| `GET` | `/metrics` | Runtime metrics, queue sizes, timings |
| `GET` | `/devices` | Audio devices and loopback availability |
| `GET` | `/transcript` | Current transcript segments |
| `POST` | `/ask` | Ask a question over indexed transcript |
| `POST` | `/summary` | Generate structured summary |
| `WS` | `/ws` | Live transcript, status, speaker update stream |

Example Q&A request:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8012/ask `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"question":"What deadline was discussed?"}'
```

## Outputs

```text
outputs/transcript.txt
outputs/transcript_final.txt
outputs/transcript_segments.jsonl
outputs/summary.md
outputs/benchmark_report.json
```

Live transcript format:

```text
[00:00:05] Speaker A: Hello everyone, today we will discuss the project deadline.
```

Q&A behavior:

```text
Question: What deadline was discussed?
Answer: The team discussed submitting the project by Friday.
Evidence: Speaker A at 00:00:05
```

If the transcript does not contain the answer:

```text
This was not mentioned in the meeting.
```

## Benchmarking

Measure runtime latency and resource use while the server is running:

```powershell
.\venv\Scripts\python.exe benchmark.py --base-url http://127.0.0.1:8012 --seconds 300
```

Measure WER with a reference transcript:

```powershell
.\venv\Scripts\python.exe benchmark.py `
  --reference .\ground_truth.txt `
  --hypothesis .\outputs\transcript.txt `
  --seconds 300
```

The project targets low-latency transcription on suitable hardware, but final latency and WER must be measured on the actual machine, audio device, room, and Whisper model. The benchmark tool exists so published numbers are measured, not guessed.

## Testing Checklist

1. Direct mic speech: `AUDIO_SOURCE=mic`, `VAD_MODE=soft`.
2. Mobile speaker audio: `AUDIO_SOURCE=mic`, `VAD_MODE=off`, `FORCE_TRANSCRIBE=true`.
3. PC YouTube audio: `AUDIO_SOURCE=system`.
4. Online meeting audio: `AUDIO_SOURCE=system` or `mixed`.
5. Long session: run for 1 hour and monitor `/metrics`.
6. Q&A during transcription: ask questions and confirm transcript still grows.
7. Summary during transcription: generate summary and confirm capture continues.
8. Diarization fallback: unset `HF_TOKEN` and confirm transcription continues.

## Desktop Packaging

The repo includes a Tauri scaffold that runs the FastAPI backend as a sidecar.

Prerequisites:

- Node.js and npm
- Rust toolchain
- Tauri Windows prerequisites
- Python virtual environment with requirements installed

Build:

```powershell
.\scripts\build_windows.ps1
```

The script:

1. Installs Python requirements.
2. Builds `server.py` into a PyInstaller sidecar.
3. Copies the sidecar into `desktop/src-tauri/bin`.
4. Runs the Tauri build.

## Privacy

Meeting Co-Pilot is local-first:

- Whisper runs locally through `faster-whisper`.
- RAG uses local ChromaDB.
- Embeddings and LLM calls use local Ollama.
- Transcript and summary files stay on disk under `outputs/`.

The only external requirement is the HuggingFace token needed to access/load pyannote diarization models. If diarization is disabled, transcription, Q&A, and summary still work locally.

## Known Limitations

- Diarization is best-effort; noisy rooms and similar voices can still cause label mistakes.
- System audio capture depends on Windows audio configuration, `soundcard`, or Stereo Mix availability.
- CPU-only systems may need `base.en` or `tiny.en` for smoother long sessions.
- `mixed` mode combines mic and system audio streams but does not perform echo cancellation.
- Desktop packaging scaffold is present, but final installer signing and release automation are not included.

## License

MIT
