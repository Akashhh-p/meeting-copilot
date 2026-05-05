from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import queue
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import torch
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
from rag import TranscriptRAG

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
audio_debug_file = LOG_DIR / "audio_debug.log"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("meeting-copilot")
audio_logger = logging.getLogger("meeting-copilot.audio")
audio_logger.setLevel(os.getenv("AUDIO_DEBUG_LOG_LEVEL", "INFO"))
if not any(isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", "") == str(audio_debug_file) for handler in audio_logger.handlers):
    audio_logger.addHandler(logging.FileHandler(audio_debug_file, encoding="utf-8"))

HF_TOKEN = os.getenv("HF_TOKEN")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small.en")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
CHANNELS = 1
AUDIO_SOURCE = os.getenv("AUDIO_SOURCE", "mic").lower()
AUDIO_DEVICE_INDEX = os.getenv("AUDIO_DEVICE_INDEX", os.getenv("AUDIO_DEVICE", "auto"))
AUDIO_SYSTEM_DEVICE_INDEX = os.getenv("AUDIO_SYSTEM_DEVICE_INDEX", "auto")
AUDIO_INPUT_GAIN = float(os.getenv("AUDIO_INPUT_GAIN", "1.5"))
FORCE_TRANSCRIBE = os.getenv("FORCE_TRANSCRIBE", "false").lower() in {"1", "true", "yes", "on"}
VAD_MODE = os.getenv("VAD_MODE", "soft").lower()
AUDIO_CHUNK_SECONDS = float(os.getenv("AUDIO_CHUNK_SECONDS", "4"))
AUDIO_OVERLAP_SECONDS = float(os.getenv("AUDIO_OVERLAP_SECONDS", "0.5"))
AUDIO_QUEUE_WARNING = int(os.getenv("AUDIO_QUEUE_WARNING", "20"))
WORKER_QUEUE_WARNING = int(os.getenv("WORKER_QUEUE_WARNING", "100"))
MIN_RMS_THRESHOLD = float(os.getenv("MIN_RMS_THRESHOLD", os.getenv("SILENCE_RMS_THRESHOLD", "0.0015")))
LOW_AUDIO_WARNING_THRESHOLD = float(os.getenv("LOW_AUDIO_WARNING_THRESHOLD", "0.003"))
SILENCE_PEAK_THRESHOLD = float(os.getenv("SILENCE_PEAK_THRESHOLD", "0.01"))
DIARIZATION_INTERVAL_SECONDS = float(os.getenv("DIARIZATION_INTERVAL_SECONDS", "30"))
DIARIZATION_NUM_SPEAKERS = os.getenv("DIARIZATION_NUM_SPEAKERS")
DIARIZATION_MIN_SPEAKERS = int(os.getenv("DIARIZATION_MIN_SPEAKERS", "1"))
DIARIZATION_MAX_SPEAKERS = int(os.getenv("DIARIZATION_MAX_SPEAKERS", "4"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

DIARIZATION_MODELS = (
    "pyannote/speaker-diarization-community-1",
    "pyannote/speaker-diarization-3.0",
)


@dataclass
class AudioBlock:
    block_id: int
    source: str
    start_sample: int
    audio: np.ndarray
    captured_at: float
    rms: float
    peak: float


@dataclass
class RuntimeAudioConfig:
    source: str = AUDIO_SOURCE
    mic_device: int | None = None
    system_device: int | None = None
    gain: float = AUDIO_INPUT_GAIN
    vad_mode: str = VAD_MODE
    force_transcribe: bool = FORCE_TRANSCRIBE


app = FastAPI(title="Meeting Co-Pilot")
process_started_at = time.time()
audio_queue: queue.Queue[AudioBlock] = queue.Queue()
audio_capture_lock = threading.Lock()
captured_samples = 0
captured_blocks = 0
stop_recording = threading.Event()
model_lock = threading.Lock()
whisper_model: WhisperModel | None = None
diarizer = None

transcript_lock = threading.Lock()
transcript_log: list[dict[str, Any]] = []
segment_keys: set[str] = set()
speaker_names: dict[str, str] = {}
session_audio_lock = threading.Lock()
session_audio_blocks: list[np.ndarray] = []
session_started_at: float | None = None
next_segment_num = 0
next_window_chunk_id = 0
runtime_metrics_lock = threading.Lock()
runtime_metrics: dict[str, Any] = {
    "captured_blocks": 0,
    "transcribed_chunks": 0,
    "skipped_chunks": 0,
    "saved_segments": 0,
    "rag_indexed_segments": 0,
    "speaker_updates": 0,
    "last_rms": 0.0,
    "last_peak": 0.0,
    "last_transcription_time": 0.0,
    "last_diarization_time": 0.0,
    "last_rag_time": 0.0,
    "last_qa_time": 0.0,
    "last_summary_time": 0.0,
}

transcript_file = OUTPUT_DIR / "transcript.txt"
transcript_final_file = OUTPUT_DIR / "transcript_final.txt"
transcript_jsonl_file = OUTPUT_DIR / "transcript_segments.jsonl"
summary_file = OUTPUT_DIR / "summary.md"
for output_file in (transcript_file, transcript_final_file, transcript_jsonl_file, summary_file):
    output_file.touch(exist_ok=True)

rag_engine = TranscriptRAG(
    embed_model=OLLAMA_EMBED_MODEL,
    top_k=RAG_TOP_K,
    keep_alive=OLLAMA_KEEP_ALIVE,
    llm_model=OLLAMA_MODEL,
)


def parse_device_index(value: str | None) -> int | None:
    if value is None or value.lower() == "auto" or value == "":
        return None
    return int(value)


def audio_stats(audio: np.ndarray) -> tuple[float, float]:
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    return rms, peak


def list_audio_devices() -> list[dict[str, Any]]:
    devices = sd.query_devices()
    default_in, default_out = sd.default.device
    result: list[dict[str, Any]] = []
    for index, device in enumerate(devices):
        result.append(
            {
                "index": index,
                "name": str(device.get("name", "")),
                "hostapi": int(device.get("hostapi", -1)),
                "max_input_channels": int(device.get("max_input_channels", 0)),
                "max_output_channels": int(device.get("max_output_channels", 0)),
                "default_samplerate": float(device.get("default_samplerate", 0)),
                "is_default_input": index == default_in,
                "is_default_output": index == default_out,
            }
        )
    return result


def find_wasapi_hostapi() -> int | None:
    for index, api in enumerate(sd.query_hostapis()):
        if "wasapi" in str(api.get("name", "")).lower():
            return index
    return None


def default_system_output_device() -> int | None:
    default_out = sd.default.device[1]
    if default_out is not None and default_out >= 0:
        return int(default_out)
    wasapi = find_wasapi_hostapi()
    if wasapi is None:
        return None
    for item in list_audio_devices():
        if item["hostapi"] == wasapi and item["max_output_channels"] > 0:
            return int(item["index"])
    return None


def system_capture_available() -> bool:
    try:
        import soundcard  # noqa: F401

        return True
    except Exception:
        return stereo_mix_device() is not None


def build_audio_config(payload: dict[str, Any] | None = None) -> RuntimeAudioConfig:
    payload = payload or {}
    source = str(payload.get("audio_source") or AUDIO_SOURCE or "mic").lower()
    if source not in {"mic", "system", "mixed", "debug"}:
        source = "mic"
    vad_mode = str(payload.get("vad_mode") or VAD_MODE or "soft").lower()
    if vad_mode not in {"off", "soft", "normal"}:
        vad_mode = "soft"
    force = payload.get("force_transcribe")
    if force is None:
        force_bool = FORCE_TRANSCRIBE
    else:
        force_bool = bool(force)
    mic_value = payload.get("audio_device_index")
    system_value = payload.get("system_device_index")
    return RuntimeAudioConfig(
        source=source,
        mic_device=parse_device_index(str(mic_value)) if mic_value not in (None, "") else parse_device_index(AUDIO_DEVICE_INDEX),
        system_device=parse_device_index(str(system_value)) if system_value not in (None, "") else parse_device_index(AUDIO_SYSTEM_DEVICE_INDEX),
        gain=float(payload.get("audio_input_gain") or AUDIO_INPUT_GAIN),
        vad_mode=vad_mode,
        force_transcribe=force_bool,
    )


def log_device_table() -> None:
    audio_logger.info("platform=%s sample_rate=%d default_device=%s", platform.platform(), SAMPLE_RATE, sd.default.device)
    for device in list_audio_devices():
        audio_logger.info(
            "device index=%s name=%s input=%s output=%s rate=%s default_in=%s default_out=%s hostapi=%s",
            device["index"],
            device["name"],
            device["max_input_channels"],
            device["max_output_channels"],
            device["default_samplerate"],
            device["is_default_input"],
            device["is_default_output"],
            device["hostapi"],
        )


def update_metric(key: str, value: Any) -> None:
    with runtime_metrics_lock:
        runtime_metrics[key] = value


def increment_metric(key: str, amount: int = 1) -> None:
    with runtime_metrics_lock:
        runtime_metrics[key] = int(runtime_metrics.get(key, 0)) + amount


def metrics_snapshot() -> dict[str, Any]:
    with runtime_metrics_lock:
        snapshot = dict(runtime_metrics)
    snapshot.update(
        {
            "uptime_seconds": round(time.time() - process_started_at, 2),
            "audio_queue_size": audio_queue.qsize(),
            "segments": len(transcript_snapshot()),
        }
    )
    try:
        import psutil

        process = psutil.Process(os.getpid())
        snapshot.update(
            {
                "cpu_percent": process.cpu_percent(interval=0.0),
                "memory_mb": round(process.memory_info().rss / (1024 * 1024), 2),
                "threads": process.num_threads(),
            }
        )
    except Exception as exc:
        snapshot["resource_error"] = str(exc)
    return snapshot


def get_whisper_model() -> WhisperModel:
    global whisper_model
    if whisper_model is None:
        with model_lock:
            if whisper_model is None:
                logger.info("Loading faster-whisper model=%s device=%s compute=%s", WHISPER_MODEL, DEVICE, COMPUTE_TYPE)
                whisper_model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
                logger.info("Whisper model ready")
    return whisper_model


def get_diarizer():
    global diarizer
    if diarizer is not None:
        return diarizer
    if not HF_TOKEN:
        logger.warning("HF_TOKEN missing; diarization disabled")
        return None
    with model_lock:
        if diarizer is not None:
            return diarizer
        from pyannote.audio import Pipeline

        for model_name in DIARIZATION_MODELS:
            try:
                logger.info("Loading pyannote diarization model=%s", model_name)
                loaded = Pipeline.from_pretrained(model_name, token=HF_TOKEN)
                if DEVICE == "cuda":
                    loaded = loaded.to(torch.device("cuda"))
                diarizer = loaded
                logger.info("Diarization model ready")
                return diarizer
            except Exception as exc:
                logger.warning("Failed to load diarization model %s: %s", model_name, exc)
    logger.error("No diarization model could be loaded")
    return None


def audio_callback_factory(source: str, config: RuntimeAudioConfig):
    def audio_callback(indata, frames, time_info, status):
        enqueue_audio_block(source, indata, status, config)

    return audio_callback


def enqueue_audio_block(source: str, indata, status, config: RuntimeAudioConfig) -> None:
    global captured_blocks, captured_samples
    if status:
        logger.warning("Audio input status source=%s status=%s", source, status)
    mono = np.asarray(indata, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    mono = mono.reshape(-1)
    if config.gain != 1.0:
        mono = np.clip(mono * config.gain, -1.0, 1.0).astype(np.float32)
    rms, peak = audio_stats(mono)
    with audio_capture_lock:
        block_id = captured_blocks
        start_sample = captured_samples
        captured_blocks += 1
        captured_samples += len(mono)
    audio_queue.put(
        AudioBlock(
            block_id=block_id,
            source=source,
            start_sample=start_sample,
            audio=mono.copy(),
            captured_at=time.time(),
            rms=rms,
            peak=peak,
        )
    )
    qsize = audio_queue.qsize()
    if qsize >= AUDIO_QUEUE_WARNING:
        logger.warning("audio queue high block_id=%d source=%s queue_size=%d", block_id, source, qsize)
    if 0 < rms < LOW_AUDIO_WARNING_THRESHOLD:
        logger.warning("low audio level block_id=%d source=%s rms=%.6f peak=%.6f", block_id, source, rms, peak)
    increment_metric("captured_blocks")
    update_metric("last_rms", rms)
    update_metric("last_peak", peak)
    audio_logger.info(
        "captured block_id=%d source=%s start_sample=%d samples=%d rms=%.6f peak=%.6f queued=%d",
        block_id,
        source,
        start_sample,
        len(mono),
        rms,
        peak,
        qsize,
    )


def reset_session() -> None:
    global captured_blocks, captured_samples, next_segment_num, next_window_chunk_id, session_started_at
    stop_recording.clear()
    with audio_capture_lock:
        captured_blocks = 0
        captured_samples = 0
    with transcript_lock:
        transcript_log.clear()
        segment_keys.clear()
        speaker_names.clear()
        next_segment_num = 0
        next_window_chunk_id = 0
    with session_audio_lock:
        session_audio_blocks.clear()
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break
    rag_engine.clear()
    session_started_at = time.time()
    transcript_file.write_text("", encoding="utf-8")
    transcript_final_file.write_text("", encoding="utf-8")
    transcript_jsonl_file.write_text("", encoding="utf-8")
    summary_file.write_text("", encoding="utf-8")
    logger.info("Session reset")


def format_time(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def format_entry(entry: dict[str, Any]) -> str:
    return f"[{format_time(entry['start'])}] {entry['speaker']}: {entry['text']}"


def vad_decision(audio: np.ndarray, config: RuntimeAudioConfig) -> tuple[bool, float, float, str]:
    rms, peak = audio_stats(audio)
    if config.force_transcribe:
        return False, rms, peak, "force_transcribe"
    if config.vad_mode == "off":
        return False, rms, peak, "vad_off"
    if config.vad_mode == "normal":
        rms_threshold = max(MIN_RMS_THRESHOLD * 2.0, 0.003)
        peak_threshold = max(SILENCE_PEAK_THRESHOLD, 0.015)
    else:
        rms_threshold = MIN_RMS_THRESHOLD
        peak_threshold = SILENCE_PEAK_THRESHOLD
    if rms == 0.0 and peak == 0.0:
        return True, rms, peak, "digital_silence"
    if rms < rms_threshold and peak < peak_threshold:
        if config.vad_mode == "soft" and peak >= peak_threshold * 0.5:
            return False, rms, peak, "soft_low_volume_kept"
        return True, rms, peak, f"{config.vad_mode}_below_threshold"
    if 0 < rms < LOW_AUDIO_WARNING_THRESHOLD:
        return False, rms, peak, "low_audio_kept"
    return False, rms, peak, "speech_or_audio_detected"


def transcribe_audio_window(audio: np.ndarray) -> list[dict[str, Any]]:
    model = get_whisper_model()
    started = time.perf_counter()
    segments, _ = model.transcribe(
        audio.astype(np.float32),
        beam_size=1,
        language="en",
        vad_filter=False,
        condition_on_previous_text=True,
        temperature=0,
    )
    output = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            output.append({"start": float(segment.start), "end": float(segment.end), "text": text})
    logger.info("whisper transcribed segments=%d transcription_time=%.3fs", len(output), time.perf_counter() - started)
    update_metric("last_transcription_time", time.perf_counter() - started)
    return output


def segment_key(start: float, end: float, text: str) -> str:
    normalized = " ".join(text.lower().split())
    return f"{round(start, 1)}:{round(end, 1)}:{normalized}"


def next_segment_id() -> str:
    global next_segment_num
    with transcript_lock:
        next_segment_num += 1
        return f"seg_{next_segment_num:06d}"


def next_chunk_id() -> int:
    global next_window_chunk_id
    with transcript_lock:
        chunk_id = next_window_chunk_id
        next_window_chunk_id += 1
        return chunk_id


def transcript_event(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "transcript",
        "id": entry["id"],
        "segment_id": entry["id"],
        "chunk_id": entry["chunk_id"],
        "start": entry["start"],
        "end": entry["end"],
        "time": format_time(entry["start"]),
        "speaker": entry["speaker"],
        "text": entry["text"],
        "final": True,
        "speaker_pending": entry.get("speaker_pending", False),
    }


def append_transcript_entry(
    entry: dict[str, Any],
    *,
    ui_queue: asyncio.Queue,
    save_queue: asyncio.Queue,
    rag_queue: asyncio.Queue,
) -> bool:
    key = segment_key(entry["start"], entry["end"], entry["text"])
    with transcript_lock:
        if key in segment_keys:
            logger.info("duplicate transcript skipped chunk_id=%s segment_id=%s", entry.get("chunk_id"), entry.get("id"))
            return False
        segment_keys.add(key)
        transcript_log.append(entry)
        snapshot_len = len(transcript_log)
    ui_queue.put_nowait(transcript_event(entry))
    save_queue.put_nowait({"type": "transcript", "entry": dict(entry)})
    rag_queue.put_nowait(dict(entry))
    warn_worker_queues(ui_queue=ui_queue, save_queue=save_queue, rag_queue=rag_queue)
    logger.info(
        "transcript queued segment_id=%s chunk_id=%s segments=%d diarization_pending=%s %s",
        entry["id"],
        entry["chunk_id"],
        snapshot_len,
        entry.get("speaker_pending"),
        format_entry(entry),
    )
    return True


def warn_worker_queues(*, ui_queue: asyncio.Queue, save_queue: asyncio.Queue, rag_queue: asyncio.Queue) -> None:
    for name, worker_queue in (("ui", ui_queue), ("save", save_queue), ("rag", rag_queue)):
        qsize = worker_queue.qsize()
        if qsize >= WORKER_QUEUE_WARNING:
            logger.warning("%s queue high queue_size=%d", name, qsize)


def transcript_snapshot() -> list[dict[str, Any]]:
    with transcript_lock:
        return [dict(entry) for entry in transcript_log]


def rewrite_transcript_file(path: Path) -> None:
    with transcript_lock:
        lines = [format_entry(entry) for entry in transcript_log]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def speaker_label(raw: str) -> str:
    if raw not in speaker_names:
        speaker_names[raw] = f"Speaker {chr(65 + len(speaker_names))}"
    return speaker_names[raw]


def diarization_options() -> dict[str, int]:
    if DIARIZATION_NUM_SPEAKERS:
        return {"num_speakers": int(DIARIZATION_NUM_SPEAKERS)}
    return {"min_speakers": DIARIZATION_MIN_SPEAKERS, "max_speakers": DIARIZATION_MAX_SPEAKERS}


def full_session_audio() -> np.ndarray:
    with session_audio_lock:
        blocks = [block.copy() for block in session_audio_blocks]
    if not blocks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(blocks).astype(np.float32)


def run_diarization(audio: np.ndarray) -> list[dict[str, Any]]:
    pipeline = get_diarizer()
    if pipeline is None or len(audio) < SAMPLE_RATE:
        return []
    started = time.perf_counter()
    audio_dict = {"waveform": torch.from_numpy(audio).unsqueeze(0).float(), "sample_rate": SAMPLE_RATE}
    result = pipeline(audio_dict, **diarization_options())
    annotation = getattr(result, "exclusive_speaker_diarization", getattr(result, "speaker_diarization", result))
    turns = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda item: (item["start"], item["end"]))
    raw_map: dict[str, str] = {}
    normalized = []
    for turn in turns:
        raw = turn["speaker"]
        if raw not in raw_map:
            raw_map[raw] = f"SPEAKER_{len(raw_map):02d}"
        normalized.append({**turn, "speaker": raw_map[raw]})
    logger.info("diarization_done turns=%d diarization_time=%.3fs", len(normalized), time.perf_counter() - started)
    return normalized


def best_speaker(start: float, end: float, diarized: list[dict[str, Any]]) -> str:
    best_raw = ""
    best_overlap = 0.0
    for turn in diarized:
        overlap = max(0.0, min(end, turn["end"]) - max(start, turn["start"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_raw = turn["speaker"]
    return speaker_label(best_raw) if best_raw else "Speaker Unknown"


def apply_diarization(diarized: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not diarized:
        return []
    updates: list[dict[str, str]] = []
    with transcript_lock:
        speaker_names.clear()
        for entry in transcript_log:
            speaker = best_speaker(entry["start"], entry["end"], diarized)
            if speaker != entry.get("speaker"):
                entry["speaker"] = speaker
                entry["speaker_pending"] = False
                updates.append({"segment_id": entry["id"], "speaker": speaker})
            elif speaker != "Speaker Unknown":
                entry["speaker_pending"] = False
    if updates:
        rewrite_transcript_file(transcript_final_file)
    return updates


async def send_safe(send_json, payload: dict[str, Any]) -> None:
    try:
        await send_json(payload)
    except Exception:
        logger.exception("Failed to send websocket payload")


async def ui_worker(send_json, ui_queue: asyncio.Queue) -> None:
    while True:
        payload = await ui_queue.get()
        if payload is None:
            ui_queue.task_done()
            break
        started = time.perf_counter()
        await send_safe(send_json, payload)
        logger.info(
            "streamed type=%s segment_id=%s websocket_send_time=%.3fs ui_queue=%d",
            payload.get("type"),
            payload.get("segment_id") or payload.get("id"),
            time.perf_counter() - started,
            ui_queue.qsize(),
        )
        ui_queue.task_done()


async def save_worker(save_queue: asyncio.Queue) -> None:
    while True:
        item = await save_queue.get()
        if item is None:
            save_queue.task_done()
            break
        started = time.perf_counter()
        try:
            if item["type"] == "transcript":
                entry = item["entry"]
                with transcript_file.open("a", encoding="utf-8") as handle:
                    handle.write(format_entry(entry) + "\n")
                with transcript_jsonl_file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"type": "transcript", **entry}, ensure_ascii=False) + "\n")
                logger.info(
                    "saved segment_id=%s chunk_id=%s save_time=%.3fs save_queue=%d",
                    entry["id"],
                    entry["chunk_id"],
                    time.perf_counter() - started,
                    save_queue.qsize(),
                )
                increment_metric("saved_segments")
            elif item["type"] == "speaker_update":
                with transcript_jsonl_file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                logger.info("saved speaker_update segment_id=%s save_time=%.3fs", item.get("segment_id"), time.perf_counter() - started)
                increment_metric("speaker_updates")
        except Exception:
            logger.exception("Save worker failed")
        finally:
            save_queue.task_done()


async def rag_index_worker(rag_queue: asyncio.Queue, ui_queue: asyncio.Queue) -> None:
    while True:
        entry = await rag_queue.get()
        if entry is None:
            rag_queue.task_done()
            break
        started = time.perf_counter()
        try:
            ui_queue.put_nowait({"type": "status", "text": "Indexing"})
            await asyncio.get_running_loop().run_in_executor(None, rag_engine.add_segment, entry)
            update_metric("last_rag_time", time.perf_counter() - started)
            increment_metric("rag_indexed_segments")
            logger.info(
                "rag indexed segment_id=%s chunk_id=%s rag_indexing_time=%.3fs rag_queue=%d",
                entry.get("id"),
                entry.get("chunk_id"),
                time.perf_counter() - started,
                rag_queue.qsize(),
            )
        except Exception:
            logger.exception("RAG worker failed")
        finally:
            rag_queue.task_done()


async def diarization_worker(ui_queue: asyncio.Queue, save_queue: asyncio.Queue, rag_queue: asyncio.Queue, reason: str) -> None:
    if not HF_TOKEN:
        ui_queue.put_nowait({"type": "error", "text": "HF_TOKEN is missing. Speaker diarization is disabled."})
        return
    try:
        ui_queue.put_nowait({"type": "status", "text": "Diarizing"})
        logger.info("diarization_pending reason=%s", reason)
        started = time.perf_counter()
        audio = full_session_audio()
        diarized = await asyncio.get_running_loop().run_in_executor(None, run_diarization, audio)
        updates = apply_diarization(diarized)
        count = len({turn["speaker"] for turn in diarized})
        logger.info("diarization completed reason=%s speakers=%d updates=%d", reason, count, len(updates))
        update_metric("last_diarization_time", time.perf_counter() - started)
        ui_queue.put_nowait({"type": "status", "text": f"Diarization found {count} speaker(s)"})
        for update in updates:
            payload = {"type": "speaker_update", **update}
            ui_queue.put_nowait(payload)
            save_queue.put_nowait(payload)
        if updates:
            rag_engine.clear()
            for entry in transcript_snapshot():
                rag_queue.put_nowait(dict(entry))
    except Exception as exc:
        logger.exception("Diarization failed")
        ui_queue.put_nowait({"type": "error", "text": f"Diarization failed: {exc}"})


async def audio_processor(ui_queue: asyncio.Queue, save_queue: asyncio.Queue, rag_queue: asyncio.Queue, config: RuntimeAudioConfig) -> None:
    chunk_samples = int(AUDIO_CHUNK_SECONDS * SAMPLE_RATE)
    overlap_samples = int(AUDIO_OVERLAP_SECONDS * SAMPLE_RATE)
    step_samples = max(1, chunk_samples - overlap_samples)
    buffer = np.zeros(0, dtype=np.float32)
    buffer_start_sample = 0
    next_diarization_at = DIARIZATION_INTERVAL_SECONDS
    diarization_task: asyncio.Task | None = None
    last_block_id = -1

    logger.info("audio processor started chunk_samples=%d overlap_samples=%d", chunk_samples, overlap_samples)

    while not stop_recording.is_set() or not audio_queue.empty() or len(buffer) >= SAMPLE_RATE:
        try:
            block = await asyncio.get_running_loop().run_in_executor(None, audio_queue.get, True, 0.2)
            if last_block_id >= 0 and block.block_id != last_block_id + 1:
                logger.error("missing audio block expected=%d received=%d", last_block_id + 1, block.block_id)
            last_block_id = block.block_id
            delay = time.time() - block.captured_at
            ui_queue.put_nowait({"type": "audio_level", "source": block.source, "rms": block.rms, "peak": block.peak})
            logger.info(
                "chunk queued block_id=%d source=%s start_sample=%d samples=%d rms=%.6f peak=%.6f queue_size=%d processing_delay=%.3fs",
                block.block_id,
                block.source,
                block.start_sample,
                len(block.audio),
                block.rms,
                block.peak,
                audio_queue.qsize(),
                delay,
            )
            with session_audio_lock:
                session_audio_blocks.append(block.audio.copy())
            if len(buffer) == 0:
                buffer_start_sample = block.start_sample
            buffer = np.concatenate([buffer, block.audio])
            audio_queue.task_done()
        except queue.Empty:
            if stop_recording.is_set() and len(buffer) < chunk_samples:
                break
            await asyncio.sleep(0.02)
            continue

        while len(buffer) >= chunk_samples:
            chunk_id = next_chunk_id()
            window = buffer[:chunk_samples].copy()
            window_start = buffer_start_sample / SAMPLE_RATE
            clear_silence, rms, peak, vad_reason = vad_decision(window, config)
            if 0 < rms < LOW_AUDIO_WARNING_THRESHOLD:
                ui_queue.put_nowait({"type": "warning", "text": "Input audio level is too low"})
            if clear_silence:
                increment_metric("skipped_chunks")
                logger.info(
                    "chunk skipped chunk_id=%d vad_mode=%s vad_decision=skip reason=%s start=%.2f rms=%.6f peak=%.6f queue_size=%d",
                    chunk_id,
                    config.vad_mode,
                    vad_reason,
                    window_start,
                    rms,
                    peak,
                    audio_queue.qsize(),
                )
            else:
                ui_queue.put_nowait({"type": "status", "text": "Transcribing"})
                logger.info(
                    "chunk processed chunk_id=%d vad_mode=%s vad_decision=transcribe reason=%s start=%.2f rms=%.6f peak=%.6f queue_size=%d",
                    chunk_id,
                    config.vad_mode,
                    vad_reason,
                    window_start,
                    rms,
                    peak,
                    audio_queue.qsize(),
                )
                try:
                    started = time.perf_counter()
                    segments = await asyncio.get_running_loop().run_in_executor(None, transcribe_audio_window, window)
                    transcription_time = time.perf_counter() - started
                    for segment in segments:
                        absolute_start = window_start + segment["start"]
                        absolute_end = window_start + segment["end"]
                        entry = {
                            "id": next_segment_id(),
                            "chunk_id": chunk_id,
                            "start": absolute_start,
                            "end": absolute_end,
                            "time": absolute_start,
                            "speaker": "Speaker Unknown",
                            "text": segment["text"],
                            "final": True,
                            "speaker_pending": True,
                        }
                        append_transcript_entry(entry, ui_queue=ui_queue, save_queue=save_queue, rag_queue=rag_queue)
                    logger.info(
                        "transcribed chunk_id=%d segment_count=%d transcription_time=%.3fs total_delay=%.3fs",
                        chunk_id,
                        len(segments),
                        transcription_time,
                        time.time() - (session_started_at or time.time()) - window_start,
                    )
                    increment_metric("transcribed_chunks")
                except Exception as exc:
                    logger.exception("Transcription failed")
                    ui_queue.put_nowait({"type": "error", "text": f"Transcription failed: {exc}"})

            buffer = buffer[step_samples:]
            buffer_start_sample += step_samples

        elapsed = time.time() - (session_started_at or time.time())
        if elapsed >= next_diarization_at and (diarization_task is None or diarization_task.done()):
            diarization_task = asyncio.create_task(diarization_worker(ui_queue, save_queue, rag_queue, "live"))
            next_diarization_at = elapsed + DIARIZATION_INTERVAL_SECONDS

    if len(buffer) >= SAMPLE_RATE and not vad_decision(buffer, config)[0]:
        chunk_id = next_chunk_id()
        window_start = buffer_start_sample / SAMPLE_RATE
        logger.info("final partial chunk processed chunk_id=%d start=%.2f samples=%d", chunk_id, window_start, len(buffer))
        segments = await asyncio.get_running_loop().run_in_executor(None, transcribe_audio_window, buffer.copy())
        for segment in segments:
            absolute_start = window_start + segment["start"]
            entry = {
                "id": next_segment_id(),
                "chunk_id": chunk_id,
                "start": absolute_start,
                "end": window_start + segment["end"],
                "time": absolute_start,
                "speaker": "Speaker Unknown",
                "text": segment["text"],
                "final": True,
                "speaker_pending": True,
            }
            append_transcript_entry(entry, ui_queue=ui_queue, save_queue=save_queue, rag_queue=rag_queue)

    if diarization_task and not diarization_task.done():
        logger.info("leaving live diarization running while final pass starts when possible")
    await diarization_worker(ui_queue, save_queue, rag_queue, "final")
    logger.info("audio processor stopped")


def open_mic_stream(config: RuntimeAudioConfig):
    device = config.mic_device
    return sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=device,
        blocksize=int(SAMPLE_RATE * 0.1),
        callback=audio_callback_factory("mic", config),
    )


class SoundcardLoopbackStream:
    def __init__(self, config: RuntimeAudioConfig):
        import soundcard as sc

        self.config = config
        self.sc = sc
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="system-loopback-capture", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)

    def close(self) -> None:
        self.stop()

    def _run(self) -> None:
        frames = max(1, int(SAMPLE_RATE * 0.1))
        try:
            speaker = self.sc.default_speaker()
            audio_logger.info("soundcard loopback speaker=%s sample_rate=%d frames=%d", speaker, SAMPLE_RATE, frames)
            with speaker.recorder(samplerate=SAMPLE_RATE, channels=[0, 1]) as recorder:
                while not self.stop_event.is_set() and not stop_recording.is_set():
                    data = recorder.record(numframes=frames)
                    enqueue_audio_block("system", data, None, self.config)
        except Exception:
            logger.exception("Soundcard loopback capture failed")


def stereo_mix_device() -> int | None:
    for device in list_audio_devices():
        name = device["name"].lower()
        if device["max_input_channels"] > 0 and ("stereo mix" in name or "what u hear" in name or "loopback" in name):
            return int(device["index"])
    return None


def open_system_stream(config: RuntimeAudioConfig):
    try:
        return SoundcardLoopbackStream(config)
    except Exception as exc:
        logger.warning("soundcard loopback unavailable: %s", exc)
    device = config.system_device if config.system_device is not None else stereo_mix_device()
    if device is None:
        raise RuntimeError(
            "System audio capture unavailable: install soundcard (`pip install soundcard`) "
            "or enable/select Windows Stereo Mix."
        )
    device_info = sd.query_devices(device)
    channels = max(1, min(2, int(device_info.get("max_input_channels", 1))))
    audio_logger.info("using system fallback input device=%s name=%s channels=%d", device, device_info.get("name"), channels)
    return sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=channels,
        dtype="float32",
        device=device,
        blocksize=int(SAMPLE_RATE * 0.1),
        callback=audio_callback_factory("system", config),
    )


def open_audio_streams(config: RuntimeAudioConfig):
    streams = []
    if config.source in {"mic", "mixed", "debug"}:
        streams.append(open_mic_stream(config))
    if config.source in {"system", "mixed"}:
        streams.append(open_system_stream(config))
    return streams


async def recording_loop(ui_queue: asyncio.Queue, save_queue: asyncio.Queue, rag_queue: asyncio.Queue, config: RuntimeAudioConfig) -> None:
    try:
        log_device_table()
        try:
            streams = open_audio_streams(config)
        except Exception as exc:
            logger.exception("Audio source open failed")
            ui_queue.put_nowait({"type": "error", "text": f"{exc} Available devices are listed in logs/audio_debug.log."})
            return
        for stream in streams:
            stream.start()
        try:
            ui_queue.put_nowait(
                {
                    "type": "status",
                    "text": f"Listening ({config.source}, VAD {config.vad_mode}, force={config.force_transcribe})",
                }
            )
            await audio_processor(ui_queue, save_queue, rag_queue)
        finally:
            for stream in streams:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    logger.exception("Failed to close audio stream")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("Recording failed")
        ui_queue.put_nowait({"type": "error", "text": f"Recording failed: {exc}"})


def answer_question(question: str) -> str:
    started = time.perf_counter()
    answer = rag_engine.answer(question)
    update_metric("last_qa_time", time.perf_counter() - started)
    return answer


def generate_summary() -> str:
    started = time.perf_counter()
    summary = rag_engine.summarize()
    summary_file.write_text(summary, encoding="utf-8")
    update_metric("last_summary_time", time.perf_counter() - started)
    return summary


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    send_lock = asyncio.Lock()
    recording_task: asyncio.Task | None = None
    worker_tasks: list[asyncio.Task] = []
    ui_queue: asyncio.Queue = asyncio.Queue()
    save_queue: asyncio.Queue = asyncio.Queue()
    rag_queue: asyncio.Queue = asyncio.Queue()

    async def send_json(payload: dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def stop_workers() -> None:
        await save_queue.join()
        await rag_queue.join()
        await ui_queue.join()
        for worker_queue in (save_queue, rag_queue, ui_queue):
            await worker_queue.put(None)
        await asyncio.gather(*worker_tasks, return_exceptions=True)

    try:
        worker_tasks = [
            asyncio.create_task(ui_worker(send_json, ui_queue)),
            asyncio.create_task(save_worker(save_queue)),
            asyncio.create_task(rag_index_worker(rag_queue, ui_queue)),
        ]
        while True:
            message = await websocket.receive_json()
            action = message.get("action")

            if action == "start":
                if recording_task and not recording_task.done():
                    ui_queue.put_nowait({"type": "status", "text": "Recording already running"})
                    continue
                reset_session()
                config = build_audio_config(message)
                if not HF_TOKEN:
                    ui_queue.put_nowait({"type": "error", "text": "HF_TOKEN is missing. Speaker diarization is disabled."})
                recording_task = asyncio.create_task(recording_loop(ui_queue, save_queue, rag_queue, config))

            elif action == "stop":
                stop_recording.set()
                if recording_task:
                    await asyncio.wait_for(recording_task, timeout=90)
                ui_queue.put_nowait({"type": "status", "text": "Recording stopped"})

            elif action == "question":
                question = str(message.get("text", "")).strip()
                if not question:
                    ui_queue.put_nowait({"type": "answer", "text": "Question:\n\nAnswer: This was not mentioned in the meeting.\nEvidence: None"})
                    continue
                answer = await asyncio.get_running_loop().run_in_executor(None, answer_question, question)
                ui_queue.put_nowait({"type": "answer", "text": answer})

            elif action == "summarize":
                summary = await asyncio.get_running_loop().run_in_executor(None, generate_summary)
                ui_queue.put_nowait({"type": "summary", "text": summary})

            elif action == "list_devices":
                ui_queue.put_nowait({"type": "devices", "devices": list_audio_devices()})

    except WebSocketDisconnect:
        stop_recording.set()
        if recording_task:
            recording_task.cancel()
        logger.info("WebSocket disconnected")
    finally:
        for task in worker_tasks:
            if not task.done():
                task.cancel()


@app.get("/")
async def get_frontend():
    html_path = ROOT / "frontend" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>frontend/index.html not found</h2>")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "whisper_model": WHISPER_MODEL,
        "audio_queue_size": audio_queue.qsize(),
        "segments": len(transcript_snapshot()),
        "metrics": metrics_snapshot(),
        "diarization_enabled": bool(HF_TOKEN),
        "rag": rag_engine.status(),
        "audio": {
            "source": AUDIO_SOURCE,
            "vad_mode": VAD_MODE,
            "force_transcribe": FORCE_TRANSCRIBE,
            "input_gain": AUDIO_INPUT_GAIN,
            "min_rms_threshold": MIN_RMS_THRESHOLD,
            "low_audio_warning_threshold": LOW_AUDIO_WARNING_THRESHOLD,
            "system_loopback_available": system_capture_available(),
        },
        "outputs": {
            "transcript": str(transcript_file),
            "transcript_final": str(transcript_final_file),
            "transcript_jsonl": str(transcript_jsonl_file),
            "summary": str(summary_file),
            "log": str(LOG_DIR / "app.log"),
            "audio_debug_log": str(audio_debug_file),
        },
    }


@app.get("/devices")
async def devices():
    return {
        "default": {"input": sd.default.device[0], "output": sd.default.device[1]},
        "system_loopback_available": system_capture_available(),
        "devices": list_audio_devices(),
    }


@app.get("/metrics")
async def metrics():
    return metrics_snapshot()


@app.get("/transcript")
async def get_transcript():
    segments = transcript_snapshot()
    return {"count": len(segments), "segments": segments, "file": str(transcript_file)}


@app.post("/ask")
async def ask(request: Request):
    payload = await request.json()
    question = str(payload.get("question", "")).strip()
    return {"question": question, "answer": answer_question(question)}


@app.post("/summary")
async def summary():
    summary_text = generate_summary()
    return {"summary": summary_text, "file": str(summary_file)}


@app.get("/api/state")
async def api_state():
    state = await health()
    return {
        **state,
        "chunk_seconds": AUDIO_CHUNK_SECONDS,
        "overlap_seconds": AUDIO_OVERLAP_SECONDS,
        "audio_device": AUDIO_DEVICE_INDEX,
    }


@app.get("/api/health")
async def api_health():
    return await health()


@app.get("/api/transcript")
async def api_transcript():
    return await get_transcript()


@app.get("/api/speakers")
async def api_speakers():
    counts: dict[str, int] = {}
    for entry in transcript_snapshot():
        counts[entry["speaker"]] = counts.get(entry["speaker"], 0) + 1
    return {"speakers": counts}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=int(os.getenv("PORT", "8000")), reload=False)
