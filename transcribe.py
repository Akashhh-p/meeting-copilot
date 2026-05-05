#!/usr/bin/env python3
"""
Meeting Copilot - Real-time meeting transcription with transcripts saved to files.
Records audio from microphone and transcribes with timestamps.
"""

import os
import sys
import signal
import json
import numpy as np
import sounddevice as sd
from datetime import datetime
from pathlib import Path
from faster_whisper import WhisperModel
import logging
import torch
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CHUNK_DURATION = 5  # seconds - captures multiple sentences per chunk
SAMPLE_RATE = 16000  # Hz
CHANNELS = 1
ENERGY_THRESHOLD = 0.02  # Minimum energy to consider as speech
LOW_AUDIO_WARNING_THRESHOLD = float(os.getenv("LOW_AUDIO_WARNING_THRESHOLD", "0.003"))
AUDIO_INPUT_GAIN = float(os.getenv("AUDIO_INPUT_GAIN", "1.5"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "3"))
FINAL_DIARIZATION_TIMEOUT = 180  # seconds - final full-session diarization pass
DIARIZATION_NUM_SPEAKERS = os.getenv("DIARIZATION_NUM_SPEAKERS")
DIARIZATION_MIN_SPEAKERS = int(os.getenv("DIARIZATION_MIN_SPEAKERS", "1"))
DIARIZATION_MAX_SPEAKERS = int(os.getenv("DIARIZATION_MAX_SPEAKERS", "4"))
DIARIZATION_SHORT_TURN_SECONDS = float(os.getenv("DIARIZATION_SHORT_TURN_SECONDS", "0.6"))
DIARIZATION_MERGE_GAP_SECONDS = float(os.getenv("DIARIZATION_MERGE_GAP_SECONDS", "0.4"))
DIARIZATION_MODELS = (
    "pyannote/speaker-diarization-community-1",
    "pyannote/speaker-diarization-3.0",
)

# Global flag for graceful shutdown
stop_recording = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global stop_recording
    logger.info("\n\nSaving and shutting down...")
    stop_recording = True

def list_audio_devices():
    """List all available audio input devices."""
    print("\n" + "="*60)
    print("AVAILABLE AUDIO DEVICES:")
    print("="*60)
    devices = sd.query_devices()
    default_in, default_out = sd.default.device
    for i, device in enumerate(devices):
        markers = []
        if i == default_in:
            markers.append("DEFAULT INPUT")
        if i == default_out:
            markers.append("DEFAULT OUTPUT")
        marker = f" [{' / '.join(markers)}]" if markers else ""
        print(
            f"{i}: {device['name']} | input={device['max_input_channels']} "
            f"output={device['max_output_channels']} rate={device['default_samplerate']}{marker}"
        )
    print("="*60 + "\n")


def debug_audio(device=None, seconds=30):
    """Print per-second RMS/peak/queue-style audio diagnostics without loading ML models."""
    print("\nAUDIO DEBUG")
    print("=" * 60)
    list_audio_devices()
    print(f"Selected device: {device if device is not None else 'default'}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Channels: {CHANNELS}")
    print(f"Input gain: {AUDIO_INPUT_GAIN}")
    print("Press Ctrl+C to stop.\n")
    frames = int(SAMPLE_RATE)
    try:
        for second in range(1, seconds + 1):
            audio = sd.rec(
                frames,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                device=device,
                dtype=np.float32,
                blocksize=4096,
            )
            sd.wait()
            mono = audio.mean(axis=1) if audio.ndim == 2 else audio.reshape(-1)
            mono = np.clip(mono * AUDIO_INPUT_GAIN, -1.0, 1.0)
            rms = float(np.sqrt(np.mean(mono**2))) if len(mono) else 0.0
            peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
            speech = rms > ENERGY_THRESHOLD or peak > 0.01
            low = 0 < rms < LOW_AUDIO_WARNING_THRESHOLD
            print(
                f"{second:03d}s rms={rms:.6f} peak={peak:.6f} "
                f"speech_detected={speech} low_audio={low} queued=1 queue_size=0"
            )
    except KeyboardInterrupt:
        print("\nAudio debug stopped.")

def select_audio_device():
    """Allow user to select audio device."""
    list_audio_devices()
    try:
        device_id = input("Enter device number (press Enter for default): ").strip()
        if device_id == "":
            device_id = None
        else:
            device_id = int(device_id)
        return device_id
    except ValueError:
        logger.warning("Invalid input, using default device")
        return None

def detect_speech(audio_chunk):
    """Detect if audio chunk contains speech."""
    # Calculate RMS energy
    rms_energy = np.sqrt(np.mean(audio_chunk**2))
    
    # Simple heuristic: speech has moderate energy.
    has_speech = rms_energy > ENERGY_THRESHOLD
    
    return has_speech, rms_energy


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

class MeetingTranscriber:
    def __init__(self, device=None, model_size="base", hf_token=None):
        """
        Initialize the transcriber.
        
        Args:
            device: Audio device ID (None for default)
            model_size: Whisper model size (tiny, base, small, medium, large)
            hf_token: HuggingFace token for speaker diarization
        """
        logger.info(f"Loading Whisper model ({model_size})...")
        self.model = WhisperModel(
            model_size,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )
        
        # Load diarization pipeline
        logger.info("Loading speaker diarization pipeline...")
        self.diarization_pipeline = None
        self.speaker_map = {}
        self.speaker_counter = 0
        
        try:
            if hf_token:
                for diarization_model in DIARIZATION_MODELS:
                    try:
                        from pyannote.audio import Pipeline

                        self.diarization_pipeline = Pipeline.from_pretrained(
                            diarization_model,
                            token=hf_token
                        )
                        if self.diarization_pipeline is not None:
                            logger.info(f"Speaker diarization enabled ({diarization_model})")
                            break
                    except Exception as e:
                        logger.warning(f"Could not load {diarization_model}: {e}")

                if self.diarization_pipeline is None:
                    raise RuntimeError("No diarization model could be loaded")
            else:
                logger.warning("No HuggingFace token provided. Speaker diarization disabled.")
                logger.warning("To enable: set HF_TOKEN environment variable or pass --hf-token")
        except Exception as e:
            logger.warning(f"Failed to load diarization pipeline: {e}")
            logger.warning("Continuing without speaker diarization")
            self.diarization_pipeline = None
        
        self.device = device
        self.session_start = datetime.now()
        self.transcripts = []
        self.audio_chunks = []
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        # Create output directory
        self.output_dir = Path("meeting_transcripts")
        self.output_dir.mkdir(exist_ok=True)
        
        # Session file paths
        session_name = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / session_name
        self.session_dir.mkdir(exist_ok=True)
        
        self.transcript_file = self.session_dir / "transcript.txt"
        self.json_file = self.session_dir / "transcript.json"
        
        logger.info(f"Session directory: {self.session_dir}")
        logger.info(f"Transcripts will be saved to: {self.transcript_file}")
    
    def _get_speaker_label(self, speaker_idx) -> str:
        """Get or create a consistent label for a speaker."""
        if speaker_idx not in self.speaker_map:
            self.speaker_map[speaker_idx] = f"Speaker {chr(65 + self.speaker_counter)}"
            self.speaker_counter += 1
        return self.speaker_map[speaker_idx]
    
    def _diarize_audio(self, audio_chunk: np.ndarray) -> dict:
        """
        Run diarization on audio chunk with timeout protection.
        Returns a mapping of timestamps to speakers.
        """
        if not self.diarization_pipeline:
            return {}
        
        try:
            # Prepare audio in pyannote format
            audio_dict = {
                "waveform": torch.from_numpy(audio_chunk).unsqueeze(0).float(),
                "sample_rate": SAMPLE_RATE
            }
            
            # Run diarization. Defaulting min speakers to 1 avoids forcing
            # one-person recordings to be split into multiple speakers.
            diarization = self.diarization_pipeline(
                audio_dict,
                **self._diarization_speaker_options()
            )
            
            # Extract speaker segments
            speaker_map = {}
            for segment, track, speaker_id in diarization.itertracks(yield_label=True):
                speaker_map[(segment.start, segment.end)] = speaker_id
            
            return self._normalize_speaker_map(self._stabilize_speaker_map(speaker_map))
            
        except Exception as e:
            logger.warning(f"Diarization error (non-fatal): {e}")
            return {}

    def _diarization_speaker_options(self) -> dict:
        if DIARIZATION_NUM_SPEAKERS:
            return {"num_speakers": int(DIARIZATION_NUM_SPEAKERS)}
        return {
            "min_speakers": DIARIZATION_MIN_SPEAKERS,
            "max_speakers": max(DIARIZATION_MIN_SPEAKERS, DIARIZATION_MAX_SPEAKERS),
        }

    def _stabilize_speaker_map(self, speaker_map: dict) -> dict:
        if not speaker_map:
            return {}

        segments = [
            {"start": start, "end": end, "speaker": speaker}
            for (start, end), speaker in speaker_map.items()
        ]
        segments.sort(key=lambda item: (item["start"], item["end"]))

        for index in range(1, len(segments) - 1):
            current = segments[index]
            previous = segments[index - 1]
            following = segments[index + 1]
            duration = current["end"] - current["start"]
            if (
                duration < DIARIZATION_SHORT_TURN_SECONDS
                and previous["speaker"] == following["speaker"]
                and current["speaker"] != previous["speaker"]
            ):
                current["speaker"] = previous["speaker"]

        merged = []
        for segment in segments:
            if (
                merged
                and merged[-1]["speaker"] == segment["speaker"]
                and segment["start"] - merged[-1]["end"] <= DIARIZATION_MERGE_GAP_SECONDS
            ):
                merged[-1]["end"] = max(merged[-1]["end"], segment["end"])
            else:
                merged.append(dict(segment))

        return {
            (segment["start"], segment["end"]): segment["speaker"]
            for segment in merged
        }

    def _normalize_speaker_map(self, speaker_map: dict) -> dict:
        normalized = {}
        raw_to_stable = {}
        for (start, end), speaker in sorted(speaker_map.items()):
            if speaker not in raw_to_stable:
                raw_to_stable[speaker] = f"SPEAKER_{len(raw_to_stable):02d}"
            normalized[(start, end)] = raw_to_stable[speaker]
        return normalized
    
    def _find_speaker_for_segment(self, segment_start, segment_end, speaker_map) -> str:
        """Find the speaker that overlaps most with the given segment."""
        if not speaker_map:
            return "Unknown"
        
        best_speaker = None
        best_overlap = 0
        
        for (sp_start, sp_end), speaker_id in speaker_map.items():
            overlap_start = max(segment_start, sp_start)
            overlap_end = min(segment_end, sp_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker_id
        
        if best_speaker is not None:
            return self._get_speaker_label(best_speaker)
        return "Unknown"
    
    def _apply_speaker_labels(self, speaker_map: dict):
        """Apply diarization labels to all saved transcript segments."""
        if not speaker_map:
            return

        for chunk_data in self.transcripts:
            for segment in chunk_data["segments"]:
                speaker = self._find_speaker_for_segment(
                    segment["start"], segment["end"], speaker_map
                )
                segment["speaker"] = speaker

    def _diarize_full_session(self):
        """Run one full-session diarization pass before saving final output."""
        if not self.diarization_pipeline or not self.audio_chunks:
            return

        try:
            logger.info("Running final speaker diarization pass...")
            full_audio = np.concatenate(self.audio_chunks).astype(np.float32)
            future = self.executor.submit(self._diarize_audio, full_audio)
            speaker_map = future.result(timeout=FINAL_DIARIZATION_TIMEOUT)
            if speaker_map:
                self._apply_speaker_labels(speaker_map)
                logger.info(f"Final diarization found {len(self.speaker_map)} speaker(s)")
            else:
                logger.warning("Final diarization returned no speaker segments")
        except FuturesTimeoutError:
            logger.warning("Final diarization timeout, saving transcript without speaker labels")
        except Exception as e:
            logger.warning(f"Final diarization failed: {e}")

    def transcribe_chunk(self, audio_chunk: np.ndarray, chunk_num: int) -> dict:
        """
        Transcribe a single audio chunk with speaker diarization.
        
        Args:
            audio_chunk: Audio data as numpy array
            chunk_num: Chunk number for tracking
            
        Returns:
            Dictionary with transcription data or None if no speech
        """
        if len(audio_chunk) == 0:
            return None
        
        # Detect if chunk has speech
        has_speech, energy = detect_speech(audio_chunk)
        if not has_speech:
            logger.debug(f"Chunk {chunk_num}: No speech detected (energy: {energy:.4f})")
            return None
        
        try:
            logger.info(f"Chunk {chunk_num}: Transcribing (energy: {energy:.4f})...")
            
            # Transcribe with Whisper
            segments, info = self.model.transcribe(
                audio_chunk,
                language="en",
                beam_size=WHISPER_BEAM_SIZE,
                temperature=0,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 350},
                condition_on_previous_text=False,
            )
            
            segments_list = list(segments)
            
            if not segments_list:
                logger.debug(f"Chunk {chunk_num}: No speech recognized")
                return None
            
            # Speaker labels are assigned in a final full-session pass before saving.
            # Very short chunks often do not give pyannote enough context.
            speaker_map = {}
            
            # Process segments
            chunk_data = {
                "chunk_num": chunk_num,
                "timestamp": datetime.now().isoformat(),
                "duration": len(audio_chunk) / SAMPLE_RATE,
                "segments": []
            }
            chunk_start = (chunk_num - 1) * CHUNK_DURATION
            
            for segment in segments_list:
                text = segment.text.strip()
                if text:
                    # Get speaker label if diarization available
                    speaker = self._find_speaker_for_segment(
                        segment.start, segment.end, speaker_map
                    )
                    
                    # Format output
                    speaker_prefix = f"{speaker}: " if speaker != "Unknown" else ""
                    
                    segment_data = {
                        "start": round(chunk_start + segment.start, 2),
                        "end": round(chunk_start + segment.end, 2),
                        "speaker": speaker,
                        "text": text
                    }
                    chunk_data["segments"].append(segment_data)
                    logger.info(
                        f"  [{format_timestamp(segment_data['start'])} - "
                        f"{format_timestamp(segment_data['end'])}] {speaker_prefix}{text}"
                    )
            
            if chunk_data["segments"]:
                return chunk_data
            else:
                logger.debug(f"Chunk {chunk_num}: No text extracted from segments")
                return None
            
        except Exception as e:
            logger.error(f"Transcription error on chunk {chunk_num}: {e}")
            return None
    
    def save_transcript(self):
        """Save accumulated transcripts to files."""
        if not self.transcripts:
            logger.info("No transcriptions to save")
            return
        
        try:
            self._diarize_full_session()

            # Save as text file with speaker labels
            with open(self.transcript_file, "w") as f:
                f.write(f"MEETING TRANSCRIPT\n")
                f.write(f"Started: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if self.speaker_map:
                    f.write(f"Speakers: {', '.join(self.speaker_map.values())}\n")
                f.write("="*60 + "\n\n")
                
                for chunk_data in self.transcripts:
                    for segment in chunk_data["segments"]:
                        speaker = segment.get("speaker", "Unknown")
                        speaker_label = f"{speaker}: " if speaker != "Unknown" else ""
                        f.write(
                            f"[{format_timestamp(segment['start'])} - "
                            f"{format_timestamp(segment['end'])}] "
                            f"{speaker_label}{segment['text']}\n"
                        )
            
            logger.info(f"Transcript saved to: {self.transcript_file}")
            
            # Save as JSON for structured data
            with open(self.json_file, "w") as f:
                json.dump({
                    "session_start": self.session_start.isoformat(),
                    "session_end": datetime.now().isoformat(),
                    "speakers": list(self.speaker_map.values()),
                    "chunks": self.transcripts
                }, f, indent=2)
            
            logger.info(f"JSON transcript saved to: {self.json_file}")
            
        except Exception as e:
            logger.error(f"Error saving transcript: {e}")
    
    def run(self):
        """Start real-time transcription from microphone."""
        global stop_recording
        
        # Register signal handler for Ctrl+C
        signal.signal(signal.SIGINT, signal_handler)
        
        logger.info("="*60)
        logger.info("MEETING COPILOT - Real-time Transcription")
        logger.info("="*60)
        logger.info(f"Recording {CHUNK_DURATION}s chunks at {SAMPLE_RATE}Hz")
        logger.info("Press Ctrl+C to stop and save transcript.")
        logger.info("-"*60)
        
        chunk_num = 0
        
        while not stop_recording:
            try:
                chunk_num += 1
                logger.info(f"\n[CHUNK {chunk_num}] Recording at {datetime.now().strftime('%H:%M:%S')}...")
                
                # Record audio chunk
                try:
                    audio_chunk = sd.rec(
                        int(CHUNK_DURATION * SAMPLE_RATE),
                        samplerate=SAMPLE_RATE,
                        channels=CHANNELS,
                        device=self.device,
                        dtype=np.float32,
                        blocksize=4096
                    )
                    sd.wait()  # Wait for recording to complete
                    
                except Exception as e:
                    logger.error(f"Audio recording error: {e}")
                    logger.error("Try running with --select-device to choose a different audio input")
                    continue
                
                # Flatten if stereo was recorded
                if len(audio_chunk.shape) > 1:
                    audio_chunk = np.mean(audio_chunk, axis=1)
                
                # Normalize audio
                max_val = np.max(np.abs(audio_chunk))
                if max_val > 0:
                    audio_chunk = audio_chunk / max_val
                self.audio_chunks.append(audio_chunk.copy())
                
                # Transcribe
                result = self.transcribe_chunk(audio_chunk, chunk_num)
                
                if result:
                    self.transcripts.append(result)
                    logger.info(f"Chunk {chunk_num} transcribed successfully")
                
                logger.info("-"*60)
                
            except Exception as e:
                logger.error(f"Error during chunk {chunk_num}: {e}")
                continue
        
        # Save transcript when done
        logger.info("\nFinalizing session...")
        self.save_transcript()
        self.executor.shutdown(wait=False)
        logger.info("Session complete!")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Meeting Copilot - Real-time transcription with speaker diarization")
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"],
                       default="base", help="Whisper model size")
    parser.add_argument("--device", type=int, default=None, help="Audio device ID")
    parser.add_argument("--select-device", action="store_true", help="Interactive device selection")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace token for speaker diarization")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--debug-audio", action="store_true", help="Show live RMS/peak diagnostics without loading ML models")
    parser.add_argument("--debug-seconds", type=int, default=30, help="Seconds to run --debug-audio")
    
    args = parser.parse_args()
    
    # Get HuggingFace token from argument or environment
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    
    device = args.device
    if args.select_device:
        device = select_audio_device()

    if args.list_devices:
        list_audio_devices()
        return

    if args.debug_audio:
        debug_audio(device=device, seconds=args.debug_seconds)
        return
    
    try:
        transcriber = MeetingTranscriber(device=device, model_size=args.model, hf_token=hf_token)
        transcriber.run()
    except KeyboardInterrupt:
        logger.info("\nShutdown complete.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
