#!/usr/bin/env python3
"""
Real-time meeting transcription with speaker diarization.
Captures audio from microphone in 3-second chunks and transcribes with speaker labels.
"""

import os
import sys
import numpy as np
import sounddevice as sd
from datetime import datetime
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CHUNK_DURATION = 3  # seconds
SAMPLE_RATE = 16000  # Hz
CHANNELS = 1
DEVICE = None  # Use default device

class MeetingTranscriber:
    def __init__(self, hf_token: str = None):
        """
        Initialize the transcriber with Whisper and speaker diarization.
        
        Args:
            hf_token: HuggingFace token for accessing pyannote models
        """
        logger.info("Loading Whisper model...")
        # Use tiny model for speed, can upgrade to base/small/medium for accuracy
        self.model = WhisperModel("tiny", device="auto", compute_type="float32")
        
        logger.info("Loading speaker diarization pipeline...")
        if hf_token:
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.0",
                use_auth_token=hf_token
            )
        else:
            logger.warning("No HuggingFace token provided. Speaker diarization may not work.")
            logger.warning("Set HF_TOKEN environment variable or pass hf_token parameter.")
            self.diarization_pipeline = None
        
        self.speaker_map = {}  # Map speaker indices to labels
        self.speaker_counter = 0
        
    def _get_speaker_label(self, speaker_idx: int) -> str:
        """Get or create a consistent label for a speaker."""
        if speaker_idx not in self.speaker_map:
            self.speaker_map[speaker_idx] = f"Speaker {chr(65 + self.speaker_counter)}"
            self.speaker_counter += 1
        return self.speaker_map[speaker_idx]
    
    def transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        """
        Transcribe a single audio chunk and add speaker labels.
        
        Args:
            audio_chunk: Audio data as numpy array
            
        Returns:
            Formatted transcript with speaker labels and timestamps
        """
        if len(audio_chunk) == 0:
            return ""
        
        try:
            # Transcribe with Whisper
            segments, info = self.model.transcribe(
                audio_chunk,
                language="en",
                beam_size=5,
                temperature=0
            )
            
            transcript_parts = []
            
            # Apply speaker diarization if available
            if self.diarization_pipeline:
                try:
                    # Run diarization
                    diarization = self.diarization_pipeline(
                        {"waveform": audio_chunk, "sample_rate": SAMPLE_RATE}
                    )
                    
                    # Format output with speaker labels
                    for segment, track, speaker_idx in diarization.itertracks(yield_label=True):
                        for transcribed_segment in segments:
                            # Match transcribed segments with speaker segments
                            seg_start = transcribed_segment.start
                            seg_end = transcribed_segment.end
                            track_start = segment.start
                            track_end = segment.end
                            
                            # Simple overlap-based matching
                            if not (seg_end < track_start or seg_start > track_end):
                                speaker_label = self._get_speaker_label(int(speaker_idx[0]))
                                timestamp = f"[{seg_start:.2f}s]"
                                text = transcribed_segment.text.strip()
                                if text:
                                    transcript_parts.append(
                                        f"{timestamp} {speaker_label}: {text}"
                                    )
                        
                except Exception as e:
                    logger.warning(f"Diarization failed: {e}. Using transcription only.")
                    for segment in segments:
                        timestamp = f"[{segment.start:.2f}s]"
                        text = segment.text.strip()
                        if text:
                            transcript_parts.append(f"{timestamp} Unknown: {text}")
            else:
                # Fallback: transcription without speaker labels
                for segment in segments:
                    timestamp = f"[{segment.start:.2f}s]"
                    text = segment.text.strip()
                    if text:
                        transcript_parts.append(f"{timestamp} {text}")
            
            return "\n".join(transcript_parts)
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
    
    def run(self):
        """Start real-time transcription from microphone."""
        logger.info(f"Starting real-time transcription. Recording {CHUNK_DURATION}s chunks...")
        logger.info("Press Ctrl+C to stop.")
        logger.info("-" * 60)
        
        try:
            while True:
                # Record audio chunk
                logger.info(f"Recording chunk at {datetime.now().strftime('%H:%M:%S')}...")
                audio_chunk = sd.rec(
                    int(CHUNK_DURATION * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    device=DEVICE,
                    dtype=np.float32,
                    blocksize=4096
                )
                sd.wait()  # Wait for recording to complete
                
                # Flatten if stereo was recorded
                if len(audio_chunk.shape) > 1:
                    audio_chunk = np.mean(audio_chunk, axis=1)
                
                # Transcribe
                transcript = self.transcribe_chunk(audio_chunk)
                
                if transcript:
                    print("\n" + transcript + "\n")
                else:
                    print("[Silent or unrecognized]")
                
                logger.info("-" * 60)
                
        except KeyboardInterrupt:
            logger.info("\nTranscription stopped by user.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            sys.exit(1)


def main():
    """Main entry point."""
    # Get HuggingFace token from environment
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        logger.warning("HF_TOKEN environment variable not set.")
        logger.warning("Speaker diarization will be disabled.")
        logger.info("To enable: export HF_TOKEN='your_huggingface_token'")
    
    # Initialize and run transcriber
    transcriber = MeetingTranscriber(hf_token=hf_token)
    transcriber.run()


if __name__ == "__main__":
    main()
