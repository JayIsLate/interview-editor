"""
Silence detector module for finding audio gaps.
"""

from pydub import AudioSegment
from pydub.silence import detect_silence
from typing import List, Tuple
import tempfile
import os


class SilenceDetector:
    """Detects silence and pauses in audio."""

    def __init__(self, min_silence_len: int = 800, silence_thresh: int = -40):
        """
        Initialize the silence detector.

        Args:
            min_silence_len: Minimum silence length in milliseconds
            silence_thresh: Silence threshold in dBFS
        """
        self.min_silence_len = min_silence_len
        self.silence_thresh = silence_thresh

    def load_audio(self, file_path: str) -> AudioSegment:
        """
        Load audio from a file (video or audio format).

        Args:
            file_path: Path to the file

        Returns:
            AudioSegment object
        """
        return AudioSegment.from_file(file_path)

    def detect_silences(self, audio: AudioSegment) -> List[Tuple[float, float]]:
        """
        Detect silence segments in audio.

        Args:
            audio: AudioSegment to analyze

        Returns:
            List of (start_sec, end_sec) tuples for each silence segment
        """
        # detect_silence returns list of [start_ms, end_ms] for each silence
        silence_ranges = detect_silence(
            audio,
            min_silence_len=self.min_silence_len,
            silence_thresh=self.silence_thresh
        )

        # Convert to seconds
        return [(start / 1000.0, end / 1000.0) for start, end in silence_ranges]

    def detect_silences_from_file(self, file_path: str) -> List[Tuple[float, float]]:
        """
        Detect silences directly from a file.

        Args:
            file_path: Path to audio/video file

        Returns:
            List of (start_sec, end_sec) tuples
        """
        audio = self.load_audio(file_path)
        return self.detect_silences(audio)

    def get_silence_cuts(self, file_path: str) -> List[dict]:
        """
        Get silence segments as cut objects.

        Args:
            file_path: Path to audio/video file

        Returns:
            List of cut dictionaries with start, end, reason, and type
        """
        silences = self.detect_silences_from_file(file_path)

        cuts = []
        for start, end in silences:
            duration = end - start
            cuts.append({
                "start": start,
                "end": end,
                "duration": duration,
                "reason": f"silence ({duration:.1f}s)",
                "type": "silence",
                "approved": None,  # None = pending, True = approved, False = rejected
            })

        return cuts

    def get_silence_cuts_and_duration(self, file_path: str):
        """
        Get silence cuts and audio duration in a single pass (one load).

        Args:
            file_path: Path to audio/video file

        Returns:
            Tuple of (list of cut dicts, duration in seconds)
        """
        audio = self.load_audio(file_path)
        silences = self.detect_silences(audio)
        duration = len(audio) / 1000.0

        cuts = []
        for start, end in silences:
            dur = end - start
            cuts.append({
                "start": start,
                "end": end,
                "duration": dur,
                "reason": f"silence ({dur:.1f}s)",
                "type": "silence",
                "approved": None,
            })

        return cuts, duration

    def get_audio_duration(self, file_path: str) -> float:
        """
        Get the duration of an audio/video file.

        Args:
            file_path: Path to file

        Returns:
            Duration in seconds
        """
        audio = self.load_audio(file_path)
        return len(audio) / 1000.0

    def extract_audio_segment(
        self, file_path: str, start: float, end: float
    ) -> AudioSegment:
        """
        Extract a segment of audio.

        Args:
            file_path: Path to audio/video file
            start: Start time in seconds
            end: End time in seconds

        Returns:
            AudioSegment of the specified range
        """
        audio = self.load_audio(file_path)
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        return audio[start_ms:end_ms]

    def export_audio_segment(
        self, file_path: str, start: float, end: float, output_format: str = "wav"
    ) -> bytes:
        """
        Export a segment of audio as bytes.

        Args:
            file_path: Path to audio/video file
            start: Start time in seconds
            end: End time in seconds
            output_format: Output format (wav, mp3, etc.)

        Returns:
            Audio data as bytes
        """
        segment = self.extract_audio_segment(file_path, start, end)

        # Export to bytes
        temp_file = tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False)
        temp_file.close()

        try:
            segment.export(temp_file.name, format=output_format)
            with open(temp_file.name, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
