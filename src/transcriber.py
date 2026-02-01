"""
Transcriber module using OpenAI Whisper API for speech-to-text.
"""

import tempfile
import os
from pydub import AudioSegment
from typing import Optional
from openai import OpenAI


class Transcriber:
    """Handles audio transcription using OpenAI Whisper API."""

    # Maximum file size for OpenAI API (25 MB)
    MAX_FILE_SIZE = 25 * 1024 * 1024

    def __init__(self, model_name: str = "base", language: Optional[str] = "en", api_key: Optional[str] = None):
        """
        Initialize the transcriber.

        Args:
            model_name: Ignored for API (kept for compatibility)
            language: Language code or None for auto-detection
            api_key: OpenAI API key (optional, will use env var if not provided)
        """
        self.model_name = model_name  # Kept for compatibility, not used with API
        self.language = language
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        """Get or create the OpenAI client."""
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()
        return self._client

    def extract_audio(self, video_path: str) -> str:
        """
        Extract audio from video file.

        Args:
            video_path: Path to the video file

        Returns:
            Path to the extracted audio file (MP3 format for smaller size)
        """
        audio = AudioSegment.from_file(video_path)

        # Create temp file for audio - use MP3 for smaller file size
        temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        temp_audio.close()

        # Export as MP3 with reasonable quality (smaller than WAV)
        audio.export(temp_audio.name, format="mp3", bitrate="128k")

        return temp_audio.name

    def _transcribe_chunk(self, audio_path: str) -> dict:
        """
        Transcribe a single audio file using the OpenAI API.

        Args:
            audio_path: Path to audio file (must be under 25 MB)

        Returns:
            API response with transcription
        """
        with open(audio_path, "rb") as audio_file:
            kwargs = {
                "model": "whisper-1",
                "file": audio_file,
                "response_format": "verbose_json",
                "timestamp_granularities": ["word", "segment"],
            }
            if self.language:
                kwargs["language"] = self.language

            response = self.client.audio.transcriptions.create(**kwargs)

        return response

    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribe audio file using OpenAI Whisper API.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcription result with word-level timestamps (compatible with local Whisper format)
        """
        file_size = os.path.getsize(audio_path)

        if file_size <= self.MAX_FILE_SIZE:
            # File is small enough, transcribe directly
            response = self._transcribe_chunk(audio_path)
            return self._convert_api_response(response)
        else:
            # File too large, need to split into chunks
            return self._transcribe_large_file(audio_path)

    def _transcribe_large_file(self, audio_path: str) -> dict:
        """
        Transcribe a large audio file by splitting it into chunks.

        Args:
            audio_path: Path to audio file

        Returns:
            Combined transcription result
        """
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)

        # Calculate chunk duration to stay under 25 MB
        # Estimate: 128kbps MP3 = ~1 MB per minute, so ~20 minutes per chunk to be safe
        chunk_duration_ms = 15 * 60 * 1000  # 15 minutes per chunk

        all_segments = []
        all_words = []
        full_text_parts = []
        time_offset = 0.0

        chunk_index = 0
        while chunk_index * chunk_duration_ms < duration_ms:
            start_ms = chunk_index * chunk_duration_ms
            end_ms = min((chunk_index + 1) * chunk_duration_ms, duration_ms)

            # Extract chunk
            chunk = audio[start_ms:end_ms]

            # Save chunk to temp file
            temp_chunk = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            temp_chunk.close()
            chunk.export(temp_chunk.name, format="mp3", bitrate="128k")

            try:
                # Transcribe chunk
                response = self._transcribe_chunk(temp_chunk.name)

                # Add text
                full_text_parts.append(response.text)

                # Adjust timestamps and add words
                if hasattr(response, 'words') and response.words:
                    for word in response.words:
                        all_words.append({
                            "word": word.word,
                            "start": word.start + time_offset,
                            "end": word.end + time_offset,
                        })

                # Adjust timestamps and add segments
                if hasattr(response, 'segments') and response.segments:
                    for segment in response.segments:
                        seg_dict = {
                            "start": segment.start + time_offset,
                            "end": segment.end + time_offset,
                            "text": segment.text,
                            "words": [],
                        }
                        # Add words to segment if available
                        if hasattr(segment, 'words') and segment.words:
                            for word in segment.words:
                                seg_dict["words"].append({
                                    "word": word.word,
                                    "start": word.start + time_offset,
                                    "end": word.end + time_offset,
                                })
                        all_segments.append(seg_dict)

            finally:
                # Clean up chunk file
                if os.path.exists(temp_chunk.name):
                    os.unlink(temp_chunk.name)

            time_offset = end_ms / 1000.0  # Convert to seconds
            chunk_index += 1

        # Combine results
        return {
            "text": " ".join(full_text_parts),
            "segments": all_segments,
            "words": all_words,
        }

    def _convert_api_response(self, response) -> dict:
        """
        Convert OpenAI API response to local Whisper-compatible format.

        Args:
            response: OpenAI API transcription response

        Returns:
            Dict matching local Whisper output format
        """
        segments = []

        # Convert segments
        if hasattr(response, 'segments') and response.segments:
            for segment in response.segments:
                seg_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "words": [],
                }
                # API doesn't nest words in segments, but we'll populate from top-level words
                segments.append(seg_dict)

        # Get words from top-level (API format)
        words = []
        if hasattr(response, 'words') and response.words:
            for word in response.words:
                words.append({
                    "word": word.word,
                    "start": word.start,
                    "end": word.end,
                })

        # Assign words to segments based on timestamps
        for segment in segments:
            segment["words"] = [
                w for w in words
                if w["start"] >= segment["start"] and w["end"] <= segment["end"]
            ]

        return {
            "text": response.text,
            "segments": segments,
            "words": words,  # Also keep at top level for convenience
        }

    def transcribe_video(self, video_path: str) -> dict:
        """
        Extract audio from video and transcribe.

        Args:
            video_path: Path to video file

        Returns:
            Whisper transcription result
        """
        # Extract audio
        audio_path = self.extract_audio(video_path)

        try:
            # Transcribe
            result = self.transcribe(audio_path)
            return result
        finally:
            # Clean up temp file
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    def get_word_timestamps(self, transcription: dict) -> list:
        """
        Extract word-level timestamps from transcription.

        Args:
            transcription: Whisper transcription result

        Returns:
            List of dicts with word, start, and end times
        """
        words = []

        # First try top-level words (API format)
        if "words" in transcription and transcription["words"]:
            for word_info in transcription["words"]:
                words.append({
                    "word": word_info.get("word", "").strip(),
                    "start": word_info.get("start", 0),
                    "end": word_info.get("end", 0),
                })
            return words

        # Fall back to segment-nested words (local Whisper format)
        for segment in transcription.get("segments", []):
            for word_info in segment.get("words", []):
                words.append({
                    "word": word_info.get("word", "").strip(),
                    "start": word_info.get("start", 0),
                    "end": word_info.get("end", 0),
                })

        return words
