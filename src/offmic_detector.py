"""
Off-mic speech detector module.

Detects quieter speech from someone not on the main microphone
(e.g. an interviewer behind the camera) by comparing windowed volume
levels against the on-mic reference level.

Uses a sliding-window approach: a region is only flagged as off-mic
when it is *consistently* quiet across the entire window, preventing
false positives from brief volume dips during normal on-mic speech.
"""

from pydub import AudioSegment
from typing import List, Tuple
import math


class OffMicDetector:
    """Detects off-mic speech using windowed volume-level analysis."""

    def __init__(
        self,
        chunk_size_ms: int = 200,
        silence_threshold: float = -40.0,
        sensitivity_offset: float = 10.0,
        min_offmic_duration: float = 0.8,
        merge_gap: float = 0.5,
        window_chunks: int = 5,
    ):
        """
        Initialize the off-mic detector.

        Args:
            chunk_size_ms: Size of each audio chunk in milliseconds.
            silence_threshold: dBFS level below which audio is considered silent.
            sensitivity_offset: How many dB quieter than on-mic counts as off-mic.
            min_offmic_duration: Minimum duration (seconds) for an off-mic region.
            merge_gap: Merge off-mic regions within this many seconds of each other.
            window_chunks: Number of chunks in the sliding window (default 5 = 1s).
        """
        self.chunk_size_ms = chunk_size_ms
        self.silence_threshold = silence_threshold
        self.sensitivity_offset = sensitivity_offset
        self.min_offmic_duration = min_offmic_duration
        self.merge_gap = merge_gap
        self.window_chunks = window_chunks

    def detect_offmic_speech(self, file_path: str = None, audio: AudioSegment = None) -> List[dict]:
        """
        Main entry point. Analyse audio and return off-mic cut dicts.

        Args:
            file_path: Path to the audio/video file (used if audio is None).
            audio: Pre-loaded AudioSegment to avoid redundant disk reads.

        Returns:
            List of cut dictionaries with type "offmic" and approved True.
        """
        if audio is None:
            audio = AudioSegment.from_file(file_path)
        chunk_levels = self._compute_chunk_levels(audio)

        # Guard rail: need enough audible chunks to calibrate
        audible = [lvl for lvl in chunk_levels if lvl > self.silence_threshold]
        if len(audible) < 0.20 * len(chunk_levels):
            return []

        onmic_level = self._determine_onmic_level(chunk_levels)
        offmic_threshold = self._offmic_threshold(onmic_level)
        labels = self._classify_windowed(chunk_levels, offmic_threshold)
        regions = self._extract_offmic_regions(labels)
        return self._regions_to_cuts(regions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_chunk_levels(self, audio: AudioSegment) -> List[float]:
        """Break audio into fixed-size chunks and return dBFS per chunk."""
        levels: List[float] = []
        total_ms = len(audio)
        for start_ms in range(0, total_ms, self.chunk_size_ms):
            chunk = audio[start_ms : start_ms + self.chunk_size_ms]
            dbfs = chunk.dBFS
            # pydub returns -inf for totally silent chunks
            if dbfs == float("-inf") or math.isinf(dbfs):
                dbfs = -100.0
            levels.append(dbfs)
        return levels

    def _determine_onmic_level(self, chunk_levels: List[float]) -> float:
        """Return the 75th-percentile dBFS of audible chunks."""
        audible = sorted(
            lvl for lvl in chunk_levels if lvl > self.silence_threshold
        )
        if not audible:
            return self.silence_threshold
        idx = int(len(audible) * 0.75)
        idx = min(idx, len(audible) - 1)
        return audible[idx]

    def _offmic_threshold(self, onmic_level: float) -> float:
        """Compute and clamp the off-mic threshold."""
        threshold = onmic_level - self.sensitivity_offset
        # Never let the threshold fall below silence_threshold + 3 dB
        floor = self.silence_threshold + 3.0
        return max(threshold, floor)

    def _classify_windowed(
        self, chunk_levels: List[float], offmic_threshold: float
    ) -> List[str]:
        """
        Classify each chunk using a windowed-average approach.

        A chunk is labelled 'offmic' when:
        - The chunk itself is at or below the off-mic threshold, AND
        - The average level of its surrounding window is also at or below
          the threshold (filters isolated quiet dips during on-mic speech).

        This catches sustained quiet regions (real off-mic speaker) while
        ignoring brief volume dips within normal on-mic speech.
        """
        n = len(chunk_levels)
        half = self.window_chunks // 2
        labels: List[str] = []

        for i in range(n):
            lvl = chunk_levels[i]

            # Silence is silence regardless of window
            if lvl <= self.silence_threshold:
                labels.append("silent")
                continue

            # The chunk itself must be below the off-mic threshold
            if lvl > offmic_threshold:
                labels.append("onmic")
                continue

            # Build window centred on i
            win_start = max(0, i - half)
            win_end = min(n, i + half + 1)
            window = chunk_levels[win_start:win_end]

            # Check if the window average is below the threshold
            avg = sum(window) / len(window)
            if avg > offmic_threshold:
                # Surrounding context is too loud — this is a quiet
                # moment within on-mic speech
                labels.append("onmic")
            else:
                has_audio = any(w > self.silence_threshold for w in window)
                if has_audio:
                    labels.append("offmic")
                else:
                    labels.append("silent")

        return labels

    def _extract_offmic_regions(
        self, labels: List[str]
    ) -> List[Tuple[float, float]]:
        """
        Find contiguous off-mic runs, merge nearby ones, and filter short ones.

        Returns list of (start_sec, end_sec) tuples.
        """
        chunk_sec = self.chunk_size_ms / 1000.0

        # 1. Collect raw contiguous off-mic regions
        raw_regions: List[Tuple[float, float]] = []
        i = 0
        while i < len(labels):
            if labels[i] == "offmic":
                start = i * chunk_sec
                j = i
                while j < len(labels) and labels[j] == "offmic":
                    j += 1
                end = j * chunk_sec
                raw_regions.append((start, end))
                i = j
            else:
                i += 1

        if not raw_regions:
            return []

        # 2. Merge regions within merge_gap of each other
        merged: List[Tuple[float, float]] = [raw_regions[0]]
        for start, end in raw_regions[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= self.merge_gap:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))

        # 3. Filter out regions shorter than min_offmic_duration
        return [
            (s, e) for s, e in merged if (e - s) >= self.min_offmic_duration
        ]

    def _regions_to_cuts(
        self, regions: List[Tuple[float, float]]
    ) -> List[dict]:
        """Convert (start, end) regions into standard cut dicts."""
        cuts: List[dict] = []
        for start, end in regions:
            duration = round(end - start, 2)
            cuts.append(
                {
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "duration": duration,
                    "reason": f"off-mic speech ({duration}s)",
                    "type": "offmic",
                    "approved": True,
                }
            )
        return cuts
