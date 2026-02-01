"""
Filler word detector module for finding "um", "uh", etc.
"""

import re
from typing import List


class FillerDetector:
    """Detects filler words in transcriptions."""

    DEFAULT_FILLER_WORDS = [
        "um", "uh", "uhm", "uhh", "er", "ah",
        "like", "you know", "basically", "actually",
        "literally", "so", "right", "i mean"
    ]

    def __init__(self, filler_words: List[str] = None):
        """
        Initialize the filler detector.

        Args:
            filler_words: List of filler words/phrases to detect
        """
        self.filler_words = filler_words or self.DEFAULT_FILLER_WORDS
        # Sort by length (longest first) to match multi-word phrases first
        self.filler_words = sorted(self.filler_words, key=len, reverse=True)

    def detect_fillers(self, word_timestamps: List[dict]) -> List[dict]:
        """
        Detect filler words in a list of word timestamps.

        Args:
            word_timestamps: List of dicts with 'word', 'start', 'end'

        Returns:
            List of cut dictionaries for each filler word found
        """
        cuts = []

        for i, word_info in enumerate(word_timestamps):
            word = word_info.get("word", "").lower().strip()
            # Remove punctuation for matching
            word_clean = re.sub(r"[^\w\s]", "", word)

            # Check single-word fillers
            for filler in self.filler_words:
                if " " not in filler:  # Single word filler
                    if word_clean == filler:
                        cuts.append({
                            "start": word_info["start"],
                            "end": word_info["end"],
                            "duration": word_info["end"] - word_info["start"],
                            "reason": f"filler: {filler}",
                            "type": "filler",
                            "word": word,
                            "approved": None,
                        })
                        break

        # Check multi-word phrases
        cuts.extend(self._detect_multi_word_fillers(word_timestamps))

        # Sort by start time
        cuts.sort(key=lambda x: x["start"])

        return cuts

    def _detect_multi_word_fillers(self, word_timestamps: List[dict]) -> List[dict]:
        """
        Detect multi-word filler phrases.

        Args:
            word_timestamps: List of word timestamp dicts

        Returns:
            List of cuts for multi-word fillers
        """
        cuts = []
        multi_word_fillers = [f for f in self.filler_words if " " in f]

        for filler in multi_word_fillers:
            filler_words = filler.lower().split()
            filler_len = len(filler_words)

            for i in range(len(word_timestamps) - filler_len + 1):
                # Get sequence of words
                sequence = [
                    re.sub(r"[^\w\s]", "", word_timestamps[i + j]["word"].lower().strip())
                    for j in range(filler_len)
                ]

                if sequence == filler_words:
                    start = word_timestamps[i]["start"]
                    end = word_timestamps[i + filler_len - 1]["end"]
                    cuts.append({
                        "start": start,
                        "end": end,
                        "duration": end - start,
                        "reason": f"filler: {filler}",
                        "type": "filler",
                        "word": filler,
                        "approved": None,
                    })

        return cuts

    def detect_long_pauses(
        self, word_timestamps: List[dict], min_pause: float = 0.5
    ) -> List[dict]:
        """
        Detect long pauses between words (mid-sentence hesitation).

        Args:
            word_timestamps: List of word timestamp dicts
            min_pause: Minimum pause duration in seconds

        Returns:
            List of cuts for long pauses
        """
        cuts = []

        for i in range(len(word_timestamps) - 1):
            current_end = word_timestamps[i]["end"]
            next_start = word_timestamps[i + 1]["start"]
            gap = next_start - current_end

            if gap >= min_pause:
                cuts.append({
                    "start": current_end,
                    "end": next_start,
                    "duration": gap,
                    "reason": f"pause ({gap:.1f}s)",
                    "type": "pause",
                    "approved": None,
                })

        return cuts

    def detect_false_starts(self, word_timestamps: List[dict]) -> List[dict]:
        """
        Detect false starts (sentence restarts) in word timestamps.

        Detects two patterns:
        a) Immediate word repetition — "I I I think" → flag the repeated words before the last
        b) Phrase restart — "So I was— so I was thinking" → flag from first occurrence through
           to just before the second occurrence

        Args:
            word_timestamps: List of dicts with 'word', 'start', 'end'

        Returns:
            List of cut dicts with type "false_start" and approved=False (flagged for review)
        """
        cuts = []

        def clean(word: str) -> str:
            return re.sub(r"[^\w]", "", word.lower().strip())

        # --- Pattern A: Immediate word repetition ---
        i = 0
        while i < len(word_timestamps):
            word_clean = clean(word_timestamps[i]["word"])
            if not word_clean:
                i += 1
                continue

            # Count consecutive duplicates
            run_end = i + 1
            while run_end < len(word_timestamps) and clean(word_timestamps[run_end]["word"]) == word_clean:
                run_end += 1

            run_length = run_end - i
            if run_length >= 2:
                # Flag all but the last repetition
                cut_start = word_timestamps[i]["start"]
                cut_end = word_timestamps[run_end - 2]["end"]
                repeated_word = word_timestamps[i]["word"].strip()
                cuts.append({
                    "start": cut_start,
                    "end": cut_end,
                    "duration": cut_end - cut_start,
                    "reason": f'false start: repeated "{repeated_word}" {run_length - 1}x',
                    "type": "false_start",
                    "approved": False,
                })
                i = run_end
            else:
                i += 1

        # --- Pattern B: Phrase restart (2-3 word sequences repeated within ~5s) ---
        for phrase_len in [3, 2]:
            if len(word_timestamps) < phrase_len * 2:
                continue

            j = 0
            while j <= len(word_timestamps) - phrase_len:
                phrase = [clean(word_timestamps[j + k]["word"]) for k in range(phrase_len)]
                if not all(phrase):
                    j += 1
                    continue

                phrase_start_time = word_timestamps[j]["start"]
                window_end_time = phrase_start_time + 5.0

                # Search for a repeat starting after this phrase
                search_start = j + phrase_len
                found = False

                for m in range(search_start, len(word_timestamps) - phrase_len + 1):
                    if word_timestamps[m]["start"] > window_end_time:
                        break

                    candidate = [clean(word_timestamps[m + k]["word"]) for k in range(phrase_len)]
                    if candidate == phrase:
                        # Found a restart — flag from first occurrence start to just before second
                        cut_start = word_timestamps[j]["start"]
                        cut_end = word_timestamps[m - 1]["end"] if m > 0 else word_timestamps[m]["start"]
                        # Avoid zero/negative duration
                        if cut_end <= cut_start:
                            cut_end = word_timestamps[m]["start"]
                        if cut_end <= cut_start:
                            break

                        # Check we're not overlapping with an already-detected repetition cut
                        overlap = False
                        for existing in cuts:
                            if (cut_start < existing["end"] and cut_end > existing["start"]):
                                overlap = True
                                break

                        if not overlap:
                            phrase_text = " ".join(
                                word_timestamps[j + k]["word"].strip() for k in range(phrase_len)
                            )
                            cuts.append({
                                "start": cut_start,
                                "end": cut_end,
                                "duration": cut_end - cut_start,
                                "reason": f'false start: restarted "{phrase_text}"',
                                "type": "false_start",
                                "approved": False,
                            })

                        # Skip past this match to avoid double-detection
                        j = m + phrase_len
                        found = True
                        break

                if not found:
                    j += 1

        # Sort by start time and deduplicate overlaps
        cuts.sort(key=lambda x: x["start"])
        return cuts

    def add_filler_word(self, word: str):
        """Add a filler word to the detection list."""
        word = word.lower().strip()
        if word not in self.filler_words:
            self.filler_words.append(word)
            self.filler_words = sorted(self.filler_words, key=len, reverse=True)

    def remove_filler_word(self, word: str):
        """Remove a filler word from the detection list."""
        word = word.lower().strip()
        if word in self.filler_words:
            self.filler_words.remove(word)
