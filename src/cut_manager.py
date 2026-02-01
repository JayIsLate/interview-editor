"""
Cut manager module for merging and managing proposed cuts.
"""

from typing import List, Optional
from dataclasses import dataclass, field
import copy


@dataclass
class Cut:
    """Represents a proposed cut in the video."""
    start: float
    end: float
    reason: str
    cut_type: str
    approved: Optional[bool] = None  # None=pending, True=approved, False=rejected
    duration: float = field(init=False)

    def __post_init__(self):
        self.duration = self.end - self.start

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "reason": self.reason,
            "type": self.cut_type,
            "approved": self.approved,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Cut":
        return cls(
            start=data["start"],
            end=data["end"],
            reason=data["reason"],
            cut_type=data.get("type", "unknown"),
            approved=data.get("approved"),
        )


class CutManager:
    """Manages, merges, and tracks cuts."""

    def __init__(self, padding: float = 0.1, merge_gap: float = 0.3):
        """
        Initialize the cut manager.

        Args:
            padding: Buffer (seconds) to add around cuts
            merge_gap: Maximum gap (seconds) between cuts to merge them
        """
        self.padding = padding
        self.merge_gap = merge_gap
        self.cuts: List[Cut] = []

    def add_cuts(self, cuts: List[dict]):
        """
        Add cuts from a list of dictionaries.

        Args:
            cuts: List of cut dictionaries
        """
        for cut_dict in cuts:
            self.cuts.append(Cut.from_dict(cut_dict))

    def clear_cuts(self):
        """Clear all cuts."""
        self.cuts = []

    def get_all_cuts(self) -> List[dict]:
        """Get all cuts as dictionaries."""
        return [cut.to_dict() for cut in self.cuts]

    def get_approved_cuts(self) -> List[dict]:
        """Get only approved cuts."""
        return [cut.to_dict() for cut in self.cuts if cut.approved is True]

    def get_rejected_cuts(self) -> List[dict]:
        """Get only rejected cuts."""
        return [cut.to_dict() for cut in self.cuts if cut.approved is False]

    def get_pending_cuts(self) -> List[dict]:
        """Get only pending (not yet reviewed) cuts."""
        return [cut.to_dict() for cut in self.cuts if cut.approved is None]

    def approve_cut(self, index: int):
        """Approve a cut by index."""
        if 0 <= index < len(self.cuts):
            self.cuts[index].approved = True

    def reject_cut(self, index: int):
        """Reject a cut by index."""
        if 0 <= index < len(self.cuts):
            self.cuts[index].approved = False

    def approve_all(self):
        """Approve all cuts."""
        for cut in self.cuts:
            cut.approved = True

    def reject_all(self):
        """Reject all cuts."""
        for cut in self.cuts:
            cut.approved = False

    def reset_all(self):
        """Reset all cuts to pending."""
        for cut in self.cuts:
            cut.approved = None

    def sort_cuts(self):
        """Sort cuts by start time."""
        self.cuts.sort(key=lambda c: c.start)

    def apply_padding(self, video_duration: float = None):
        """
        Apply padding to all cuts.

        Args:
            video_duration: Total video duration (to clamp end times)
        """
        for cut in self.cuts:
            cut.start = max(0, cut.start - self.padding)
            if video_duration:
                cut.end = min(video_duration, cut.end + self.padding)
            else:
                cut.end = cut.end + self.padding
            cut.duration = cut.end - cut.start

    def merge_overlapping_cuts(self) -> List[Cut]:
        """
        Merge cuts that overlap or are very close together.

        Returns:
            List of merged cuts
        """
        if not self.cuts:
            return []

        self.sort_cuts()

        merged = []
        current = copy.copy(self.cuts[0])

        for next_cut in self.cuts[1:]:
            # Check if cuts should be merged (overlap or within merge_gap)
            if next_cut.start <= current.end + self.merge_gap:
                # Merge: extend current cut
                current.end = max(current.end, next_cut.end)
                current.duration = current.end - current.start

                # Combine reasons if different
                if next_cut.reason not in current.reason:
                    current.reason = f"{current.reason} + {next_cut.reason}"

                # Keep type as "merged" if combining different types
                if next_cut.cut_type != current.cut_type:
                    current.cut_type = "merged"
            else:
                # No overlap, save current and start new
                merged.append(current)
                current = copy.copy(next_cut)

        merged.append(current)

        return merged

    def get_merged_approved_cuts(self) -> List[dict]:
        """
        Get merged list of approved cuts.

        Returns:
            List of merged cut dictionaries
        """
        # Temporarily filter to approved cuts only
        original_cuts = self.cuts
        self.cuts = [c for c in self.cuts if c.approved is True]

        merged = self.merge_overlapping_cuts()

        # Restore original cuts
        self.cuts = original_cuts

        return [cut.to_dict() for cut in merged]

    def get_keep_segments(self, video_duration: float) -> List[dict]:
        """
        Get the segments to KEEP (inverse of cuts).

        Args:
            video_duration: Total video duration

        Returns:
            List of segments to keep with start/end times
        """
        approved_cuts = self.get_merged_approved_cuts()

        if not approved_cuts:
            return [{"start": 0, "end": video_duration}]

        # Sort cuts by start time
        cuts_sorted = sorted(approved_cuts, key=lambda c: c["start"])

        keep_segments = []
        current_pos = 0

        for cut in cuts_sorted:
            if cut["start"] > current_pos:
                keep_segments.append({
                    "start": current_pos,
                    "end": cut["start"]
                })
            current_pos = cut["end"]

        # Add final segment if there's content after last cut
        if current_pos < video_duration:
            keep_segments.append({
                "start": current_pos,
                "end": video_duration
            })

        return keep_segments

    def get_statistics(self) -> dict:
        """Get statistics about the cuts."""
        total = len(self.cuts)
        approved = sum(1 for c in self.cuts if c.approved is True)
        rejected = sum(1 for c in self.cuts if c.approved is False)
        pending = sum(1 for c in self.cuts if c.approved is None)

        total_cut_time = sum(c.duration for c in self.cuts if c.approved is True)

        by_type = {}
        for cut in self.cuts:
            if cut.cut_type not in by_type:
                by_type[cut.cut_type] = 0
            by_type[cut.cut_type] += 1

        return {
            "total": total,
            "approved": approved,
            "rejected": rejected,
            "pending": pending,
            "total_cut_time": total_cut_time,
            "by_type": by_type,
        }
