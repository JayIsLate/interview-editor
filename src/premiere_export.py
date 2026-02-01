"""
Premiere Pro XML export module.
Generates Final Cut Pro 7 XML format that Premiere can import.
"""

import os
import uuid
from typing import List
from xml.etree import ElementTree as ET
from xml.dom import minidom


class PremiereExporter:
    """Exports edit decisions to Premiere Pro compatible XML."""

    def __init__(self, frame_rate: float = 29.97, original_filename: str = None):
        """
        Initialize the exporter.

        Args:
            frame_rate: Video frame rate for timecode calculations
            original_filename: Original video filename for export
        """
        self.frame_rate = frame_rate
        self.timebase = round(frame_rate)
        self.original_filename = original_filename
        self.is_ntsc = frame_rate in [29.97, 59.94, 23.976]

    def seconds_to_frames(self, seconds: float) -> int:
        """Convert seconds to frame count."""
        return int(seconds * self.frame_rate)

    def create_fcpxml(
        self,
        video_path: str,
        keep_segments: List[dict],
        video_duration: float,
        project_name: str = "Interview Edit",
        original_filename: str = None
    ) -> str:
        """
        Create Final Cut Pro 7 XML with edit points.
        """
        # Use original filename if provided
        if original_filename:
            video_name = original_filename
        elif self.original_filename:
            video_name = self.original_filename
        else:
            video_name = os.path.basename(video_path)

        # Create proper file:// URL for the video path
        # Handle both absolute paths and relative paths
        if video_path.startswith('/'):
            pathurl = f"file://localhost{video_path}"
        else:
            pathurl = video_path

        total_frames = self.seconds_to_frames(video_duration)
        file_id = f"file-{uuid.uuid4().hex[:8]}"
        masterclip_id = f"masterclip-{uuid.uuid4().hex[:8]}"

        # Build XML manually for better control
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<!DOCTYPE xmeml>',
            '<xmeml version="4">',
            '  <sequence>',
            f'    <name>{project_name}</name>',
            f'    <duration>{total_frames}</duration>',
            '    <rate>',
            f'      <timebase>{self.timebase}</timebase>',
            f'      <ntsc>{"TRUE" if self.is_ntsc else "FALSE"}</ntsc>',
            '    </rate>',
            '    <timecode>',
            '      <rate>',
            f'        <timebase>{self.timebase}</timebase>',
            f'        <ntsc>{"TRUE" if self.is_ntsc else "FALSE"}</ntsc>',
            '      </rate>',
            '      <string>00:00:00:00</string>',
            '      <frame>0</frame>',
            '      <displayformat>NDF</displayformat>',
            '    </timecode>',
            '    <media>',
            '      <video>',
            '        <format>',
            '          <samplecharacteristics>',
            '            <width>1920</width>',
            '            <height>1080</height>',
            '            <pixelaspectratio>square</pixelaspectratio>',
            '            <rate>',
            f'              <timebase>{self.timebase}</timebase>',
            f'              <ntsc>{"TRUE" if self.is_ntsc else "FALSE"}</ntsc>',
            '            </rate>',
            '          </samplecharacteristics>',
            '        </format>',
            '        <track>',
        ]

        # Add video clips
        timeline_position = 0
        for i, segment in enumerate(keep_segments):
            start_frame = self.seconds_to_frames(segment["start"])
            end_frame = self.seconds_to_frames(segment["end"])
            seg_duration = end_frame - start_frame

            xml_lines.extend([
                f'          <clipitem id="clipitem-v{i+1}">',
                f'            <name>{video_name}</name>',
                f'            <duration>{total_frames}</duration>',
                '            <rate>',
                f'              <timebase>{self.timebase}</timebase>',
                f'              <ntsc>{"TRUE" if self.is_ntsc else "FALSE"}</ntsc>',
                '            </rate>',
                f'            <start>{timeline_position}</start>',
                f'            <end>{timeline_position + seg_duration}</end>',
                f'            <in>{start_frame}</in>',
                f'            <out>{end_frame}</out>',
                f'            <file id="{file_id}">',
                f'              <name>{video_name}</name>',
                f'              <pathurl>{pathurl}</pathurl>',
                f'              <duration>{total_frames}</duration>',
                '              <rate>',
                f'                <timebase>{self.timebase}</timebase>',
                f'                <ntsc>{"TRUE" if self.is_ntsc else "FALSE"}</ntsc>',
                '              </rate>',
                '            </file>',
                '          </clipitem>',
            ])
            timeline_position += seg_duration

        xml_lines.extend([
            '        </track>',
            '      </video>',
            '      <audio>',
            '        <format>',
            '          <samplecharacteristics>',
            '            <samplerate>48000</samplerate>',
            '            <depth>16</depth>',
            '          </samplecharacteristics>',
            '        </format>',
            '        <track>',
        ])

        # Add audio clips
        timeline_position = 0
        for i, segment in enumerate(keep_segments):
            start_frame = self.seconds_to_frames(segment["start"])
            end_frame = self.seconds_to_frames(segment["end"])
            seg_duration = end_frame - start_frame

            xml_lines.extend([
                f'          <clipitem id="clipitem-a{i+1}">',
                f'            <name>{video_name}</name>',
                f'            <duration>{total_frames}</duration>',
                '            <rate>',
                f'              <timebase>{self.timebase}</timebase>',
                f'              <ntsc>{"TRUE" if self.is_ntsc else "FALSE"}</ntsc>',
                '            </rate>',
                f'            <start>{timeline_position}</start>',
                f'            <end>{timeline_position + seg_duration}</end>',
                f'            <in>{start_frame}</in>',
                f'            <out>{end_frame}</out>',
                f'            <file id="{file_id}"/>',
                '            <sourcetrack>',
                '              <mediatype>audio</mediatype>',
                '              <trackindex>1</trackindex>',
                '            </sourcetrack>',
                '          </clipitem>',
            ])
            timeline_position += seg_duration

        xml_lines.extend([
            '        </track>',
            '      </audio>',
            '    </media>',
            '  </sequence>',
            '</xmeml>',
        ])

        return '\n'.join(xml_lines)

    def create_markers_xml(
        self,
        video_path: str,
        cuts: List[dict],
        video_duration: float,
        project_name: str = "Interview Markers",
        original_filename: str = None
    ) -> str:
        """Create XML with markers at cut points."""
        if original_filename:
            video_name = original_filename
        elif self.original_filename:
            video_name = self.original_filename
        else:
            video_name = os.path.basename(video_path)

        # Create proper file:// URL
        if video_path.startswith('/'):
            pathurl = f"file://localhost{video_path}"
        else:
            pathurl = video_path

        total_frames = self.seconds_to_frames(video_duration)
        file_id = f"file-{uuid.uuid4().hex[:8]}"

        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<!DOCTYPE xmeml>',
            '<xmeml version="4">',
            '  <sequence>',
            f'    <name>{project_name}</name>',
            f'    <duration>{total_frames}</duration>',
            '    <rate>',
            f'      <timebase>{self.timebase}</timebase>',
            f'      <ntsc>{"TRUE" if self.is_ntsc else "FALSE"}</ntsc>',
            '    </rate>',
            '    <media>',
            '      <video>',
            '        <track>',
            '          <clipitem id="clipitem-1">',
            f'            <name>{video_name}</name>',
            f'            <duration>{total_frames}</duration>',
            '            <rate>',
            f'              <timebase>{self.timebase}</timebase>',
            f'              <ntsc>{"TRUE" if self.is_ntsc else "FALSE"}</ntsc>',
            '            </rate>',
            '            <start>0</start>',
            f'            <end>{total_frames}</end>',
            '            <in>0</in>',
            f'            <out>{total_frames}</out>',
            f'            <file id="{file_id}">',
            f'              <name>{video_name}</name>',
            f'              <pathurl>{pathurl}</pathurl>',
            f'              <duration>{total_frames}</duration>',
            '            </file>',
        ]

        # Add markers
        for cut in cuts:
            in_frame = self.seconds_to_frames(cut["start"])
            out_frame = self.seconds_to_frames(cut["end"])
            reason = cut.get("reason", "cut").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            xml_lines.extend([
                '            <marker>',
                f'              <name>CUT: {reason}</name>',
                f'              <comment>{cut["start"]:.2f}s - {cut["end"]:.2f}s</comment>',
                f'              <in>{in_frame}</in>',
                f'              <out>{out_frame}</out>',
                '            </marker>',
            ])

        xml_lines.extend([
            '          </clipitem>',
            '        </track>',
            '      </video>',
            '    </media>',
            '  </sequence>',
            '</xmeml>',
        ])

        return '\n'.join(xml_lines)

    def create_edl(
        self,
        keep_segments: List[dict],
        project_name: str = "Interview Edit",
        original_filename: str = None
    ) -> str:
        """Create an EDL (Edit Decision List)."""
        video_name = original_filename or self.original_filename or "SOURCE_VIDEO"

        lines = [
            f"TITLE: {project_name}",
            "FCM: NON-DROP FRAME",
            ""
        ]

        rec_position = 0

        for i, segment in enumerate(keep_segments, 1):
            src_in = self._seconds_to_timecode(segment["start"])
            src_out = self._seconds_to_timecode(segment["end"])

            segment_duration = segment["end"] - segment["start"]
            rec_in = self._seconds_to_timecode(rec_position)
            rec_out = self._seconds_to_timecode(rec_position + segment_duration)
            rec_position += segment_duration

            lines.append(f"{i:03d}  AX       AA/V  C        {src_in} {src_out} {rec_in} {rec_out}")
            lines.append(f"* FROM CLIP NAME: {video_name}")
            lines.append("")

        return "\n".join(lines)

    def create_csv(
        self,
        keep_segments: List[dict],
        cuts: List[dict],
        original_filename: str = None
    ) -> str:
        """Create a simple CSV with cut timecodes (for manual editing)."""
        video_name = original_filename or self.original_filename or "SOURCE_VIDEO"

        lines = [
            "Type,Start Timecode,End Timecode,Start Seconds,End Seconds,Duration,Note"
        ]

        # Add segments to KEEP
        for i, seg in enumerate(keep_segments):
            tc_in = self._seconds_to_timecode(seg["start"])
            tc_out = self._seconds_to_timecode(seg["end"])
            duration = seg["end"] - seg["start"]
            lines.append(f"KEEP,{tc_in},{tc_out},{seg['start']:.2f},{seg['end']:.2f},{duration:.2f},Segment {i+1}")

        lines.append("")

        # Add sections to CUT
        for cut in cuts:
            tc_in = self._seconds_to_timecode(cut["start"])
            tc_out = self._seconds_to_timecode(cut["end"])
            reason = cut.get("reason", "").replace(",", ";")
            lines.append(f"CUT,{tc_in},{tc_out},{cut['start']:.2f},{cut['end']:.2f},{cut['duration']:.2f},{reason}")

        return "\n".join(lines)

    def _seconds_to_timecode(self, seconds: float) -> str:
        """Convert seconds to SMPTE timecode."""
        total_frames = int(seconds * self.frame_rate)
        frames = total_frames % self.timebase
        total_seconds = total_frames // self.timebase
        secs = total_seconds % 60
        total_minutes = total_seconds // 60
        mins = total_minutes % 60
        hours = total_minutes // 60

        return f"{hours:02d}:{mins:02d}:{secs:02d}:{frames:02d}"

    def export_to_file(self, content: str, output_path: str):
        """Write content to a file."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
