"""
Interview Auto-Editor - Streamlit Application
Automatically detect pauses, filler words, and dead space in interview videos.
"""

import streamlit as st
import tempfile
import os
import io
import shutil
import yaml
from pathlib import Path
from pydub import AudioSegment

from src.transcriber import Transcriber
from src.silence_detector import SilenceDetector
from src.filler_detector import FillerDetector
from src.cut_manager import CutManager
from src.premiere_export import PremiereExporter
from src.offmic_detector import OffMicDetector
from src.thumbnail_generator import ThumbnailGenerator, THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT
from PIL import Image


# Page configuration
st.set_page_config(
    page_title="Interview Auto-Editor",
    page_icon="✂️",
    layout="wide"
)


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def format_timestamp(seconds):
    """Format seconds to MM:SS format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def format_time_range(start, end):
    """Format a time range."""
    return f"{format_timestamp(start)} → {format_timestamp(end)}"


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "video_path": None,
        "video_duration": None,
        "transcription": None,
        "word_timestamps": None,
        "cuts": [],
        "processed": False,
        "config": load_config(),
        "original_filename": None,
        "thumbnail_frames": None,
        "selected_frame": None,
        "selected_timestamp": None,
        "thumbnail_preview": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """Render the sidebar with settings."""
    st.sidebar.header("Settings")

    config = st.session_state.config

    # Edit style preset
    st.sidebar.subheader("Edit Style")
    edit_style = st.sidebar.radio(
        "How aggressive should the cuts be?",
        options=["Conversational (raw)", "Conservative (natural)", "Moderate", "Aggressive (tight)"],
        index=0,
        help="Conversational keeps speech raw and only removes dead air between questions. Conservative keeps more natural pauses, Aggressive removes more dead air."
    )

    # Set defaults based on style
    if edit_style == "Conversational (raw)":
        default_silence = 3.0
        default_padding = 0.3
        default_pause_threshold = 999
        detect_fillers = False
        detect_pauses_default = False
        detect_false_starts_default = True
        default_min_keep = 2.0
        default_min_cut = 1.0
        detect_offmic_default = True
        default_offmic_sensitivity = 8
    elif edit_style == "Conservative (natural)":
        default_silence = 2.0
        default_padding = 0.3
        default_pause_threshold = 1.5
        detect_fillers = False
        detect_pauses_default = False
        detect_false_starts_default = False
        default_min_keep = 1.0
        default_min_cut = 0.3
        detect_offmic_default = False
        default_offmic_sensitivity = 10
    elif edit_style == "Moderate":
        default_silence = 1.2
        default_padding = 0.2
        default_pause_threshold = 0.8
        detect_fillers = True
        detect_pauses_default = False
        detect_false_starts_default = False
        default_min_keep = 1.0
        default_min_cut = 0.3
        detect_offmic_default = False
        default_offmic_sensitivity = 10
    else:  # Aggressive
        default_silence = 0.8
        default_padding = 0.1
        default_pause_threshold = 0.5
        detect_fillers = True
        detect_pauses_default = False
        detect_false_starts_default = False
        default_min_keep = 1.0
        default_min_cut = 0.3
        detect_offmic_default = False
        default_offmic_sensitivity = 8

    st.sidebar.markdown("---")

    # Silence settings
    st.sidebar.subheader("Silence Detection")
    silence_duration = st.sidebar.slider(
        "Minimum silence to cut (seconds)",
        min_value=0.5,
        max_value=5.0,
        value=default_silence,
        step=0.1,
        help="Only cut silences LONGER than this. Higher = more natural, keeps thinking pauses."
    )
    silence_threshold = st.sidebar.slider(
        "Silence threshold (dB)",
        min_value=-60,
        max_value=-20,
        value=config.get("silence", {}).get("threshold_db", -40),
        step=5,
        help="Audio below this level is considered silence"
    )

    # Off-mic detection
    st.sidebar.subheader("Off-Mic Detection")
    detect_offmic = st.sidebar.checkbox(
        "Cut off-mic audio (interviewer behind camera)",
        value=detect_offmic_default,
        help="Detects quieter speech from someone not on the main mic and flags it for removal."
    )

    if detect_offmic:
        offmic_sensitivity = st.sidebar.slider(
            "Off-mic sensitivity (dB)",
            min_value=3,
            max_value=20,
            value=default_offmic_sensitivity,
            step=1,
            help="How much quieter than the main speaker counts as off-mic. Lower = more aggressive."
        )
    else:
        offmic_sensitivity = default_offmic_sensitivity

    # Filler word settings
    st.sidebar.subheader("Filler Words")
    detect_filler_words = st.sidebar.checkbox(
        "Detect filler words (um, uh, like...)",
        value=detect_fillers,
        help="Uncheck to keep all filler words for a more natural feel"
    )

    if detect_filler_words:
        default_fillers = ["um", "uh", "uhm", "uhh"]  # Reduced list - only clear fillers
        filler_words_text = st.sidebar.text_area(
            "Filler words (one per line)",
            value="\n".join(default_fillers),
            height=100,
            help="Words to detect. Fewer words = more natural feel."
        )
        filler_words = [w.strip() for w in filler_words_text.split("\n") if w.strip()]
    else:
        filler_words = []

    # Pause detection
    st.sidebar.subheader("Pause Detection")
    detect_pauses = st.sidebar.checkbox(
        "Detect long pauses between words",
        value=detect_pauses_default,
        help="Cuts mid-sentence pauses. Usually best to leave OFF for natural interviews."
    )

    if detect_pauses:
        pause_threshold = st.sidebar.slider(
            "Minimum pause to cut (seconds)",
            min_value=0.5,
            max_value=3.0,
            value=default_pause_threshold if default_pause_threshold < 999 else 1.5,
            step=0.1,
            help="Pauses longer than this between words will be flagged"
        )
    else:
        pause_threshold = 999  # Effectively disabled

    # False start detection
    st.sidebar.subheader("False Start Detection")
    detect_false_starts = st.sidebar.checkbox(
        "Detect false starts (sentence restarts)",
        value=detect_false_starts_default,
        help="Flags moments where someone starts a sentence, stops, and restarts. Shows them for review — not auto-cut."
    )

    # Cut settings
    st.sidebar.subheader("Cut Smoothness")
    padding = st.sidebar.slider(
        "Padding around cuts (seconds)",
        min_value=0.1,
        max_value=1.0,
        value=default_padding,
        step=0.05,
        help="Extra buffer around each cut. Higher = smoother but keeps more dead air."
    )

    min_keep_duration = st.sidebar.slider(
        "Minimum segment length (seconds)",
        min_value=0.5,
        max_value=3.0,
        value=default_min_keep,
        step=0.1,
        help="Don't create segments shorter than this. Prevents choppy micro-cuts."
    )

    st.sidebar.markdown("---")

    # OpenAI API settings
    st.sidebar.subheader("Transcription (OpenAI API)")

    # Check for API key in secrets first, then allow manual input
    api_key = None
    if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
        api_key = st.secrets['OPENAI_API_KEY']
        st.sidebar.success("API key loaded from secrets")
    else:
        api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="Get your API key from platform.openai.com. Cost: ~$0.006/min of audio.",
            placeholder="sk-..."
        )
        if not api_key:
            st.sidebar.warning("Enter your OpenAI API key to enable transcription")

    # Keep whisper_model for compatibility (not used by API)
    whisper_model = "whisper-1"

    # Export settings
    st.sidebar.subheader("Export")
    frame_rate = st.sidebar.selectbox(
        "Frame rate",
        options=[23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0],
        index=3,  # default to 29.97
        help="Match your video's frame rate"
    )

    return {
        "silence_duration": silence_duration,
        "silence_threshold": silence_threshold,
        "detect_offmic": detect_offmic,
        "offmic_sensitivity": offmic_sensitivity,
        "filler_words": filler_words,
        "detect_filler_words": detect_filler_words,
        "detect_pauses": detect_pauses,
        "pause_threshold": pause_threshold,
        "detect_false_starts": detect_false_starts,
        "padding": padding,
        "min_keep_duration": min_keep_duration,
        "min_cut_duration": default_min_cut,
        "whisper_model": whisper_model,
        "frame_rate": frame_rate,
        "api_key": api_key,
    }


def extract_audio_once(video_path):
    """Extract audio from a video file to a temporary WAV file.

    Returns the path to the temporary WAV file. The caller is responsible
    for deleting it when done.
    """
    audio = AudioSegment.from_file(video_path)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    audio.export(tmp.name, format="wav")
    return tmp.name


def process_video(video_path, settings, progress_callback=None):
    """Process video and detect cuts."""
    cuts = []

    # Step 1: Transcribe
    if progress_callback:
        progress_callback(0.1, "Connecting to OpenAI Whisper API...")

    transcriber = Transcriber(
        model_name=settings["whisper_model"],
        language="en",
        api_key=settings.get("api_key")
    )

    if progress_callback:
        progress_callback(0.2, "Extracting audio and transcribing (this may take a moment)...")

    transcription = transcriber.transcribe_video(video_path)
    word_timestamps = transcriber.get_word_timestamps(transcription)

    st.session_state.transcription = transcription
    st.session_state.word_timestamps = word_timestamps

    # Extract audio once — all detectors reuse this WAV file
    if progress_callback:
        progress_callback(0.4, "Extracting audio for analysis...")

    audio_path = extract_audio_once(video_path)
    try:
        # Step 2: Detect silence and get duration in a single load
        if progress_callback:
            progress_callback(0.5, "Detecting silence...")

        silence_detector = SilenceDetector(
            min_silence_len=int(settings["silence_duration"] * 1000),
            silence_thresh=settings["silence_threshold"]
        )
        silence_cuts, video_duration = silence_detector.get_silence_cuts_and_duration(audio_path)
        cuts.extend(silence_cuts)
        st.session_state.video_duration = video_duration

        # Step 2b: Detect off-mic speech (only if enabled)
        if settings.get("detect_offmic", False):
            if progress_callback:
                progress_callback(0.6, "Detecting off-mic speech...")
            offmic_detector = OffMicDetector(
                silence_threshold=settings["silence_threshold"],
                sensitivity_offset=settings.get("offmic_sensitivity", 10),
            )
            offmic_cuts = offmic_detector.detect_offmic_speech(audio_path)
            cuts.extend(offmic_cuts)
    finally:
        os.unlink(audio_path)

    # Step 3: Detect filler words (only if enabled)
    if progress_callback:
        progress_callback(0.7, "Detecting filler words...")

    if settings.get("detect_filler_words", True) and settings["filler_words"]:
        filler_detector = FillerDetector(filler_words=settings["filler_words"])
        filler_cuts = filler_detector.detect_fillers(word_timestamps)
        cuts.extend(filler_cuts)
    else:
        filler_detector = FillerDetector(filler_words=[])

    # Detect long pauses between words (only if enabled)
    if settings.get("detect_pauses", False):
        pause_threshold = settings.get("pause_threshold", 1.0)
        pause_cuts = filler_detector.detect_long_pauses(word_timestamps, min_pause=pause_threshold)
        cuts.extend(pause_cuts)

    # Detect false starts (only if enabled)
    if settings.get("detect_false_starts", False):
        if progress_callback:
            progress_callback(0.8, "Detecting false starts...")
        false_start_cuts = filler_detector.detect_false_starts(word_timestamps)
        cuts.extend(false_start_cuts)

    # Step 4: Filter and organize cuts
    if progress_callback:
        progress_callback(0.9, "Organizing cuts...")

    # Filter out cuts that are too short to matter
    min_cut_duration = settings.get("min_cut_duration", 0.3)
    cuts = [c for c in cuts if c["duration"] >= min_cut_duration]

    cut_manager = CutManager(padding=settings["padding"])
    cut_manager.add_cuts(cuts)
    cut_manager.sort_cuts()

    # Filter out cuts that would leave segments too short
    min_keep = settings.get("min_keep_duration", 1.0)
    filtered_cuts = filter_for_natural_flow(cut_manager.get_all_cuts(), video_duration, min_keep)

    # Pre-approve all cuts by default (user can reject)
    # Exception: false starts default to not-approved (flagged for review only)
    for cut in filtered_cuts:
        if cut.get("type") == "false_start":
            cut["approved"] = False
        else:
            cut["approved"] = True

    if progress_callback:
        progress_callback(1.0, "Done!")

    return filtered_cuts


def filter_for_natural_flow(cuts, video_duration, min_segment_duration=1.0):
    """
    Filter cuts to ensure remaining segments aren't too short (prevents choppy edits).

    Args:
        cuts: List of cut dictionaries
        video_duration: Total video duration
        min_segment_duration: Minimum duration for kept segments

    Returns:
        Filtered list of cuts
    """
    if not cuts:
        return cuts

    # Sort cuts by start time
    sorted_cuts = sorted(cuts, key=lambda c: c["start"])

    # Check what segments would remain if we made all cuts
    # and filter out cuts that would create too-short segments
    filtered = []

    for i, cut in enumerate(sorted_cuts):
        # Calculate segment before this cut
        if i == 0:
            segment_before = cut["start"]
        else:
            segment_before = cut["start"] - sorted_cuts[i-1]["end"]

        # Calculate segment after this cut
        if i == len(sorted_cuts) - 1:
            segment_after = video_duration - cut["end"]
        else:
            segment_after = sorted_cuts[i+1]["start"] - cut["end"]

        # Only include cut if it doesn't create too-short segments
        if segment_before >= min_segment_duration and segment_after >= min_segment_duration:
            filtered.append(cut)
        elif cut["duration"] > 3.0:
            # Always include very long silences even if it makes short segments
            filtered.append(cut)

    return filtered


def render_cuts_list(cuts, settings):
    """Render the list of cuts for review."""
    if not cuts:
        st.info("No cuts detected. Your video might not have any significant pauses or filler words.")
        return cuts

    # Summary stats at top
    total = len(cuts)
    approved = sum(1 for c in cuts if c.get("approved") is True)
    rejected = sum(1 for c in cuts if c.get("approved") is False)
    total_cut_time = sum(c["duration"] for c in cuts if c.get("approved") is True)

    # Summary box
    st.markdown("---")
    st.subheader("Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cuts Found", total)
    col2.metric("Approved (will cut)", approved)
    col3.metric("Rejected (will keep)", rejected)
    col4.metric("Time You'll Save", f"{total_cut_time:.1f}s")

    st.markdown("---")

    # Explanation
    st.markdown("""
    **How this works:**
    - Each row below is a section of video the tool wants to CUT (remove)
    - **Approve (✓)** = Yes, cut this out of my video
    - **Reject (✗)** = No, keep this in my video
    """)

    # Bulk actions
    st.subheader("Quick Actions")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])

    if col1.button("✓ Approve All", type="primary", help="Mark all cuts for removal"):
        for cut in cuts:
            cut["approved"] = True
        st.rerun()

    if col2.button("✗ Reject All", help="Keep everything, remove no cuts"):
        for cut in cuts:
            cut["approved"] = False
        st.rerun()

    if col3.button("↺ Reset All", help="Reset to default (all approved)"):
        for cut in cuts:
            cut["approved"] = True
        st.rerun()

    st.markdown("---")

    # Filter options
    st.subheader("Review Individual Cuts")

    filter_type = st.selectbox(
        "Show:",
        ["All cuts", "Silence only", "Off-mic speech only", "Filler words only", "Pauses only", "False starts only", "Approved only", "Rejected only"],
        index=0
    )

    # Filter cuts based on selection
    filtered_cuts = cuts
    if filter_type == "Silence only":
        filtered_cuts = [c for c in cuts if c.get("type") == "silence"]
    elif filter_type == "Off-mic speech only":
        filtered_cuts = [c for c in cuts if c.get("type") == "offmic"]
    elif filter_type == "Filler words only":
        filtered_cuts = [c for c in cuts if c.get("type") == "filler"]
    elif filter_type == "Pauses only":
        filtered_cuts = [c for c in cuts if c.get("type") == "pause"]
    elif filter_type == "False starts only":
        filtered_cuts = [c for c in cuts if c.get("type") == "false_start"]
    elif filter_type == "Approved only":
        filtered_cuts = [c for c in cuts if c.get("approved") is True]
    elif filter_type == "Rejected only":
        filtered_cuts = [c for c in cuts if c.get("approved") is False]

    st.caption(f"Showing {len(filtered_cuts)} of {len(cuts)} cuts")

    # Table header
    header_cols = st.columns([2, 3, 2, 1, 1])
    header_cols[0].markdown("**Time**")
    header_cols[1].markdown("**What was detected**")
    header_cols[2].markdown("**Status**")
    header_cols[3].markdown("**Keep**")
    header_cols[4].markdown("**Cut**")

    st.markdown("---")

    # Individual cuts - paginate for performance
    PAGE_SIZE = 50
    total_pages = (len(filtered_cuts) + PAGE_SIZE - 1) // PAGE_SIZE

    if total_pages > 1:
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1) - 1
    else:
        page = 0

    start_idx = page * PAGE_SIZE
    end_idx = min(start_idx + PAGE_SIZE, len(filtered_cuts))

    for display_idx, cut in enumerate(filtered_cuts[start_idx:end_idx]):
        # Find original index for button keys
        original_idx = cuts.index(cut)

        cols = st.columns([2, 3, 2, 1, 1])

        # Time range
        time_str = format_time_range(cut["start"], cut["end"])
        cols[0].write(time_str)

        # What was detected
        reason = cut["reason"]
        cut_type = cut.get("type", "unknown")
        if cut_type == "silence":
            cols[1].write(f"🔇 {reason}")
        elif cut_type == "offmic":
            cols[1].write(f"🎤 {reason}")
        elif cut_type == "filler":
            cols[1].write(f"💬 {reason}")
        elif cut_type == "pause":
            cols[1].write(f"⏸️ {reason}")
        elif cut_type == "false_start":
            cols[1].write(f"↩️ {reason}")
        else:
            cols[1].write(reason)

        # Status
        if cut.get("approved") is True:
            cols[2].write("🔴 Will be CUT")
        else:
            cols[2].write("🟢 Will be KEPT")

        # Keep button (reject cut)
        if cols[3].button("Keep", key=f"keep_{original_idx}", help="Keep this in video"):
            cuts[original_idx]["approved"] = False
            st.rerun()

        # Cut button (approve cut)
        if cols[4].button("Cut", key=f"cut_{original_idx}", type="primary" if not cut.get("approved") else "secondary", help="Remove this from video"):
            cuts[original_idx]["approved"] = True
            st.rerun()

    if total_pages > 1:
        st.caption(f"Page {page + 1} of {total_pages}")

    return cuts


def render_export_section(cuts, video_path, duration, settings, original_filename):
    """Render the export section."""
    st.markdown("---")
    st.header("4. Export to Premiere Pro")

    approved_cuts = [c for c in cuts if c.get("approved") is True]

    if not approved_cuts:
        st.warning("No cuts approved. Either approve some cuts above, or there's nothing to export.")
        return

    # Summary of what will happen
    total_cut_time = sum(c["duration"] for c in approved_cuts)
    remaining_time = duration - total_cut_time

    st.success(f"""
    **Ready to export!**
    - Original video: {format_timestamp(duration)}
    - Cutting: {format_timestamp(total_cut_time)} ({len(approved_cuts)} cuts)
    - Final length: ~{format_timestamp(remaining_time)}
    """)

    # Video file path input for proper linking
    st.markdown("---")
    st.subheader("Video File Location")
    st.markdown("**Enter the full path to your video file** so Premiere can find it automatically:")
    st.caption("Example: /Users/jay/Movies/my_interview.mp4")

    video_file_path = st.text_input(
        "Full path to your video file",
        value=f"/Users/jay/Movies/{original_filename}" if original_filename else "",
        help="The exact location of your video file on your computer"
    )

    if not video_file_path:
        st.warning("Please enter the path to your video file above for proper media linking.")
        return

    cut_manager = CutManager(padding=settings["padding"])
    cut_manager.add_cuts(approved_cuts)
    keep_segments = cut_manager.get_keep_segments(duration)

    # Use the user-provided path for export
    exporter = PremiereExporter(frame_rate=settings["frame_rate"])
    export_video_path = video_file_path

    st.subheader("Download Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Edited Sequence (Recommended)**")
        st.caption("Opens in Premiere with cuts already applied. Just review and export.")

        xml_content = exporter.create_fcpxml(
            video_path=export_video_path,
            keep_segments=keep_segments,
            video_duration=duration,
            project_name="Interview Edit",
            original_filename=os.path.basename(export_video_path)
        )

        st.download_button(
            label="Download XML (Edited)",
            data=xml_content,
            file_name="interview_edit.xml",
            mime="application/xml",
            type="primary"
        )

    with col2:
        st.markdown("**Markers Only**")
        st.caption("Full video with markers at cut points. You make the cuts manually.")

        markers_xml = exporter.create_markers_xml(
            video_path=export_video_path,
            cuts=approved_cuts,
            video_duration=duration,
            project_name="Interview Markers",
            original_filename=os.path.basename(export_video_path)
        )

        st.download_button(
            label="Download XML (Markers)",
            data=markers_xml,
            file_name="interview_markers.xml",
            mime="application/xml"
        )

    with col3:
        st.markdown("**EDL Format**")
        st.caption("Works with other editors like DaVinci Resolve, Avid, etc.")

        edl_content = exporter.create_edl(
            keep_segments=keep_segments,
            project_name="Interview Edit",
            original_filename=os.path.basename(export_video_path)
        )

        st.download_button(
            label="Download EDL",
            data=edl_content,
            file_name="interview_edit.edl",
            mime="text/plain"
        )

    # CSV fallback option
    st.markdown("---")
    st.markdown("**CSV Timecode List** (works with any editor)")
    st.caption("Simple spreadsheet with all timecodes - use for manual editing or if XML fails to import")

    csv_content = exporter.create_csv(
        keep_segments=keep_segments,
        cuts=approved_cuts,
        original_filename=os.path.basename(export_video_path)
    )

    st.download_button(
        label="Download CSV",
        data=csv_content,
        file_name="interview_timecodes.csv",
        mime="text/csv"
    )

    # Instructions
    st.markdown("---")
    st.subheader("How to Import into Premiere Pro")
    st.markdown(f"""
    1. **Import the XML file**: File → Import → select the downloaded XML
    2. **Relink the media**: Premiere will ask you to locate your video file
       - Navigate to where **{original_filename}** is stored on your computer
       - Select it and click "OK"
    3. **Review the sequence**: Your edited timeline will appear with all cuts applied
    4. **Fine-tune if needed**: Adjust any cuts that need tweaking
    5. **Export**: File → Export → Media
    """)


def render_transcript():
    """Render the transcript section."""
    if st.session_state.transcription:
        with st.expander("View Full Transcript"):
            text = st.session_state.transcription.get("text", "")
            st.text_area("Transcript", value=text, height=300, disabled=True)


def parse_timestamp(ts_string: str) -> float:
    """Parse a timestamp string to seconds.

    Accepts:
        "MM:SS"       -> float seconds
        "HH:MM:SS"    -> float seconds
        raw number    -> float seconds directly
    """
    ts_string = ts_string.strip()
    parts = ts_string.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(ts_string)


def render_thumbnail_section(video_path):
    """Render the YouTube thumbnail generator section."""
    st.markdown("---")
    st.header("5. YouTube Thumbnail")

    with st.expander("Generate Thumbnail", expanded=False):
        gen = ThumbnailGenerator(video_path)

        # ----------------------------------------------------------
        # Frame selection
        # ----------------------------------------------------------
        st.subheader("Select a Frame")

        if not st.session_state.thumbnail_frames:
            if st.button("Extract frames from video"):
                with st.spinner("Extracting frames..."):
                    frames = gen.extract_frame_grid(num_frames=20)
                    st.session_state.thumbnail_frames = frames
                st.rerun()

        # Display frame grid
        if st.session_state.thumbnail_frames:
            frames = st.session_state.thumbnail_frames
            cols_per_row = 5
            for row_start in range(0, len(frames), cols_per_row):
                row_frames = frames[row_start : row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for col_idx, (ts, img) in enumerate(row_frames):
                    with cols[col_idx]:
                        st.image(img, caption=format_timestamp(ts), use_container_width=True)
                        frame_idx = row_start + col_idx
                        if st.button("Select", key=f"thumb_select_{frame_idx}"):
                            st.session_state.selected_frame = img
                            st.session_state.selected_timestamp = ts
                            st.rerun()

            if st.button("Extract more frames"):
                batch = len(frames) // 20
                with st.spinner("Extracting more frames..."):
                    new_frames = gen.extract_frame_grid(num_frames=20, batch=batch)
                    st.session_state.thumbnail_frames = frames + new_frames
                st.rerun()

        # Manual timestamp input
        st.markdown("**Or enter a specific timestamp:**")
        ts_col1, ts_col2 = st.columns([3, 1])
        with ts_col1:
            manual_ts = st.text_input(
                "Timestamp (MM:SS, HH:MM:SS, or seconds)",
                placeholder="1:30",
                key="thumb_manual_ts",
            )
        with ts_col2:
            if st.button("Extract this frame"):
                if manual_ts:
                    try:
                        ts_seconds = parse_timestamp(manual_ts)
                        with st.spinner("Extracting frame..."):
                            frame = gen.extract_frame(ts_seconds)
                        st.session_state.selected_frame = frame
                        st.session_state.selected_timestamp = ts_seconds
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not extract frame: {e}")

        # Show selected frame
        if st.session_state.selected_frame is not None:
            st.success(
                f"Frame selected at {format_timestamp(st.session_state.selected_timestamp)}"
            )

            # ----------------------------------------------------------
            # Text overlay controls
            # ----------------------------------------------------------
            st.subheader("Text Overlay")

            line1 = st.text_input("Main text (line 1)", key="thumb_line1")
            line2 = st.text_input("Secondary text (line 2, optional)", key="thumb_line2")

            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
            with ctrl_col1:
                font_name = st.selectbox(
                    "Font",
                    options=list(
                        ["Impact", "Arial Black", "Arial Bold", "Helvetica", "Futura", "Gill Sans"]
                    ),
                    index=0,
                    key="thumb_font",
                )
                font_size_1 = st.slider(
                    "Main text size", 24, 144, 72, key="thumb_fs1"
                )
                font_size_2 = st.slider(
                    "Secondary text size", 16, 96, 48, key="thumb_fs2"
                )

            with ctrl_col2:
                text_color = st.color_picker(
                    "Text color", "#FFFFFF", key="thumb_text_color"
                )
                outline_color = st.color_picker(
                    "Outline color", "#000000", key="thumb_outline_color"
                )
                outline_thickness = st.slider(
                    "Outline thickness (px)", 0, 10, 3, key="thumb_outline"
                )

            with ctrl_col3:
                position = st.selectbox(
                    "Text position",
                    options=[
                        "Bottom center",
                        "Bottom left",
                        "Top center",
                        "Top left",
                        "Center",
                    ],
                    index=0,
                    key="thumb_position",
                )
                y_offset = st.slider(
                    "Vertical offset (px)",
                    min_value=-300,
                    max_value=300,
                    value=-60,
                    step=5,
                    key="thumb_y_offset",
                    help="Negative = move text up, positive = move text down",
                )
                enhance = st.checkbox(
                    "Contrast boost", value=True, key="thumb_enhance"
                )

            # ----------------------------------------------------------
            # Background image (green screen replacement)
            # ----------------------------------------------------------
            st.subheader("Background Image")
            st.caption(
                "Since the video is on a green screen, you can replace the "
                "background with any image. Leave empty to keep the original frame."
            )

            bg_image = None
            bg_col1, bg_col2 = st.columns([1, 1])
            with bg_col1:
                bg_upload = st.file_uploader(
                    "Upload a background image",
                    type=["png", "jpg", "jpeg", "webp"],
                    key="thumb_bg_upload",
                )
                if bg_upload is not None:
                    bg_image = Image.open(bg_upload).convert("RGB")

            with bg_col2:
                bg_path = st.text_input(
                    "Or paste a file path",
                    placeholder="/Users/jay/Pictures/background.jpg",
                    key="thumb_bg_path",
                )
                if bg_path and not bg_upload:
                    bg_path = bg_path.strip().strip("'\"")
                    if os.path.isfile(bg_path):
                        bg_image = Image.open(bg_path).convert("RGB")
                    else:
                        st.error("Background file not found.")

            if bg_image is not None:
                st.image(bg_image, caption="Background preview", use_container_width=True)

            # Subject positioning (only when using green screen)
            subject_scale = 1.0
            subject_x_offset = 0
            subject_y_offset = 0
            if bg_image is not None:
                st.markdown("**Subject Position**")
                subj_col1, subj_col2, subj_col3 = st.columns(3)
                with subj_col1:
                    subject_scale = st.slider(
                        "Subject scale",
                        min_value=0.5,
                        max_value=2.0,
                        value=1.0,
                        step=0.05,
                        key="thumb_subj_scale",
                    )
                with subj_col2:
                    subject_x_offset = st.slider(
                        "Subject left/right",
                        min_value=-500,
                        max_value=500,
                        value=0,
                        step=10,
                        key="thumb_subj_x",
                        help="Negative = left, positive = right",
                    )
                with subj_col3:
                    subject_y_offset = st.slider(
                        "Subject up/down",
                        min_value=-300,
                        max_value=300,
                        value=0,
                        step=10,
                        key="thumb_subj_y",
                        help="Negative = up, positive = down",
                    )

            # ----------------------------------------------------------
            # Gradient overlay
            # ----------------------------------------------------------
            st.subheader("Gradient Overlay")
            grad_col1, grad_col2, grad_col3 = st.columns(3)
            with grad_col1:
                use_gradient = st.checkbox(
                    "Dark bottom gradient",
                    value=True,
                    key="thumb_gradient",
                    help="Adds a dark fade at the bottom so text pops",
                )
            with grad_col2:
                gradient_height = st.slider(
                    "Gradient height",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    key="thumb_grad_height",
                    help="Fraction of image covered by gradient",
                )
            with grad_col3:
                gradient_opacity = st.slider(
                    "Gradient darkness",
                    min_value=0,
                    max_value=255,
                    value=180,
                    step=5,
                    key="thumb_grad_opacity",
                    help="0 = invisible, 255 = solid black",
                )

            # ----------------------------------------------------------
            # Overlay images (logos, cutouts, icons)
            # ----------------------------------------------------------
            st.subheader("Overlay Images")
            st.caption("Add PNG cutouts, logos, or icons on top of the thumbnail.")

            overlay_images = []
            overlay_files = st.file_uploader(
                "Upload overlay images (PNG with transparency)",
                type=["png"],
                accept_multiple_files=True,
                key="thumb_overlays",
            )
            if overlay_files:
                for ov_idx, ov_file in enumerate(overlay_files):
                    ov_img = Image.open(ov_file).convert("RGBA")
                    st.markdown(f"**Overlay {ov_idx + 1}: {ov_file.name}**")
                    ov_col1, ov_col2, ov_col3 = st.columns(3)
                    with ov_col1:
                        ov_x = st.slider(
                            "X position",
                            min_value=-200,
                            max_value=THUMBNAIL_WIDTH,
                            value=THUMBNAIL_WIDTH - ov_img.width - 40,
                            step=10,
                            key=f"ov_x_{ov_idx}",
                        )
                    with ov_col2:
                        ov_y = st.slider(
                            "Y position",
                            min_value=-200,
                            max_value=THUMBNAIL_HEIGHT,
                            value=40,
                            step=10,
                            key=f"ov_y_{ov_idx}",
                        )
                    with ov_col3:
                        ov_scale = st.slider(
                            "Scale",
                            min_value=0.1,
                            max_value=3.0,
                            value=0.5,
                            step=0.05,
                            key=f"ov_s_{ov_idx}",
                        )
                    overlay_images.append(
                        {"image": ov_img, "x": ov_x, "y": ov_y, "scale": ov_scale}
                    )

            # ----------------------------------------------------------
            # Live preview
            # ----------------------------------------------------------
            st.subheader("Preview")

            preview = ThumbnailGenerator.compose_thumbnail(
                base_image=st.session_state.selected_frame.copy(),
                line1=line1,
                line2=line2,
                font_name=font_name,
                font_size_1=font_size_1,
                font_size_2=font_size_2,
                text_color=text_color,
                outline_color=outline_color,
                position=position,
                outline_thickness=outline_thickness,
                enhance=enhance,
                background_image=bg_image,
                y_offset=y_offset,
                gradient=use_gradient,
                gradient_height=gradient_height,
                gradient_opacity=gradient_opacity,
                overlay_images=overlay_images if overlay_images else None,
                subject_scale=subject_scale,
                subject_x_offset=subject_x_offset,
                subject_y_offset=subject_y_offset,
            )

            st.session_state.thumbnail_preview = preview
            st.image(preview, caption="1280 x 720 preview", use_container_width=True)

            # ----------------------------------------------------------
            # Download
            # ----------------------------------------------------------
            st.subheader("Download")
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                jpeg_bytes = ThumbnailGenerator.export_jpeg(preview, quality=95)
                st.download_button(
                    label="Download JPEG",
                    data=jpeg_bytes,
                    file_name="thumbnail.jpg",
                    mime="image/jpeg",
                    type="primary",
                )
            with dl_col2:
                png_bytes = ThumbnailGenerator.export_png(preview)
                st.download_button(
                    label="Download PNG",
                    data=png_bytes,
                    file_name="thumbnail.png",
                    mime="image/png",
                )


def main():
    """Main application."""
    init_session_state()

    st.title("✂️ Interview Auto-Editor")
    st.markdown("**Automatically find and remove pauses, filler words, and dead air from your interview videos.**")

    # Sidebar settings
    settings = render_sidebar()

    # Main content
    st.header("1. Select Your Video")

    local_path = st.text_input(
        "Paste the full path to your video file",
        placeholder="/Users/jay/Movies/my_interview.mp4",
        help="Supports MP4, MOV, AVI, MKV, WebM. The file stays on your machine — nothing is uploaded."
    )

    video_path = None

    if local_path:
        local_path = local_path.strip().strip("'\"")
        if os.path.isfile(local_path):
            video_path = local_path
            st.session_state.video_path = video_path
            st.session_state.original_filename = os.path.basename(local_path)
            st.success(f"✓ Found: **{os.path.basename(local_path)}**")
        else:
            st.error(f"File not found: {local_path}")

    if not video_path:
        with st.expander("Or drag and drop a smaller file"):
            uploaded_file = st.file_uploader(
                "Drag and drop your video file here",
                type=["mp4", "mov", "avi", "mkv", "webm"],
                help="For large files (1 GB+), use the path input above instead."
            )
            if uploaded_file is not None:
                # Only copy to disk once per file
                if (
                    st.session_state.video_path is None
                    or st.session_state.original_filename != uploaded_file.name
                    or not os.path.exists(st.session_state.video_path)
                ):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                        uploaded_file.seek(0)
                        shutil.copyfileobj(uploaded_file, tmp)
                        st.session_state.video_path = tmp.name
                        st.session_state.original_filename = uploaded_file.name

                video_path = st.session_state.video_path
                st.success(f"✓ Uploaded: **{uploaded_file.name}**")

    if video_path:
        # Process button
        st.header("2. Analyze Video")
        st.markdown("Click below to scan your video for silence, filler words, and pauses.")

        # Check for API key before allowing analysis
        if not settings.get("api_key"):
            st.warning("Please enter your OpenAI API key in the sidebar to analyze videos.")
            st.info("Get your API key at [platform.openai.com](https://platform.openai.com/api-keys). Cost: ~$0.006 per minute of audio.")
            analyze_disabled = True
        else:
            analyze_disabled = False

        if st.button("🔍 Analyze Video", type="primary", disabled=analyze_disabled):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(value, text):
                progress_bar.progress(value)
                status_text.write(text)

            try:
                with st.spinner("Processing video... This may take a while for long videos."):
                    cuts = process_video(video_path, settings, update_progress)
                    st.session_state.cuts = cuts
                    st.session_state.processed = True

                progress_bar.empty()
                status_text.empty()
                st.rerun()
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Error during processing: {str(e)}")
                if "api_key" in str(e).lower() or "authentication" in str(e).lower():
                    st.error("Please check that your OpenAI API key is valid.")

        # Show results if processed
        if st.session_state.processed and st.session_state.cuts:
            st.header("3. Review Detected Cuts")
            st.markdown("Below are all the sections the tool wants to remove. Review and adjust as needed.")

            # Cuts list
            st.session_state.cuts = render_cuts_list(st.session_state.cuts, settings)

            # Transcript
            render_transcript()

            # Export section
            render_export_section(
                st.session_state.cuts,
                video_path,
                st.session_state.video_duration,
                settings,
                st.session_state.original_filename
            )

        # Thumbnail generation — available as soon as a video is loaded
        render_thumbnail_section(video_path)

    elif not local_path:
        st.info("👆 Paste the path to your video file to get started")

        with st.expander("How it works"):
            st.markdown("""
            ### Step 1: Select your video
            Paste the file path, or drag-and-drop a smaller file.

            ### Step 2: Analyze
            The tool will automatically:
            - Transcribe the audio using AI (Whisper)
            - Find silence gaps
            - Detect filler words ("um", "uh", "like", "you know", etc.)
            - Identify awkward pauses

            ### Step 3: Review
            Go through the detected cuts and decide:
            - **Keep** sections you want to preserve
            - **Cut** sections you want removed

            ### Step 4: Export
            Download a Premiere Pro XML file with your edits, then import and export your final video!
            """)


if __name__ == "__main__":
    main()
