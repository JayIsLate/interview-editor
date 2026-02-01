"""
YouTube Thumbnail Generator

Extracts frames from video via ffmpeg/ffprobe and composites
YouTube-optimized text overlays using Pillow.
"""

import subprocess
import json
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance


FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"

THUMBNAIL_WIDTH = 1280
THUMBNAIL_HEIGHT = 720

# macOS system font paths
FONT_MAP = {
    "Impact": "/System/Library/Fonts/Supplemental/Impact.ttf",
    "Arial Black": "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "Arial Bold": "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "Helvetica": "/System/Library/Fonts/Helvetica.ttc",
    "Futura": "/System/Library/Fonts/Supplemental/Futura.ttc",
    "Gill Sans": "/System/Library/Fonts/Supplemental/GillSans.ttc",
}


class ThumbnailGenerator:
    """Generates YouTube thumbnails from video frames with text overlays."""

    def __init__(self, video_path: str):
        self.video_path = video_path

    def get_video_duration(self) -> float:
        """Get video duration in seconds using ffprobe."""
        result = subprocess.run(
            [
                FFPROBE,
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                self.video_path,
            ],
            capture_output=True,
            text=True,
        )
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])

    def extract_frame(self, timestamp: float) -> Image.Image:
        """Extract a single frame at the given timestamp as a PIL Image."""
        result = subprocess.run(
            [
                FFMPEG,
                "-ss", str(timestamp),
                "-i", self.video_path,
                "-vframes", "1",
                "-f", "image2pipe",
                "-vcodec", "png",
                "-",
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to extract frame at {timestamp}s: "
                f"{result.stderr.decode(errors='replace')}"
            )
        return Image.open(io.BytesIO(result.stdout)).convert("RGB")

    def extract_frame_grid(self, num_frames: int = 20, batch: int = 0) -> list:
        """Extract evenly-spaced frames, skipping first/last 2%.

        Args:
            num_frames: How many frames to extract.
            batch: Batch number (0, 1, 2, ...).  Each batch offsets the
                   sample points so repeated calls return different frames.

        Returns:
            List of (timestamp, PIL.Image) tuples.
        """
        duration = self.get_video_duration()
        start = duration * 0.02
        end = duration * 0.98
        span = end - start

        if num_frames < 1:
            num_frames = 1

        step = span / num_frames
        # Shift by half-step fraction per batch so each batch is unique
        batch_offset = (step / (batch + 1)) * 0.5 if batch > 0 else 0
        frames = []
        for i in range(num_frames):
            ts = start + step * (i + 0.5) + batch_offset
            if ts > end:
                ts = end - 0.1
            try:
                img = self.extract_frame(ts)
                frames.append((ts, img))
            except RuntimeError:
                continue
        return frames

    # ------------------------------------------------------------------
    # Green screen removal
    # ------------------------------------------------------------------

    @staticmethod
    def remove_greenscreen(
        img: Image.Image,
        hue_range: tuple = (35, 85),
        sat_min: int = 40,
        val_min: int = 40,
        spill_strength: float = 0.6,
    ) -> Image.Image:
        """Chroma-key green pixels to transparency.

        Converts to HSV, masks pixels whose hue falls within the green
        range (and have enough saturation/value to be real green, not
        shadow), then returns an RGBA image with green areas transparent.

        Args:
            img: RGB source image.
            hue_range: (low, high) hue in 0-180 OpenCV-style scale.
                       Default (35, 85) covers typical green screens.
            sat_min: Minimum saturation (0-255) to count as green.
            val_min: Minimum value/brightness (0-255) to count as green.
            spill_strength: How aggressively to suppress green spill on
                            edge pixels (0.0 = off, 1.0 = full).

        Returns:
            RGBA PIL Image with green keyed out.
        """
        arr = np.array(img.convert("RGB"), dtype=np.float32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        # Convert to HSV manually (avoids OpenCV dependency)
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        diff = maxc - minc

        # Hue (0-180 scale, matching OpenCV convention)
        hue = np.zeros_like(maxc)
        mask_r = (maxc == r) & (diff > 0)
        mask_g = (maxc == g) & (diff > 0)
        mask_b = (maxc == b) & (diff > 0)
        hue[mask_r] = (30.0 * ((g[mask_r] - b[mask_r]) / diff[mask_r])) % 180
        hue[mask_g] = 30.0 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 60
        hue[mask_b] = 30.0 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 120

        # Saturation (0-255 scale)
        sat = np.where(maxc > 0, (diff / maxc) * 255, 0)

        # Value (0-255 scale)
        val = maxc

        # Build green mask
        green_mask = (
            (hue >= hue_range[0])
            & (hue <= hue_range[1])
            & (sat >= sat_min)
            & (val >= val_min)
        )

        # Soft edge: dilate the mask boundary for smoother edges
        # Use a simple distance-based softness
        alpha = np.where(green_mask, 0, 255).astype(np.uint8)

        # Green spill suppression on non-masked pixels
        if spill_strength > 0:
            spill_mask = (~green_mask) & (g > r) & (g > b)
            if np.any(spill_mask):
                excess = (g[spill_mask] - np.maximum(r[spill_mask], b[spill_mask])) * spill_strength
                arr[spill_mask, 1] = g[spill_mask] - excess

        result = np.dstack([arr.astype(np.uint8), alpha])
        return Image.fromarray(result, "RGBA")

    # ------------------------------------------------------------------
    # Compositing
    # ------------------------------------------------------------------

    @staticmethod
    def compose_thumbnail(
        base_image: Image.Image,
        line1: str = "",
        line2: str = "",
        font_name: str = "Impact",
        font_size_1: int = 72,
        font_size_2: int = 48,
        text_color: str = "#FFFFFF",
        outline_color: str = "#000000",
        position: str = "Bottom center",
        outline_thickness: int = 3,
        enhance: bool = True,
        background_image: Image.Image = None,
        y_offset: int = 0,
        gradient: bool = False,
        gradient_height: float = 0.5,
        gradient_opacity: int = 180,
        overlay_images: list = None,
        subject_scale: float = 1.0,
        subject_x_offset: int = 0,
        subject_y_offset: int = 0,
    ) -> Image.Image:
        """Composite text onto a base image and return a 1280x720 thumbnail.

        If *background_image* is provided the green screen in *base_image*
        is keyed out and the subject is composited onto the background.

        Args:
            base_image: Source frame (may contain green screen).
            line1: Main text line.
            line2: Secondary text line (optional).
            font_name: Key into FONT_MAP.
            font_size_1: Font size for line 1.
            font_size_2: Font size for line 2.
            text_color: Hex colour for text fill.
            outline_color: Hex colour for text outline.
            position: One of "Bottom center", "Bottom left", "Top center",
                      "Top left", "Center".
            outline_thickness: Stroke width in pixels.
            enhance: Whether to apply contrast/saturation/sharpness boost.
            background_image: Optional replacement background. When set,
                              green pixels in *base_image* are keyed out
                              and the subject is placed on this background.
            y_offset: Vertical pixel offset for text. Negative = up.
            gradient: Whether to draw a dark bottom gradient behind text.
            gradient_height: Fraction of the image height the gradient
                             covers (0.0-1.0). Default 0.5.
            gradient_opacity: Peak opacity at the very bottom (0-255).
            overlay_images: List of dicts with keys ``image`` (RGBA Image),
                            ``x`` (int), ``y`` (int), ``scale`` (float).
            subject_scale: Scale factor for the keyed subject (1.0 = no
                           change). Only used with *background_image*.
            subject_x_offset: Horizontal pixel offset for the keyed subject.
            subject_y_offset: Vertical pixel offset for the keyed subject.

        Returns:
            Composited 1280x720 PIL Image.
        """
        if background_image is not None:
            bg = ThumbnailGenerator._fit_to_thumbnail(background_image.convert("RGB"))
            fg = ThumbnailGenerator._fit_to_thumbnail(base_image)
            fg_rgba = ThumbnailGenerator.remove_greenscreen(fg)

            # Scale and reposition the subject
            if subject_scale != 1.0:
                new_w = int(fg_rgba.width * subject_scale)
                new_h = int(fg_rgba.height * subject_scale)
                fg_rgba = fg_rgba.resize((new_w, new_h), Image.LANCZOS)

            # Paste centred, then apply offsets
            paste_x = (THUMBNAIL_WIDTH - fg_rgba.width) // 2 + subject_x_offset
            paste_y = (THUMBNAIL_HEIGHT - fg_rgba.height) // 2 + subject_y_offset
            bg.paste(fg_rgba, (paste_x, paste_y), fg_rgba)
            img = bg
        else:
            img = ThumbnailGenerator._fit_to_thumbnail(base_image)

        if enhance:
            img = ThumbnailGenerator._enhance_for_youtube(img)

        # Bottom gradient overlay
        if gradient:
            img = ThumbnailGenerator._draw_gradient(
                img, gradient_height, gradient_opacity
            )

        # Extra overlay images (logos, cutouts, etc.)
        if overlay_images:
            for ov in overlay_images:
                img = ThumbnailGenerator._paste_overlay(
                    img,
                    ov["image"],
                    ov.get("x", 0),
                    ov.get("y", 0),
                    ov.get("scale", 1.0),
                )

        if line1 or line2:
            img = ThumbnailGenerator._draw_text_with_outline(
                img,
                line1=line1,
                line2=line2,
                font_name=font_name,
                font_size_1=font_size_1,
                font_size_2=font_size_2,
                text_color=text_color,
                outline_color=outline_color,
                position=position,
                outline_thickness=outline_thickness,
                y_offset=y_offset,
            )

        return img

    # ------------------------------------------------------------------
    # Gradient & overlay helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_gradient(
        img: Image.Image, height_frac: float = 0.5, opacity: int = 180
    ) -> Image.Image:
        """Draw a dark gradient from the bottom up.

        Args:
            img: Base RGB image.
            height_frac: How much of the image height the gradient covers.
            opacity: Peak alpha at the very bottom (0-255).
        """
        grad = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(grad)
        grad_px = int(THUMBNAIL_HEIGHT * height_frac)
        start_y = THUMBNAIL_HEIGHT - grad_px
        for y in range(grad_px):
            frac = y / grad_px  # 0 at top of gradient, 1 at bottom
            a = int(frac * opacity)
            draw.line([(0, start_y + y), (THUMBNAIL_WIDTH, start_y + y)], fill=(0, 0, 0, a))
        img = img.convert("RGBA")
        img = Image.alpha_composite(img, grad)
        return img.convert("RGB")

    @staticmethod
    def _paste_overlay(
        img: Image.Image,
        overlay: Image.Image,
        x: int,
        y: int,
        scale: float = 1.0,
    ) -> Image.Image:
        """Paste an RGBA overlay onto the image at (x, y) with optional scale."""
        ov = overlay.convert("RGBA")
        if scale != 1.0:
            new_w = max(1, int(ov.width * scale))
            new_h = max(1, int(ov.height * scale))
            ov = ov.resize((new_w, new_h), Image.LANCZOS)
        img = img.convert("RGBA")
        img.paste(ov, (x, y), ov)
        return img.convert("RGB")

    @staticmethod
    def _fit_to_thumbnail(img: Image.Image) -> Image.Image:
        """Resize and center-crop to 1280x720."""
        target_ratio = THUMBNAIL_WIDTH / THUMBNAIL_HEIGHT
        src_ratio = img.width / img.height

        if src_ratio > target_ratio:
            # Wider than target — fit height, crop width
            new_height = THUMBNAIL_HEIGHT
            new_width = int(img.width * (THUMBNAIL_HEIGHT / img.height))
        else:
            # Taller than target — fit width, crop height
            new_width = THUMBNAIL_WIDTH
            new_height = int(img.height * (THUMBNAIL_WIDTH / img.width))

        img = img.resize((new_width, new_height), Image.LANCZOS)

        # Center crop
        left = (new_width - THUMBNAIL_WIDTH) // 2
        top = (new_height - THUMBNAIL_HEIGHT) // 2
        img = img.crop((left, top, left + THUMBNAIL_WIDTH, top + THUMBNAIL_HEIGHT))
        return img

    @staticmethod
    def _enhance_for_youtube(img: Image.Image) -> Image.Image:
        """Boost contrast +15%, saturation +15%, sharpness +10%."""
        img = ImageEnhance.Contrast(img).enhance(1.15)
        img = ImageEnhance.Color(img).enhance(1.15)
        img = ImageEnhance.Sharpness(img).enhance(1.10)
        return img

    @staticmethod
    def _draw_text_with_outline(
        img: Image.Image,
        line1: str,
        line2: str,
        font_name: str,
        font_size_1: int,
        font_size_2: int,
        text_color: str,
        outline_color: str,
        position: str,
        outline_thickness: int,
        y_offset: int = 0,
    ) -> Image.Image:
        """Draw outlined text on the image."""
        draw = ImageDraw.Draw(img)

        font1 = ThumbnailGenerator._load_font(font_name, font_size_1)
        font2 = ThumbnailGenerator._load_font(font_name, font_size_2)

        lines = []
        if line1:
            bbox1 = draw.textbbox((0, 0), line1, font=font1)
            lines.append((line1, font1, bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]))
        if line2:
            bbox2 = draw.textbbox((0, 0), line2, font=font2)
            lines.append((line2, font2, bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]))

        if not lines:
            return img

        line_gap = 10
        total_height = sum(h for _, _, _, h in lines) + line_gap * (len(lines) - 1)
        max_width = max(w for _, _, w, _ in lines)

        x, y = ThumbnailGenerator._calculate_text_position(
            position, max_width, total_height
        )
        y += y_offset

        current_y = y
        for text, font, w, h in lines:
            # Horizontal alignment per line
            if "center" in position.lower():
                lx = (THUMBNAIL_WIDTH - w) // 2
            else:
                lx = x

            draw.text(
                (lx, current_y),
                text,
                font=font,
                fill=text_color,
                stroke_width=outline_thickness,
                stroke_fill=outline_color,
            )
            current_y += h + line_gap

        return img

    @staticmethod
    def _calculate_text_position(
        position: str, block_width: int, block_height: int
    ) -> tuple:
        """Return (x, y) for the text block origin.

        Positions: Bottom center, Bottom left, Top center, Top left, Center.
        """
        margin = 40

        if position == "Bottom center":
            x = (THUMBNAIL_WIDTH - block_width) // 2
            y = THUMBNAIL_HEIGHT - block_height - margin
        elif position == "Bottom left":
            x = margin
            y = THUMBNAIL_HEIGHT - block_height - margin
        elif position == "Top center":
            x = (THUMBNAIL_WIDTH - block_width) // 2
            y = margin
        elif position == "Top left":
            x = margin
            y = margin
        elif position == "Center":
            x = (THUMBNAIL_WIDTH - block_width) // 2
            y = (THUMBNAIL_HEIGHT - block_height) // 2
        else:
            x = (THUMBNAIL_WIDTH - block_width) // 2
            y = THUMBNAIL_HEIGHT - block_height - margin

        return x, y

    @staticmethod
    def _load_font(font_name: str, size: int) -> ImageFont.FreeTypeFont:
        """Load a macOS system font, falling back to Pillow default."""
        path = FONT_MAP.get(font_name)
        if path:
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
        # Fallback
        try:
            return ImageFont.truetype("Arial", size)
        except OSError:
            return ImageFont.load_default()

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    @staticmethod
    def export_jpeg(img: Image.Image, quality: int = 95) -> bytes:
        """Serialize a PIL Image to JPEG bytes."""
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    @staticmethod
    def export_png(img: Image.Image) -> bytes:
        """Serialize a PIL Image to PNG bytes."""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
