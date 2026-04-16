"""
youtube_route.py — Add this router to your main.py

Usage in main.py:
    from youtube_route import youtube_router
    app.include_router(youtube_router)

Requirements (add to requirements.txt):
    yt-dlp>=2024.1.0
    groq>=0.9.0
"""

import io
import os
import re
import json
import tempfile
import subprocess
import textwrap
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
import imageio_ffmpeg

# Tell yt-dlp where to find ffmpeg
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

youtube_router = APIRouter(tags=["YouTube"])


# ── Request model ─────────────────────────────────────────────────────────────

class YouTubePDFRequest(BaseModel):
    url:     str
    api_key: str          # Groq API key for Whisper transcription


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_valid_youtube_url(url: str) -> bool:
    patterns = [
        r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[\w\-]+",
    ]
    return any(re.match(p, url.strip()) for p in patterns)


def extract_video_id(url: str) -> str:
    patterns = [
        r"youtube\.com/watch\?v=([\w\-]+)",
        r"youtu\.be/([\w\-]+)",
        r"youtube\.com/shorts/([\w\-]+)",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return "unknown"


def get_video_metadata(url: str) -> dict:
    """Use yt-dlp to fetch video title, channel, duration without downloading."""
    try:
        result = subprocess.run(
            [
                "yt-dlp", "--dump-json", "--no-playlist",
                "--user-agent", "Mozilla/5.0 ...",
                "--add-header", "Accept-Language:en-US,en;q=0.9",
                "--js-runtimes", "nodejs",   # ← add here too
                url
            ],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return {
                "title":    data.get("title", "YouTube Video"),
                "channel":  data.get("channel") or data.get("uploader", "Unknown"),
                "duration": data.get("duration_string") or str(data.get("duration", "?")),
                "upload_date": data.get("upload_date", ""),
                "view_count": data.get("view_count", 0),
                "description": (data.get("description") or "")[:500],
            }
    except Exception:
        pass
    return {"title": "YouTube Video", "channel": "Unknown", "duration": "?",
            "upload_date": "", "view_count": 0, "description": ""}


def download_audio(url: str, output_path: str) -> str:
    """Download audio as MP3 using yt-dlp. Returns path to file."""
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--format", "bestaudio[ext=m4a]/bestaudio/best",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "5",
        "--max-filesize", "50m",
        "--ffmpeg-location", FFMPEG_PATH,
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "--add-header", "Accept-Language:en-US,en;q=0.9",
        "--sleep-interval", "2",
        "--max-sleep-interval", "5",
        "--js-runtimes", "nodejs",   # ← add here
        "--output", output_path,
        "--quiet",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr[:400]}")

    # yt-dlp appends .mp3 automatically
    if os.path.exists(output_path):
        return output_path
    mp3_path = output_path + ".mp3"
    if os.path.exists(mp3_path):
        return mp3_path
    raise RuntimeError("Audio file not found after download.")


def transcribe_with_groq(audio_path: str, api_key: str) -> dict:
    """
    Call Groq's Whisper endpoint.
    Returns {"text": "...", "segments": [...]} or just {"text": "..."}.
    """
    import httpx

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    # Groq Whisper: max 25 MB per request
    if len(audio_bytes) > 25 * 1024 * 1024:
        raise RuntimeError(
            "Audio file exceeds 25 MB. Try a shorter video (< ~45 minutes)."
        )

    with httpx.Client(timeout=300.0) as client:
        resp = client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (os.path.basename(audio_path), audio_bytes, "audio/mpeg")},
            data={
                "model":            "whisper-large-v3",
                "response_format":  "verbose_json",   # gives segments + timestamps
                "language":         "en",
                "temperature":      "0",
            },
        )

    if resp.status_code != 200:
        body = resp.text[:300]
        raise RuntimeError(f"Groq Whisper error {resp.status_code}: {body}")

    return resp.json()


# ── PDF builder ───────────────────────────────────────────────────────────────

def build_transcript_pdf(meta: dict, transcription: dict) -> bytes:
    buffer  = io.BytesIO()
    doc     = SimpleDocTemplate(
        buffer,
        pagesize        = A4,
        rightMargin     = 0.85 * inch,
        leftMargin      = 0.85 * inch,
        topMargin       = 0.9  * inch,
        bottomMargin    = 0.75 * inch,
        title           = meta["title"],
        author          = "QNA-AI YouTube Transcriber",
    )

    # ── Colour palette ────────────────────────────────────────────────────────
    RED     = HexColor("#e53e3e")
    DARK    = HexColor("#1a202c")
    MID     = HexColor("#4a5568")
    LIGHT   = HexColor("#718096")
    RULE    = HexColor("#e2e8f0")
    BG_META = HexColor("#f7fafc")

    # ── Styles ─────────────────────────────────────────────────────────────────
    ss = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "YTTitle", parent=ss["Title"],
        fontSize=20, leading=26, textColor=DARK,
        fontName="Helvetica-Bold", spaceAfter=4,
    )
    meta_label_style = ParagraphStyle(
        "MetaLabel", parent=ss["Normal"],
        fontSize=8, textColor=LIGHT, fontName="Helvetica",
        leading=12, spaceAfter=0,
    )
    meta_value_style = ParagraphStyle(
        "MetaValue", parent=ss["Normal"],
        fontSize=10, textColor=MID, fontName="Helvetica",
        leading=14, spaceAfter=0,
    )
    section_style = ParagraphStyle(
        "Section", parent=ss["Heading2"],
        fontSize=11, leading=16, textColor=RED,
        fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body", parent=ss["Normal"],
        fontSize=10, leading=16, textColor=DARK,
        fontName="Helvetica", spaceAfter=8, alignment=TA_JUSTIFY,
    )
    ts_label_style = ParagraphStyle(
        "TSLabel", parent=ss["Normal"],
        fontSize=8, leading=11, textColor=RED,
        fontName="Helvetica-Bold", spaceAfter=2,
    )
    ts_text_style = ParagraphStyle(
        "TSText", parent=ss["Normal"],
        fontSize=10, leading=15, textColor=DARK,
        fontName="Helvetica", spaceAfter=6,
    )
    desc_style = ParagraphStyle(
        "Desc", parent=ss["Normal"],
        fontSize=9, leading=14, textColor=MID,
        fontName="Helvetica", spaceAfter=4, alignment=TA_JUSTIFY,
    )
    footer_style = ParagraphStyle(
        "Footer", parent=ss["Normal"],
        fontSize=8, leading=11, textColor=LIGHT,
        fontName="Helvetica", alignment=TA_CENTER,
    )

    # ── Helpers ───────────────────────────────────────────────────────────────
    def fmt_seconds(s):
        s = int(s)
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        if h:
            return f"{h}:{m:02d}:{sec:02d}"
        return f"{m}:{sec:02d}"

    def safe(text: str) -> str:
        """Escape XML special chars for ReportLab."""
        return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # ── Upload date ───────────────────────────────────────────────────────────
    raw_date = meta.get("upload_date", "")
    pretty_date = ""
    if len(raw_date) == 8:
        try:
            pretty_date = datetime.strptime(raw_date, "%Y%m%d").strftime("%B %d, %Y")
        except Exception:
            pretty_date = raw_date

    # ── Build story ───────────────────────────────────────────────────────────
    story = []

    # ── YouTube "logo" strip ─────────────────────────────────────────────────
    logo_table = Table(
        [[Paragraph('<font color="#e53e3e" size="14"><b>▶ YouTube Transcript</b></font>', ss["Normal"])]],
        colWidths=["100%"]
    )
    logo_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), HexColor("#fff5f5")),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(logo_table)
    story.append(Spacer(1, 10))

    # Title
    story.append(Paragraph(safe(meta["title"]), title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=RED, spaceAfter=10))

    # Meta table
    vc = f"{meta.get('view_count', 0):,}" if meta.get("view_count") else "—"
    meta_rows = [
        [Paragraph("CHANNEL",     meta_label_style), Paragraph(safe(meta["channel"]),  meta_value_style),
         Paragraph("DURATION",    meta_label_style), Paragraph(safe(meta["duration"]), meta_value_style)],
        [Paragraph("UPLOADED",    meta_label_style), Paragraph(safe(pretty_date) or "—", meta_value_style),
         Paragraph("VIEWS",       meta_label_style), Paragraph(vc,                    meta_value_style)],
        [Paragraph("GENERATED BY",meta_label_style), Paragraph("QNA-AI · Groq Whisper", meta_value_style),
         Paragraph("DATE",        meta_label_style), Paragraph(datetime.now().strftime("%B %d, %Y"), meta_value_style)],
    ]
    cw = doc.width / 2
    meta_table = Table(meta_rows, colWidths=[cw * 0.28, cw * 0.72, cw * 0.28, cw * 0.72])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), BG_META),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # Description (if any)
    desc = (meta.get("description") or "").strip()
    if desc:
        story.append(Paragraph("VIDEO DESCRIPTION", section_style))
        story.append(Paragraph(safe(desc[:600] + ("…" if len(desc) > 600 else "")), desc_style))

    # ── Transcript section ────────────────────────────────────────────────────
    story.append(Paragraph("FULL TRANSCRIPT", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=RULE, spaceAfter=8))

    segments = transcription.get("segments", [])

    if segments:
        # Segmented transcript with timestamps
        for seg in segments:
            start = fmt_seconds(seg.get("start", 0))
            text  = (seg.get("text") or "").strip()
            if not text:
                continue
            story.append(Paragraph(f"[{start}]", ts_label_style))
            story.append(Paragraph(safe(text), ts_text_style))
    else:
        # Plain text — wrap into paragraphs of ~5 sentences each
        full_text = (transcription.get("text") or "").strip()
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        chunk_size = 5
        for i in range(0, len(sentences), chunk_size):
            chunk = " ".join(sentences[i: i + chunk_size]).strip()
            if chunk:
                story.append(Paragraph(safe(chunk), body_style))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=RULE))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Generated by QNA-AI · Transcribed with Groq Whisper · {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC",
        footer_style
    ))
    story.append(Paragraph(
        "This transcript is AI-generated and may contain errors. Always verify important content.",
        footer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ── Route ─────────────────────────────────────────────────────────────────────

@youtube_router.post("/convert/youtube-to-pdf")
async def youtube_to_pdf(req: YouTubePDFRequest):
    """
    1. Validate YouTube URL
    2. Fetch video metadata (title, channel, duration)
    3. Download audio via yt-dlp
    4. Transcribe with Groq Whisper
    5. Build & return a formatted PDF
    """
    url = req.url.strip()

    if not is_valid_youtube_url(url):
        raise HTTPException(400, "Invalid YouTube URL. Please provide a valid youtube.com or youtu.be link.")

    if not req.api_key or not req.api_key.startswith("gsk_"):
        raise HTTPException(400, "A valid Groq API key (starting with gsk_) is required for transcription.")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")

        # 1. Metadata
        try:
            meta = get_video_metadata(url)
        except Exception as e:
            raise HTTPException(500, f"Failed to fetch video metadata: {e}")

        # 2. Download audio
        try:
            audio_path = download_audio(url, audio_path)
        except RuntimeError as e:
            raise HTTPException(422, str(e))
        except Exception as e:
            raise HTTPException(500, f"Audio download failed: {e}")

        # 3. Transcribe
        try:
            transcription = transcribe_with_groq(audio_path, req.api_key)
        except RuntimeError as e:
            raise HTTPException(422, str(e))
        except Exception as e:
            raise HTTPException(500, f"Transcription failed: {e}")

        # 4. Build PDF
        try:
            pdf_bytes = build_transcript_pdf(meta, transcription)
        except Exception as e:
            raise HTTPException(500, f"PDF generation failed: {e}")

    safe_title = re.sub(r"[^\w\-]", "_", meta["title"])[:60]
    filename   = f"transcript_{safe_title}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
