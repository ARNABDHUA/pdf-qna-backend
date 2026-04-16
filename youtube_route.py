"""
youtube_route.py — YouTube → PDF via youtube-transcript-api

Usage in main.py:
    from youtube_route import youtube_router
    app.include_router(youtube_router)

Requirements (add to requirements.txt):
    youtube-transcript-api>=0.6.2
    httpx>=0.24.0
    reportlab>=4.0.0

No Groq key needed. No yt-dlp. No ffmpeg. No cookies.
Works for any YouTube video that has captions/subtitles (auto-generated or manual).
"""

import io
import re
import httpx
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle,
)
from reportlab.lib.colors import HexColor

youtube_router = APIRouter(tags=["YouTube"])


# ── Request model ─────────────────────────────────────────────────────────────

class YouTubePDFRequest(BaseModel):
    url: str
    # api_key no longer required — kept optional for frontend backward-compat
    api_key: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_valid_youtube_url(url: str) -> bool:
    return bool(re.match(
        r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[\w\-]+",
        url.strip()
    ))


def extract_video_id(url: str) -> str:
    for pattern in [
        r"youtube\.com/watch\?v=([\w\-]+)",
        r"youtu\.be/([\w\-]+)",
        r"youtube\.com/shorts/([\w\-]+)",
    ]:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return ""


async def get_video_metadata(video_id: str) -> dict:
    """
    Fetch video title, channel, duration via YouTube oEmbed (no API key needed).
    Falls back to safe defaults if unavailable.
    """
    default = {
        "title": "YouTube Video",
        "channel": "Unknown",
        "duration": "?",
        "upload_date": "",
        "view_count": 0,
        "description": "",
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://www.youtube.com/oembed",
                params={
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "format": "json",
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                default["title"]   = data.get("title",        default["title"])
                default["channel"] = data.get("author_name",  default["channel"])
    except Exception:
        pass
    return default


def fetch_transcript(video_id: str) -> list[dict]:
    """
    Fetch transcript segments. Tries English first, then any available language.
    Returns list of {"start": float, "duration": float, "text": str}.
    """
    try:
        # Try English (manual + auto-generated)
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "en-US", "en-GB"])
        return transcript
    except NoTranscriptFound:
        pass

    # Fall back to whatever is available and translate to English if possible
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        for t in transcript_list:
            try:
                if t.is_translatable:
                    return t.translate("en").fetch()
                return t.fetch()
            except Exception:
                continue
    except Exception:
        pass

    return []


def fmt_seconds(s: float) -> str:
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def safe_xml(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ── PDF builder ───────────────────────────────────────────────────────────────

def build_transcript_pdf(meta: dict, segments: list[dict]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.85 * inch,
        leftMargin=0.85 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.75 * inch,
        title=meta["title"],
        author="QNA-AI YouTube Transcriber",
    )

    # ── Colour palette ────────────────────────────────────────────────────────
    RED     = HexColor("#e53e3e")
    DARK    = HexColor("#1a202c")
    MID     = HexColor("#4a5568")
    LIGHT   = HexColor("#718096")
    RULE    = HexColor("#e2e8f0")
    BG_META = HexColor("#f7fafc")

    ss = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "YTTitle", parent=ss["Title"],
        fontSize=20, leading=26, textColor=DARK,
        fontName="Helvetica-Bold", spaceAfter=4,
    )
    meta_label_style = ParagraphStyle(
        "MetaLabel", parent=ss["Normal"],
        fontSize=8, textColor=LIGHT, fontName="Helvetica", leading=12,
    )
    meta_value_style = ParagraphStyle(
        "MetaValue", parent=ss["Normal"],
        fontSize=10, textColor=MID, fontName="Helvetica", leading=14,
    )
    section_style = ParagraphStyle(
        "Section", parent=ss["Heading2"],
        fontSize=11, leading=16, textColor=RED,
        fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=6,
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
    plain_style = ParagraphStyle(
        "Plain", parent=ss["Normal"],
        fontSize=10, leading=16, textColor=DARK,
        fontName="Helvetica", spaceAfter=8, alignment=TA_JUSTIFY,
    )
    footer_style = ParagraphStyle(
        "Footer", parent=ss["Normal"],
        fontSize=8, leading=11, textColor=LIGHT,
        fontName="Helvetica", alignment=TA_CENTER,
    )
    notice_style = ParagraphStyle(
        "Notice", parent=ss["Normal"],
        fontSize=9, leading=13, textColor=MID,
        fontName="Helvetica", alignment=TA_CENTER, spaceAfter=6,
    )

    story = []

    # ── Header banner ─────────────────────────────────────────────────────────
    logo_table = Table(
        [[Paragraph(
            '<font color="#e53e3e" size="14"><b>▶ YouTube Transcript</b></font>',
            ss["Normal"]
        )]],
        colWidths=["100%"],
    )
    logo_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), HexColor("#fff5f5")),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    story.append(logo_table)
    story.append(Spacer(1, 10))

    # Title
    story.append(Paragraph(safe_xml(meta["title"]), title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=RED, spaceAfter=10))

    # Meta table
    video_url = meta.get("url", f"https://www.youtube.com/watch?v={meta.get('video_id','')}")
    meta_rows = [
        [
            Paragraph("CHANNEL",      meta_label_style),
            Paragraph(safe_xml(meta["channel"]), meta_value_style),
            Paragraph("VIDEO ID",     meta_label_style),
            Paragraph(safe_xml(meta.get("video_id", "—")), meta_value_style),
        ],
        [
            Paragraph("SOURCE",       meta_label_style),
            Paragraph(
                f'<a href="{video_url}" color="#3b82f6">{safe_xml(video_url)}</a>',
                meta_value_style
            ),
            Paragraph("GENERATED BY", meta_label_style),
            Paragraph("QNA-AI · YouTube Transcript API", meta_value_style),
        ],
        [
            Paragraph("DATE",         meta_label_style),
            Paragraph(datetime.now().strftime("%B %d, %Y"), meta_value_style),
            Paragraph("SEGMENTS",     meta_label_style),
            Paragraph(str(len(segments)), meta_value_style),
        ],
    ]
    cw = doc.width / 2
    mt = Table(meta_rows, colWidths=[cw * 0.28, cw * 0.72, cw * 0.28, cw * 0.72])
    mt.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), BG_META),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(mt)
    story.append(Spacer(1, 12))

    # ── Transcript ────────────────────────────────────────────────────────────
    story.append(Paragraph("FULL TRANSCRIPT", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=RULE, spaceAfter=8))

    if segments:
        for seg in segments:
            start = fmt_seconds(seg.get("start", 0))
            text  = (seg.get("text") or "").strip()
            if not text:
                continue
            story.append(Paragraph(f"[{start}]", ts_label_style))
            story.append(Paragraph(safe_xml(text), ts_text_style))
    else:
        story.append(Paragraph(
            "No transcript segments were returned.", plain_style
        ))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=RULE))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Generated by QNA-AI · YouTube Transcript API · "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')} UTC",
        footer_style,
    ))
    story.append(Paragraph(
        "Transcripts are sourced from YouTube captions and may contain inaccuracies.",
        footer_style,
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ── Route ─────────────────────────────────────────────────────────────────────

@youtube_router.post("/convert/youtube-to-pdf")
async def youtube_to_pdf(req: YouTubePDFRequest):
    """
    1. Validate YouTube URL and extract video ID
    2. Fetch video metadata via oEmbed (no API key needed)
    3. Fetch transcript via youtube-transcript-api (captions must exist)
    4. Build and return a formatted timestamped PDF
    """
    url = req.url.strip()

    if not is_valid_youtube_url(url):
        raise HTTPException(
            400,
            "Invalid YouTube URL. Please provide a valid youtube.com or youtu.be link.",
        )

    video_id = extract_video_id(url)
    if not video_id:
        raise HTTPException(400, "Could not extract video ID from the URL.")

    # 1. Metadata (non-fatal — we carry on with defaults if it fails)
    meta = await get_video_metadata(video_id)

    # 2. Transcript
    try:
        segments = fetch_transcript(video_id)
    except TranscriptsDisabled:
        raise HTTPException(
            422,
            "Transcripts are disabled for this video. The video owner has turned off captions.",
        )
    except VideoUnavailable:
        raise HTTPException(
            422,
            "This video is unavailable (private, deleted, or region-locked).",
        )
    except Exception as e:
        raise HTTPException(
            422,
            f"Could not retrieve transcript: {e}. "
            "Make sure the video has captions/subtitles enabled.",
        )

    if not segments:
        raise HTTPException(
            422,
            "No transcript found for this video. "
            "Only videos with captions (auto-generated or manual) are supported.",
        )

    # 3. Build PDF
    try:
        pdf_bytes = build_transcript_pdf(meta, segments)
    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {e}")

    safe_title = re.sub(r"[^\w\-]", "_", meta["title"])[:60]
    filename   = f"transcript_{safe_title}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
