"""
ppt_routes.py  –  AI-powered PowerPoint generation
----------------------------------------------------
Add to main.py:
    from ppt_routes import ppt_router
    app.include_router(ppt_router)

Dependencies:
    pip install python-pptx pptxgenjs   ← JS side (see gen_ppt.js)
    npm install -g pptxgenjs            ← run once on the server
"""

import io
import json
import os
import re
import subprocess
import tempfile
import textwrap
from typing import Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

ppt_router = APIRouter(prefix="/ppt", tags=["PPT Generation"])

# ── LLM helpers ──────────────────────────────────────────────────────────────

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


async def _call_llm(
    prompt: str,
    provider: str,
    model: str,
    api_key: str,
) -> str:
    """
    Unified LLM caller – returns the full response string.
    Supports: ollama | openai | anthropic | gemini | groq
    """
    system = (
        "You are a presentation designer assistant. "
        "Always respond with ONLY valid JSON – no markdown fences, no preamble."
    )

    # ── Ollama ────────────────────────────────────────────────────────────────
    if provider == "ollama":
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model": model or "llama3",
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    # ── OpenAI ────────────────────────────────────────────────────────────────
    if provider == "openai":
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model or "gpt-4o-mini",
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    # ── Anthropic / Claude ────────────────────────────────────────────────────
    if provider == "anthropic":
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": model or "claude-3-haiku-20240307",
                    "max_tokens": 4096,
                    "system": system,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]

    # ── Google Gemini ─────────────────────────────────────────────────────────
    if provider == "gemini":
        m = model or "gemini-1.5-flash"
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
                params={"key": api_key},
                json={
                    "contents": [
                        {
                            "parts": [
                                {"text": f"{system}\n\n{prompt}"}
                            ]
                        }
                    ],
                    "generationConfig": {"responseMimeType": "application/json"},
                },
            )
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

    # ── Groq ──────────────────────────────────────────────────────────────────
    if provider == "groq":
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model or "llama3-8b-8192",
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    raise HTTPException(400, f"Unknown provider: {provider}")


def _parse_json_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"LLM returned invalid JSON: {e}\n\nRaw:\n{clean[:400]}")


# ── Slide-plan prompt ─────────────────────────────────────────────────────────

PLAN_PROMPT = textwrap.dedent("""
You will create a detailed PowerPoint presentation plan in JSON.

TOPIC / CONTENT:
{content}

SLIDE COUNT: {slide_count}
THEME: {theme}

Return ONLY a JSON object with this exact shape:
{{
  "title": "Presentation Title",
  "subtitle": "Optional subtitle or tagline",
  "theme": {{
    "primary": "1E2761",
    "secondary": "CADCFC",
    "accent": "F9E795",
    "bg_dark": "0F1642",
    "bg_light": "F8F9FF",
    "font_heading": "Calibri",
    "font_body": "Calibri"
  }},
  "slides": [
    {{
      "type": "title",
      "title": "Main Title",
      "subtitle": "Subtitle or tagline",
      "speaker_notes": "Opening remarks..."
    }},
    {{
      "type": "bullets",
      "title": "Slide Title",
      "points": ["Key point 1", "Key point 2", "Key point 3"],
      "icon_keyword": "chart",
      "speaker_notes": "Explain the points..."
    }},
    {{
      "type": "two_column",
      "title": "Comparison",
      "left_header": "Option A",
      "left_points": ["Pro 1", "Pro 2"],
      "right_header": "Option B",
      "right_points": ["Pro 1", "Pro 2"],
      "speaker_notes": "Compare the two..."
    }},
    {{
      "type": "stat_callout",
      "title": "Key Metrics",
      "stats": [
        {{"value": "94%", "label": "Satisfaction Rate"}},
        {{"value": "3x", "label": "Performance Boost"}},
        {{"value": "$2M", "label": "Cost Savings"}}
      ],
      "speaker_notes": "Impressive numbers..."
    }},
    {{
      "type": "timeline",
      "title": "Roadmap",
      "steps": [
        {{"step": "1", "label": "Phase 1", "desc": "Short description"}},
        {{"step": "2", "label": "Phase 2", "desc": "Short description"}},
        {{"step": "3", "label": "Phase 3", "desc": "Short description"}}
      ],
      "speaker_notes": "Walk through the plan..."
    }},
    {{
      "type": "closing",
      "title": "Thank You",
      "tagline": "Call to action or contact info",
      "speaker_notes": "Wrap up..."
    }}
  ]
}}

Slide type mix guidance:
- Slide 1: always "title"
- Last slide: always "closing"
- Mix "bullets", "two_column", "stat_callout", "timeline" for the middle slides
- Use "stat_callout" if content has numbers/data
- Use "two_column" for comparisons or pros/cons
- Use "timeline" for processes or roadmaps
- Choose theme colors that match the topic (don't default to generic blue)
- Keep bullet points short (< 10 words each), max 4 per slide
""")


# ── Node.js PptxGenJS script generator ───────────────────────────────────────

def _build_js_script(plan: dict, output_path: str) -> str:
    """
    Generates a complete Node.js script using PptxGenJS
    that builds the presentation from the LLM plan.
    """
    p = json.dumps(plan)   # embed the plan as a JSON literal

    script = f"""
const pptxgen = require('pptxgenjs');
const plan = {p};

const prs = new pptxgen();
prs.layout = 'LAYOUT_16x9';
prs.title  = plan.title || 'Presentation';

const T  = plan.theme;
const PRI  = T.primary   || '1E2761';
const SEC  = T.secondary || 'CADCFC';
const ACC  = T.accent    || 'F9E795';
const DARK = T.bg_dark   || '0F1642';
const LITE = T.bg_light  || 'F8F9FF';
const FH   = T.font_heading || 'Calibri';
const FB   = T.font_body    || 'Calibri';

function hex(c) {{ return c.replace('#',''); }}

/* ─── helpers ─────────────────────────────────────────── */
function titleBar(slide, text, dark) {{
  slide.background = {{ color: hex(dark ? DARK : LITE) }};
  slide.addShape(prs.shapes.RECTANGLE, {{
    x: 0, y: 0, w: 10, h: 0.9,
    fill: {{ color: hex(PRI) }}, line: {{ color: hex(PRI) }}
  }});
  slide.addText(text, {{
    x: 0.4, y: 0.1, w: 9.2, h: 0.7,
    fontSize: 22, bold: true, color: 'FFFFFF',
    fontFace: FH, valign: 'middle', margin: 0
  }});
}}

function accentRect(slide, x, y, h) {{
  slide.addShape(prs.shapes.RECTANGLE, {{
    x, y, w: 0.06, h,
    fill: {{ color: hex(ACC) }}, line: {{ color: hex(ACC) }}
  }});
}}

/* ─── slide renderers ─────────────────────────────────── */
function renderTitle(slide, s) {{
  slide.background = {{ color: hex(DARK) }};

  // large accent shape
  slide.addShape(prs.shapes.RECTANGLE, {{
    x: 0, y: 3.8, w: 10, h: 1.825,
    fill: {{ color: hex(PRI) }}, line: {{ color: hex(PRI) }}
  }});
  slide.addShape(prs.shapes.RECTANGLE, {{
    x: 0, y: 3.75, w: 3.5, h: 0.12,
    fill: {{ color: hex(ACC) }}, line: {{ color: hex(ACC) }}
  }});

  slide.addText(s.title || plan.title, {{
    x: 0.6, y: 1.1, w: 8.8, h: 1.5,
    fontSize: 40, bold: true, color: 'FFFFFF',
    fontFace: FH, align: 'left', valign: 'middle'
  }});
  if (s.subtitle) {{
    slide.addText(s.subtitle, {{
      x: 0.6, y: 2.8, w: 8.8, h: 0.8,
      fontSize: 18, color: hex(SEC), fontFace: FB, align: 'left', italic: true
    }});
  }}
  slide.addText(plan.title, {{
    x: 0.4, y: 4.0, w: 9.2, h: 0.7,
    fontSize: 13, color: 'FFFFFF', fontFace: FB, align: 'left', transparency: 40
  }});
}}

function renderBullets(slide, s) {{
  titleBar(slide, s.title, false);
  accentRect(slide, 0.5, 1.15, 3.8);

  const bullets = (s.points || []).map((p, i) => ({{
    text: p,
    options: {{ bullet: true, breakLine: i < s.points.length - 1, color: '2D3748', fontSize: 15, fontFace: FB }}
  }}));
  slide.addText(bullets, {{
    x: 0.75, y: 1.15, w: 8.8, h: 3.8,
    valign: 'top', paraSpaceAfter: 12
  }});
}}

function renderTwoColumn(slide, s) {{
  titleBar(slide, s.title, false);

  // Left card
  slide.addShape(prs.shapes.RECTANGLE, {{
    x: 0.4, y: 1.1, w: 4.3, h: 4.0,
    fill: {{ color: hex(PRI) }}, line: {{ color: hex(PRI) }},
    shadow: {{ type:'outer', blur:8, offset:3, angle:135, color:'000000', opacity:0.12 }}
  }});
  slide.addText(s.left_header || 'Option A', {{
    x: 0.55, y: 1.2, w: 4.0, h: 0.5,
    fontSize: 14, bold: true, color: hex(ACC), fontFace: FH, margin: 0
  }});
  const lp = (s.left_points || []).map((p,i) => ({{
    text: p,
    options: {{ bullet: true, breakLine: i < s.left_points.length-1, color:'FFFFFF', fontSize:13, fontFace: FB }}
  }}));
  slide.addText(lp, {{ x: 0.55, y: 1.75, w: 4.0, h: 3.2, valign:'top', paraSpaceAfter:10 }});

  // Right card
  slide.addShape(prs.shapes.RECTANGLE, {{
    x: 5.3, y: 1.1, w: 4.3, h: 4.0,
    fill: {{ color: hex(SEC) }}, line: {{ color: hex(SEC) }},
    shadow: {{ type:'outer', blur:8, offset:3, angle:135, color:'000000', opacity:0.12 }}
  }});
  slide.addText(s.right_header || 'Option B', {{
    x: 5.45, y: 1.2, w: 4.0, h: 0.5,
    fontSize: 14, bold: true, color: hex(PRI), fontFace: FH, margin: 0
  }});
  const rp = (s.right_points || []).map((p,i) => ({{
    text: p,
    options: {{ bullet: true, breakLine: i < s.right_points.length-1, color: hex(DARK), fontSize:13, fontFace: FB }}
  }}));
  slide.addText(rp, {{ x: 5.45, y: 1.75, w: 4.0, h: 3.2, valign:'top', paraSpaceAfter:10 }});
}}

function renderStatCallout(slide, s) {{
  titleBar(slide, s.title, false);
  const stats = (s.stats || []).slice(0, 3);
  const total = stats.length;
  const cardW = total === 2 ? 4.0 : 2.8;
  const spacing = total === 2 ? 1.5 : 0.55;
  const startX = total === 2 ? 1.0 : 0.55;

  stats.forEach((st, i) => {{
    const cx = startX + i * (cardW + spacing);
    slide.addShape(prs.shapes.RECTANGLE, {{
      x: cx, y: 1.2, w: cardW, h: 3.6,
      fill: {{ color: i % 2 === 0 ? hex(PRI) : hex(SEC) }},
      line: {{ color: i % 2 === 0 ? hex(PRI) : hex(SEC) }},
      shadow: {{ type:'outer', blur:10, offset:4, angle:135, color:'000000', opacity:0.15 }}
    }});
    const textColor = i % 2 === 0 ? 'FFFFFF' : hex(DARK);
    slide.addText(st.value, {{
      x: cx, y: 1.6, w: cardW, h: 1.5,
      fontSize: 44, bold: true, color: textColor,
      fontFace: FH, align: 'center', valign: 'middle', margin: 0
    }});
    slide.addShape(prs.shapes.RECTANGLE, {{
      x: cx + cardW*0.25, y: 3.1, w: cardW*0.5, h: 0.04,
      fill: {{ color: hex(ACC) }}, line: {{ color: hex(ACC) }}
    }});
    slide.addText(st.label, {{
      x: cx, y: 3.25, w: cardW, h: 0.9,
      fontSize: 13, color: textColor, fontFace: FB,
      align: 'center', valign: 'middle', margin: 0
    }});
  }});
}}

function renderTimeline(slide, s) {{
  titleBar(slide, s.title, false);
  const steps = (s.steps || []).slice(0, 5);
  const n = steps.length;
  const stepW = 9.0 / n;

  // connector line
  slide.addShape(prs.shapes.RECTANGLE, {{
    x: 0.5, y: 2.9, w: 9.0, h: 0.06,
    fill: {{ color: hex(SEC) }}, line: {{ color: hex(SEC) }}
  }});

  steps.forEach((st, i) => {{
    const cx = 0.5 + i * stepW + stepW / 2;
    // circle
    slide.addShape(prs.shapes.OVAL, {{
      x: cx - 0.32, y: 2.6, w: 0.64, h: 0.64,
      fill: {{ color: hex(PRI) }}, line: {{ color: hex(ACC), width: 2 }}
    }});
    slide.addText(st.step || String(i+1), {{
      x: cx - 0.32, y: 2.6, w: 0.64, h: 0.64,
      fontSize: 13, bold: true, color: 'FFFFFF',
      fontFace: FH, align: 'center', valign: 'middle', margin: 0
    }});
    // label
    slide.addText(st.label || '', {{
      x: cx - stepW/2 + 0.05, y: 1.7, w: stepW - 0.1, h: 0.75,
      fontSize: 12, bold: true, color: hex(PRI),
      fontFace: FH, align: 'center', valign: 'bottom', margin: 0
    }});
    // desc
    slide.addText(st.desc || '', {{
      x: cx - stepW/2 + 0.05, y: 3.4, w: stepW - 0.1, h: 1.6,
      fontSize: 11, color: '4A5568', fontFace: FB,
      align: 'center', valign: 'top', margin: 0, wrap: true
    }});
  }});
}}

function renderClosing(slide, s) {{
  slide.background = {{ color: hex(DARK) }};
  slide.addShape(prs.shapes.RECTANGLE, {{
    x: 0, y: 0, w: 10, h: 5.625,
    fill: {{ color: hex(PRI) }}, line: {{ color: hex(PRI) }}, transparency: 70
  }});
  slide.addShape(prs.shapes.RECTANGLE, {{
    x: 3, y: 2.8, w: 4, h: 0.08,
    fill: {{ color: hex(ACC) }}, line: {{ color: hex(ACC) }}
  }});

  slide.addText(s.title || 'Thank You', {{
    x: 0.5, y: 1.3, w: 9, h: 1.4,
    fontSize: 48, bold: true, color: 'FFFFFF',
    fontFace: FH, align: 'center', valign: 'middle'
  }});
  if (s.tagline) {{
    slide.addText(s.tagline, {{
      x: 0.5, y: 3.1, w: 9, h: 0.8,
      fontSize: 16, color: hex(SEC), fontFace: FB,
      align: 'center', italic: true
    }});
  }}
}}

/* ─── build slides ────────────────────────────────────── */
plan.slides.forEach(s => {{
  const slide = prs.addSlide();
  if (s.speaker_notes) slide.addNotes(s.speaker_notes);

  switch (s.type) {{
    case 'title':       renderTitle(slide, s);      break;
    case 'bullets':     renderBullets(slide, s);    break;
    case 'two_column':  renderTwoColumn(slide, s);  break;
    case 'stat_callout':renderStatCallout(slide, s);break;
    case 'timeline':    renderTimeline(slide, s);   break;
    case 'closing':     renderClosing(slide, s);    break;
    default:            renderBullets(slide, s);    break;
  }}
}});

prs.writeFile({{ fileName: '{output_path}' }})
  .then(() => process.exit(0))
  .catch(e => {{ console.error(e); process.exit(1); }});
"""
    return script


# ── Route: generate from text ─────────────────────────────────────────────────

class PPTRequest(BaseModel):
    content: str
    provider: str = "ollama"
    model: str = ""
    api_key: str = ""
    slide_count: int = 8
    theme: str = "auto"   # "auto" lets LLM pick; or pass e.g. "dark executive"


@ppt_router.post("/generate")
async def generate_ppt(req: PPTRequest):
    """
    Accepts text content + LLM config, returns a .pptx file.
    """
    prompt = PLAN_PROMPT.format(
        content=req.content[:6000],      # cap to avoid token overflow
        slide_count=max(4, min(req.slide_count, 20)),
        theme=req.theme,
    )

    raw = await _call_llm(prompt, req.provider, req.model, req.api_key)
    plan = _parse_json_response(raw)

    pptx_bytes = await _run_pptxgenjs(plan)

    filename = re.sub(r"[^\w\-]", "_", plan.get("title", "presentation"))[:50] + ".pptx"
    return StreamingResponse(
        io.BytesIO(pptx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── Route: generate from uploaded file ───────────────────────────────────────

@ppt_router.post("/generate-from-file")
async def generate_ppt_from_file(
    file: UploadFile = File(...),
    provider: str = Form("ollama"),
    model: str = Form(""),
    api_key: str = Form(""),
    slide_count: int = Form(8),
    theme: str = Form("auto"),
):
    """
    Accepts a .txt / .pdf / .docx file, extracts text, generates PPT.
    For PDF extraction, re-uses pymupdf if available.
    """
    content_bytes = await file.read()
    filename = file.filename or ""

    if filename.endswith(".txt"):
        content = content_bytes.decode("utf-8", errors="ignore")

    elif filename.endswith(".pdf"):
        content = _extract_pdf_text(content_bytes)

    elif filename.endswith(".docx"):
        content = _extract_docx_text(content_bytes)

    else:
        # Try plain text fallback
        content = content_bytes.decode("utf-8", errors="ignore")

    if not content.strip():
        raise HTTPException(400, "Could not extract text from the uploaded file.")

    req = PPTRequest(
        content=content,
        provider=provider,
        model=model,
        api_key=api_key,
        slide_count=slide_count,
        theme=theme,
    )
    return await generate_ppt(req)


# ── Route: preview slide plan (JSON only, no file) ────────────────────────────

@ppt_router.post("/plan")
async def preview_plan(req: PPTRequest):
    """
    Returns the JSON slide plan without generating the PPTX.
    Useful for frontend previews or editing before download.
    """
    prompt = PLAN_PROMPT.format(
        content=req.content[:6000],
        slide_count=max(4, min(req.slide_count, 20)),
        theme=req.theme,
    )
    raw = await _call_llm(prompt, req.provider, req.model, req.api_key)
    plan = _parse_json_response(raw)
    return {"plan": plan}


# ── Route: generate from existing plan (edit → download) ─────────────────────

class PlanRequest(BaseModel):
    plan: dict


@ppt_router.post("/generate-from-plan")
async def generate_from_plan(req: PlanRequest):
    """
    Accepts a previously returned plan (possibly edited) and renders it to PPTX.
    """
    pptx_bytes = await _run_pptxgenjs(req.plan)
    filename = re.sub(r"[^\w\-]", "_", req.plan.get("title", "presentation"))[:50] + ".pptx"
    return StreamingResponse(
        io.BytesIO(pptx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── PptxGenJS runner ──────────────────────────────────────────────────────────

async def _run_pptxgenjs(plan: dict) -> bytes:
    """Writes a temp JS script, runs it with Node, returns the .pptx bytes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "gen_ppt.js")
        output_path = os.path.join(tmpdir, "output.pptx")

        js_code = _build_js_script(plan, output_path)
        with open(script_path, "w") as f:
            f.write(js_code)

        result = subprocess.run(
            ["node", script_path],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise HTTPException(
                500,
                detail=f"PptxGenJS error: {result.stderr[:800]}",
            )

        if not os.path.exists(output_path):
            raise HTTPException(500, "PPTX file was not created by PptxGenJS.")

        with open(output_path, "rb") as f:
            return f.read()


# ── Text extraction helpers ───────────────────────────────────────────────────

def _extract_pdf_text(data: bytes) -> str:
    try:
        import pymupdf  # type: ignore
        doc = pymupdf.open(stream=data, filetype="pdf")
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages)
    except ImportError:
        raise HTTPException(500, "pymupdf not installed. Run: pip install pymupdf")


def _extract_docx_text(data: bytes) -> str:
    try:
        from docx import Document  # type: ignore
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    except ImportError:
        raise HTTPException(500, "python-docx not installed. Run: pip install python-docx")