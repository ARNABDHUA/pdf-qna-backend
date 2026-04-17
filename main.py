import io
import json
import os
import tempfile
import urllib.request
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT

from docx import Document
from pdf2docx import Converter
import re

from rag_engine import RAGEngine, CLOUD_MODELS
from collab_routes import collab_router
from youtube_route import youtube_router
import subprocess

# ── Tessdata setup (no apt-get needed — pymupdf has Tesseract compiled in) ───
# We only need the language data file (.traineddata), downloaded once at startup.
TESSDATA_DIR = os.path.join(os.path.dirname(__file__), "tessdata")
TESSDATA_ENG = os.path.join(TESSDATA_DIR, "eng.traineddata")
TESSDATA_URL = (
    "https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata"
)

def ensure_tessdata():
    """Download eng.traineddata on first run if it's not already present."""
    if os.path.exists(TESSDATA_ENG):
        return
    os.makedirs(TESSDATA_DIR, exist_ok=True)
    print("Downloading eng.traineddata (~4 MB) for OCR support…")
    try:
        urllib.request.urlretrieve(TESSDATA_URL, TESSDATA_ENG)
        print("eng.traineddata downloaded successfully.")
    except Exception as e:
        print(f"Warning: Could not download tessdata: {e}")
        print("OCR endpoint will be unavailable until tessdata is present.")

ensure_tessdata()

# ── Import pymupdf after tessdata is ready ────────────────────────────────────
try:
    import pymupdf  # pip install pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: pymupdf not installed. Run: pip install pymupdf")


def format_text_to_story(text, styles):
    story = []
    lines = text.split('\n')

    section_title = ParagraphStyle(
        'SectionTitle', parent=styles['Heading2'], fontSize=12,
        spaceBefore=12, spaceAfter=6, borderPadding=2, fontName='Helvetica-Bold'
    )
    body_text = ParagraphStyle(
        'BodyText', parent=styles['Normal'], fontSize=10, leading=12, spaceAfter=4
    )

    in_education_section = False
    edu_data = []

    for line in lines:
        clean_line = line.strip()
        if not clean_line: continue

        formatted_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_line)

        if clean_line in ["EXPERIENCE", "PROJECT", "EDUCATION", "SKILLS", "ACHIEVEMENTS"]:
            if in_education_section and edu_data:
                t = Table(edu_data, colWidths=[300, 150])
                t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'), ('FONTSIZE',(0,0),(-1,-1), 9)]))
                story.append(t)
                edu_data = []
            in_education_section = (clean_line == "EDUCATION")
            story.append(Paragraph(clean_line, section_title))
            continue

        if in_education_section:
            if ',' in clean_line:
                parts = clean_line.split(',', 1)
                edu_data.append([Paragraph(parts[0], body_text), Paragraph(parts[1], body_text)])
            else:
                edu_data.append([Paragraph(clean_line, body_text), ""])
            continue

        if clean_line.startswith(('-', '•', '*')):
            p_text = clean_line.lstrip('-•* ').strip()
            formatted_bullet = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', p_text)
            story.append(Paragraph(f"• {formatted_bullet}", body_text))
        else:
            if any(x in clean_line for x in ["7076853097", "@gmail.com"]):
                styles['Normal'].alignment = TA_LEFT
                story.append(Paragraph(formatted_line, styles['Normal']))
            else:
                story.append(Paragraph(formatted_line, body_text))

    if edu_data:
        t = Table(edu_data, colWidths=[300, 150])
        story.append(t)

    return story


app = FastAPI(title="RAG Agent API", version="3.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

rag = RAGEngine()
app.include_router(collab_router)
app.include_router(youtube_router)

# ── Valid modes ───────────────────────────────────────────────────────────────
VALID_MODES = {"chat", "legal", "drafting", "brief"}


class QueryRequest(BaseModel):
    question:           str
    provider:           str  = "ollama"
    model:              str  = ""
    api_key:            str  = ""
    mode:               str  = "chat"
    web_search_enabled: bool = False


class ContextRequest(BaseModel):
    question: str
    mode:     str = "chat"


class SummarizeRequest(BaseModel):
    doc_name:    str
    provider:    str = "ollama"
    model:       str = ""
    api_key:     str = ""


class FollowUpRequest(BaseModel):
    prompt:   str
    provider: str = "ollama"
    model:    str = ""
    api_key:  str = ""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root(): return {"message": "RAG Agent API v3.1 running"}

@app.get("/debug/node")
async def debug_node():
    result = subprocess.run(
        ["which", "node"], capture_output=True, text=True
    )
    result2 = subprocess.run(
        ["find", "/", "-name", "node", "-type", "f"],
        capture_output=True, text=True, timeout=10
    )
    return {
        "which_node": result.stdout.strip(),
        "find_node": result2.stdout.strip(),
    }

@app.post("/summarize")
async def summarize_document(req: SummarizeRequest):
    all_chunks = [c for c in rag.chunks if c["source"] == req.doc_name]
    if not all_chunks:
        raise HTTPException(404, f"No chunks found for '{req.doc_name}'")

    sample_text = " ".join(c["text"] for c in all_chunks[:6])[:3000]
    system_prompt = (
        "You are a document analyst. Summarise the provided document excerpt in "
        "2-3 sentences covering: document type, main parties or subjects, and the "
        "primary purpose or key topics. Be concise and factual. "
        "Output only the summary text — no preamble, no bullet points."
    )
    user_prompt = f"Document excerpt:\n\n{sample_text}\n\nSummary:"

    full = ""
    async for chunk in rag.query_stream(
        question=user_prompt,
        provider=req.provider,
        model=req.model,
        api_key=req.api_key,
        mode="chat",
        web_search_enabled=False,
    ):
        if chunk and not chunk.startswith("__"):
            full += chunk

    return {"summary": full.strip()}


@app.get("/providers")
async def get_providers():
    ollama_models = await rag.get_ollama_models()
    return {
        "ollama":    {"models": ollama_models,             "needs_key": False},
        "openai":    {"models": CLOUD_MODELS["openai"],    "needs_key": True},
        "anthropic": {"models": CLOUD_MODELS["anthropic"], "needs_key": True},
        "gemini":    {"models": CLOUD_MODELS["gemini"],    "needs_key": True},
        "groq":      {"models": CLOUD_MODELS["groq"],      "needs_key": True},
    }


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    return await rag.process_pdf(await file.read(), file.filename)


@app.get("/documents")
async def get_documents(): return {"documents": rag.get_documents()}


@app.delete("/documents/{doc_name}")
async def delete_document(doc_name: str):
    return await rag.delete_document(doc_name)


@app.post("/query")
async def query(req: QueryRequest):
    mode = req.mode if req.mode in VALID_MODES else "chat"

    if not rag.has_documents() and not (req.web_search_enabled and mode == "chat"):
        raise HTTPException(400, "No documents uploaded. Please upload a PDF first.")

    async def generate():
        async for chunk in rag.query_stream(
            req.question, req.provider, req.model,
            req.api_key, mode, req.web_search_enabled
        ):
            if chunk == "__WEB_SEARCHING__":
                yield f"data: {json.dumps({'searching': True})}\n\n"
            else:
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/context")
async def get_context(req: ContextRequest):
    if not rag.has_documents():
        raise HTTPException(400, "No documents uploaded. Please upload a PDF first.")

    mode = req.mode if req.mode in VALID_MODES else "chat"

    if mode == "legal":
        query = "legal analysis contract parties clauses obligations rights"
        top_k = 10
    elif mode == "drafting":
        query = ("parties facts cause of action relief sought court jurisdiction "
                 "plaint pleadings drafting conveyancing deed")
        top_k = 10
    elif mode == "brief":
        query = ("case facts parties procedural history issues holding ratio decidendi "
                 "judgment court ruling legal principle reasoning")
        top_k = 10
    else:
        query = req.question
        top_k = 5

    chunks = await rag._retrieve_async(query, top_k=top_k)
    if not chunks:
        raise HTTPException(404, "No relevant context found in the uploaded documents.")

    return {"context": rag._context(chunks)}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "ollama": await rag.check_ollama(),
        "documents_loaded": len(rag.get_documents()),
        "ocr_available": PYMUPDF_AVAILABLE and os.path.exists(TESSDATA_ENG),
    }


@app.post("/convert/text-to-pdf")
async def text_to_pdf(text: str = Query(...)):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    if 'Justify' not in styles:
        styles.add(ParagraphStyle(name='Justify', alignment=TA_LEFT, fontSize=11, leading=14))

    story = []
    lines = text.split('\n')
    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            story.append(Spacer(1, 12)); continue
        if clean_line.startswith('---'):
            story.append(Spacer(1, 6)); continue
        formatted_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_line)
        if clean_line.startswith('###'):
            p_text = clean_line.replace('###', '').strip()
            formatted_h = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', p_text)
            story.append(Paragraph(formatted_h, styles['Heading2']))
        else:
            story.append(Paragraph(formatted_line, styles['Justify']))
            story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf",
                             headers={"Content-Disposition": "attachment; filename=formatted_legal.pdf"})


@app.post("/convert/word-to-pdf")
async def word_to_pdf(file: UploadFile = File(...)):
    try:
        content  = await file.read()
        word_doc = Document(io.BytesIO(content))
        full_text = "\n".join([para.text for para in word_doc.paragraphs])
        buffer = io.BytesIO()
        pdf_doc = SimpleDocTemplate(buffer, pagesize=letter,
                                    rightMargin=50, leftMargin=50,
                                    topMargin=50, bottomMargin=50)
        styles = getSampleStyleSheet()
        story  = format_text_to_story(full_text, styles)
        pdf_doc.build(story)
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="application/pdf",
                                 headers={"Content-Disposition": "attachment; filename=consistent_output.pdf"})
    except Exception as e:
        raise HTTPException(500, detail=f"Formatting Error: {str(e)}")


@app.post("/convert/pdf-to-word")
async def pdf_to_word(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(await file.read())
        tmp_pdf_path = tmp_pdf.name

    tmp_docx_path = tmp_pdf_path.replace(".pdf", ".docx")
    cv = Converter(tmp_pdf_path)
    cv.convert(tmp_docx_path)
    cv.close()

    with open(tmp_docx_path, "rb") as f:
        docx_content = f.read()

    os.remove(tmp_pdf_path)
    os.remove(tmp_docx_path)

    return StreamingResponse(io.BytesIO(docx_content),
                             media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                             headers={"Content-Disposition": "attachment; filename=pdf_to_word.docx"})


@app.post("/convert/pdf-image-to-pdf")
async def pdf_image_to_pdf(file: UploadFile = File(...)):
    """
    OCR a scanned/image-only PDF and return a searchable PDF.

    Uses PyMuPDF's built-in Tesseract (compiled in — no system tesseract needed).
    Only requires: pip install pymupdf
    And the tessdata/eng.traineddata language file (auto-downloaded at startup).
    """
    if not PYMUPDF_AVAILABLE:
        raise HTTPException(
            500,
            detail="pymupdf is not installed. Run: pip install pymupdf"
        )

    if not os.path.exists(TESSDATA_ENG):
        raise HTTPException(
            503,
            detail=(
                "Tesseract language data not found. "
                "The server will download it automatically on the next restart, "
                "or place eng.traineddata in the tessdata/ folder manually."
            )
        )

    try:
        content = await file.read()

        # Open the uploaded PDF with pymupdf
        src_doc = pymupdf.open(stream=content, filetype="pdf")

        all_text_pages = []

        for page_index, page in enumerate(src_doc):
            # First try native text extraction (fast, no OCR needed for text PDFs)
            native_text = page.get_text().strip()

            if native_text and len(native_text) > 20:
                # Page already has embedded text — use it directly
                all_text_pages.append(f"[Page {page_index + 1}]\n{native_text}")
            else:
                # Image-only page — run OCR via pymupdf's built-in Tesseract
                # tessdata dir must be set so MuPDF finds the language pack
                tp = page.get_textpage_ocr(
                    dpi=300,
                    full=True,
                    tessdata=TESSDATA_DIR,
                    language="eng",
                )
                ocr_text = page.get_text(textpage=tp).strip()
                all_text_pages.append(f"[Page {page_index + 1}]\n{ocr_text}")

        src_doc.close()

        # Build a clean PDF from all extracted text using reportlab
        full_text = "\n\n".join(all_text_pages)

        buffer = io.BytesIO()
        pdf_doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()
        if 'OCRBody' not in styles:
            styles.add(ParagraphStyle(
                name='OCRBody',
                parent=styles['Normal'],
                fontSize=11,
                leading=15,
                spaceAfter=4,
                alignment=TA_LEFT,
            ))
        if 'OCRPageHeader' not in styles:
            styles.add(ParagraphStyle(
                name='OCRPageHeader',
                parent=styles['Heading3'],
                fontSize=10,
                leading=14,
                spaceBefore=14,
                spaceAfter=6,
                textColor=colors.HexColor('#888888'),
            ))

        story = []
        for line in full_text.split('\n'):
            clean = line.strip()
            if not clean:
                story.append(Spacer(1, 6))
                continue
            # Page header lines like "[Page 1]"
            if re.match(r'^\[Page \d+\]$', clean):
                story.append(Paragraph(clean, styles['OCRPageHeader']))
                continue
            # Escape XML special characters for reportlab
            safe = clean.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(safe, styles['OCRBody']))

        if not story:
            story.append(Paragraph("No text could be extracted from this PDF.", styles['OCRBody']))

        pdf_doc.build(story)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=searchable_ocr.pdf"},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"OCR Error: {str(e)}")


@app.post("/followups")
async def get_followups(req: FollowUpRequest):
    system = (
        "You are a helpful assistant. When given a Q&A exchange, "
        "you respond with ONLY a valid JSON array of exactly 3 short follow-up questions. "
        "No explanation, no markdown, no preamble. Just the raw JSON array."
    )
    full_response = ""
    async for chunk in rag.query_stream(
        question=req.prompt,
        provider=req.provider,
        model=req.model,
        api_key=req.api_key,
        mode="chat",
        web_search_enabled=False,
    ):
        if chunk and not chunk.startswith("__"):
            full_response += chunk

    try:
        cleaned = re.sub(r"```(?:json)?|```", "", full_response).strip()
        match = re.search(r'\[.*?\]', cleaned, re.DOTALL)
        if match:
            suggestions = json.loads(match.group())
            if isinstance(suggestions, list):
                return {"suggestions": [str(s) for s in suggestions[:3]]}
    except Exception:
        pass

    return {"suggestions": []}


if __name__ == "__main__":
    print("🚀 Starting server...")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
