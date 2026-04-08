import io
import os
import re
import json
from typing import AsyncGenerator, List, Dict, Any
import httpx
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50

CLOUD_MODELS = {
    "openai": [
        {"id": "gpt-4o",        "label": "GPT-4o"},
        {"id": "gpt-4o-mini",   "label": "GPT-4o Mini"},
        {"id": "gpt-4-turbo",   "label": "GPT-4 Turbo"},
        {"id": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo"},
    ],
    "anthropic": [
        {"id": "claude-opus-4-5",           "label": "Claude Opus 4.5"},
        {"id": "claude-sonnet-4-5",         "label": "Claude Sonnet 4.5"},
        {"id": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5"},
    ],
    "gemini": [
        {"id": "gemini-2.5-flash",  "label": "Gemini 2.5 Flash"},
        {"id": "gemini-2.0-flash",  "label": "Gemini 2.0 Flash"},
        {"id": "gemini-1.5-pro",    "label": "Gemini 1.5 Pro"},
        {"id": "gemini-1.5-flash",  "label": "Gemini 1.5 Flash"},
    ],
    "groq": [
        {"id": "llama-3.3-70b-versatile",    "label": "LLaMA 3.3 70B"},
        {"id": "llama-3.1-8b-instant",       "label": "LLaMA 3.1 8B Instant"},
        {"id": "llama3-70b-8192",            "label": "LLaMA3 70B"},
        {"id": "llama3-8b-8192",             "label": "LLaMA3 8B"},
        {"id": "mixtral-8x7b-32768",         "label": "Mixtral 8x7B"},
        {"id": "gemma2-9b-it",               "label": "Gemma2 9B"},
        {"id": "deepseek-r1-distill-llama-70b", "label": "DeepSeek R1 70B"},
        {"id": "qwen-qwq-32b",               "label": "Qwen QwQ 32B"},
    ],
}

# ── Simple web search via DuckDuckGo (no API key needed) ─────────────────────
DDGS_API = "https://api.duckduckgo.com/"

async def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Perform a web search using DuckDuckGo Instant Answer API + HTML scrape fallback.
    Returns list of {title, snippet, url}.
    """
    results = []
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            # DuckDuckGo Instant Answer (JSON)
            resp = await client.get(DDGS_API, params={
                "q": query, "format": "json", "no_redirect": "1",
                "no_html": "1", "skip_disambig": "1"
            })
            data = resp.json()

            # AbstractText (single main result)
            if data.get("AbstractText"):
                results.append({
                    "title": data.get("Heading", "Result"),
                    "snippet": data["AbstractText"][:400],
                    "url": data.get("AbstractURL", ""),
                })

            # RelatedTopics
            for topic in data.get("RelatedTopics", []):
                if len(results) >= max_results:
                    break
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title": topic.get("Text", "")[:80],
                        "snippet": topic.get("Text", "")[:400],
                        "url": topic.get("FirstURL", ""),
                    })

        # Fallback: DuckDuckGo HTML search (lite endpoint)
        if not results:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"}) as client:
                resp = await client.get("https://html.duckduckgo.com/html/",
                                        params={"q": query})
                html = resp.text
                # Very simple regex scrape of result snippets
                snippets = re.findall(
                    r'<a class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
                titles   = re.findall(
                    r'<a class="result__a"[^>]*>(.*?)</a>', html, re.DOTALL)
                urls     = re.findall(
                    r'<a class="result__a" href="([^"]+)"', html)
                for i, snippet in enumerate(snippets[:max_results]):
                    clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                    clean_title   = re.sub(r'<[^>]+>', '', titles[i]).strip() if i < len(titles) else ""
                    url           = urls[i] if i < len(urls) else ""
                    if clean_snippet:
                        results.append({
                            "title": clean_title[:80],
                            "snippet": clean_snippet[:400],
                            "url": url,
                        })
    except Exception as e:
        results.append({
            "title": "Search Error",
            "snippet": f"Web search failed: {e}",
            "url": "",
        })
    return results[:max_results]


def format_search_results(results: List[Dict[str, str]]) -> str:
    if not results:
        return ""
    parts = ["### Web Search Results\n"]
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] **{r['title']}**")
        parts.append(f"    {r['snippet']}")
        if r["url"]:
            parts.append(f"    Source: {r['url']}")
        parts.append("")
    return "\n".join(parts)


class RAGEngine:
    def __init__(self):
        print("Loading embedding model...")
        self.embedder  = SentenceTransformer(EMBEDDING_MODEL)
        self.index     = None
        self.chunks:    List[Dict[str, Any]] = []
        self.documents: List[Dict[str, str]] = []
        self.dimension = 384
        self._init_index()

    def _init_index(self):
        self.index = faiss.IndexFlatL2(self.dimension)

    def _chunk_text(self, text: str, filename: str) -> List[Dict[str, Any]]:
        words, chunks, i = text.split(), [], 0
        while i < len(words):
            chunk_text = " ".join(words[i: i + CHUNK_SIZE])
            if chunk_text.strip():
                chunks.append({"text": chunk_text, "source": filename,
                                "chunk_id": len(self.chunks) + len(chunks)})
            i += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    async def process_pdf(self, content: bytes, filename: str) -> Dict:
        try:
            pages = []
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                total = len(pdf.pages)
                for p in pdf.pages:
                    t = p.extract_text()
                    if t: pages.append(t)
            full = re.sub(r'\s+', ' ', "\n".join(pages)).strip()
            if not full:
                return {"error": "No text could be extracted from the PDF"}
            new_chunks = self._chunk_text(full, filename)
            if not new_chunks:
                return {"error": "No chunks created from PDF"}
            emb = np.array(self.embedder.encode(
                [c["text"] for c in new_chunks], show_progress_bar=False), dtype=np.float32)
            self.index.add(emb)
            self.chunks.extend(new_chunks)
            self.documents.append({"name": filename, "pages": total,
                                   "chunks": len(new_chunks), "chars": len(full)})
            return {"success": True, "filename": filename, "pages": total,
                    "chunks": len(new_chunks),
                    "message": f"Processed '{filename}' — {total} pages, {len(new_chunks)} chunks"}
        except Exception as e:
            return {"error": f"Failed to process PDF: {e}"}

    def _retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        q = np.array(self.embedder.encode([query], show_progress_bar=False), dtype=np.float32)
        dists, idxs = self.index.search(q, min(top_k, self.index.ntotal))
        results = []
        for d, i in zip(dists[0], idxs[0]):
            if i < len(self.chunks):
                c = self.chunks[i].copy()
                c["score"] = float(d)
                results.append(c)
        return results

    def _context(self, chunks: List[Dict]) -> str:
        return "\n\n---\n\n".join(
            f"[Source: {c['source']}, Chunk {i+1}]\n{c['text']}"
            for i, c in enumerate(chunks))

    # ── System prompts ────────────────────────────────────────────────────────

    def _system(self, mode: str = "chat") -> str:
        if mode == "legal":
            return """You are a senior legal counsel and expert legal document analyst with 20+ years of experience.
Your task is to produce a comprehensive legal analysis in professional legal draft format.

STRICT OUTPUT FORMAT — you MUST produce ALL of the following sections in order:

═══════════════════════════════════════════════════
LEGAL ANALYSIS MEMORANDUM
═══════════════════════════════════════════════════

RE: [State the subject matter of analysis]
DATE: [Today's date]
PREPARED BY: AI Legal Analysis System

───────────────────────────────────────────────────
I. EXECUTIVE SUMMARY
───────────────────────────────────────────────────
[2-3 paragraph high-level summary of the document and key legal findings]

───────────────────────────────────────────────────
II. DOCUMENT IDENTIFICATION & NATURE
───────────────────────────────────────────────────
• Document Type: [Contract / Agreement / Deed / Notice / etc.]
• Parties Involved: [List all parties with their roles]
• Effective Date: [If mentioned]
• Governing Law / Jurisdiction: [If mentioned]

───────────────────────────────────────────────────
III. KEY PROVISIONS & CLAUSES
───────────────────────────────────────────────────
[List and explain each significant clause or provision found in the document]
For each clause: State the clause → Explain its legal effect → Note any implications

───────────────────────────────────────────────────
IV. RIGHTS & OBLIGATIONS
───────────────────────────────────────────────────
A. Rights Granted:
[List all rights explicitly granted to each party]

B. Obligations & Duties:
[List all obligations imposed on each party]

C. Restrictions & Prohibitions:
[List any restrictions, non-compete, non-disclosure, or prohibited acts]

───────────────────────────────────────────────────
V. RISK ASSESSMENT & RED FLAGS
───────────────────────────────────────────────────
⚠ HIGH RISK:
[Clauses or provisions that pose significant legal risk]

⚠ MEDIUM RISK:
[Clauses that require attention or negotiation]

✓ LOW RISK / FAVORABLE:
[Protective or standard clauses]

───────────────────────────────────────────────────
VI. LEGAL ISSUES & CONCERNS
───────────────────────────────────────────────────
[Identify ambiguous language, missing clauses, enforceability issues, compliance concerns]

───────────────────────────────────────────────────
VII. RECOMMENDATIONS
───────────────────────────────────────────────────
[Numbered list of specific actionable recommendations for the client]
1.
2.
3.

───────────────────────────────────────────────────
VIII. CONCLUSION
───────────────────────────────────────────────────
[Final legal opinion and overall assessment]

───────────────────────────────────────────────────
DISCLAIMER: This analysis is generated by an AI system for informational purposes only
and does not constitute legal advice. Consult a qualified attorney for legal counsel.
═══════════════════════════════════════════════════

Base your analysis STRICTLY on the provided document context. Do not fabricate clauses or provisions not present in the document. If certain information is not available in the document, explicitly state "Not specified in the document."
"""
        else:
            return ("You are a helpful AI assistant. Answer the user's question based on the "
                    "provided context. If web search results are included, use them to supplement "
                    "the document context. Be concise, accurate, and cite sources when relevant. "
                    "If the answer cannot be found in any context, say so clearly.")

    # ── User message builders ─────────────────────────────────────────────────

    def _user_msg(self, question: str, context: str, mode: str = "chat",
                  web_context: str = "") -> str:
        if mode == "legal":
            return f"""DOCUMENT CONTEXT:
{context}

TASK: {question if question.strip() else "Perform a complete legal analysis of the above document in the specified legal draft format."}

Produce a thorough legal analysis memorandum following the exact format specified in your instructions."""

        # Chat mode — combine doc context + web context if present
        parts = []
        if context:
            parts.append(f"DOCUMENT CONTEXT:\n{context}")
        if web_context:
            parts.append(f"WEB SEARCH CONTEXT:\n{web_context}")
        combined = "\n\n".join(parts)
        return f"{combined}\n\nQUESTION: {question}\n\nANSWER:"

    # ── Main query stream entry point ─────────────────────────────────────────

    async def query_stream(self, question: str, provider: str,
                           model: str, api_key: str = "",
                           mode: str = "chat",
                           web_search_enabled: bool = False) -> AsyncGenerator[str, None]:

        web_context = ""

        # Perform web search if enabled and in chat mode
        if web_search_enabled and mode == "chat":
            yield "__WEB_SEARCHING__"
            search_results = await web_search(question)
            web_context = format_search_results(search_results)

        if mode == "legal":
            top_k  = 10
            chunks = self._retrieve(
                "legal analysis contract parties clauses obligations rights", top_k=top_k)
            if not chunks:
                yield "No relevant context found in the uploaded documents."
                return
            context = self._context(chunks)
        else:
            top_k  = 5
            chunks = self._retrieve(question, top_k=top_k)
            # In chat mode with web search, we can proceed even without doc chunks
            if not chunks and not web_context:
                yield "No relevant context found in the uploaded documents."
                return
            context = self._context(chunks) if chunks else ""

        system = self._system(mode)
        user   = self._user_msg(question, context, mode, web_context)

        if provider == "ollama":
            async for t in self._ollama_msg(system, user, model): yield t
        elif provider == "openai":
            async for t in self._openai_msg(system, user, model, api_key): yield t
        elif provider == "anthropic":
            async for t in self._anthropic_msg(system, user, model, api_key): yield t
        elif provider == "gemini":
            async for t in self._gemini_msg(system, user, model, api_key): yield t
        elif provider == "groq":
            async for t in self._groq_msg(system, user, model, api_key): yield t
        else:
            yield f"Unknown provider: {provider}"

    # ── Provider implementations ──────────────────────────────────────────────

    async def _ollama_msg(self, system: str, user: str, model: str):
        prompt = f"{system}\n\n{user}"
        async with httpx.AsyncClient(timeout=180.0) as client:
            try:
                async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": True,
                          "options": {"temperature": 0.1, "top_p": 0.9,
                                      "num_predict": 4096}}) as resp:
                    if resp.status_code != 200:
                        yield f"Error: Ollama returned status {resp.status_code}"; return
                    async for line in resp.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "response" in data: yield data["response"]
                                if data.get("done"): break
                            except json.JSONDecodeError: continue
            except httpx.ConnectError:
                yield "Error: Cannot connect to Ollama. Make sure it is running on port 11434."
            except Exception as e:
                yield f"Error: {e}"

    async def _openai_msg(self, system: str, user: str, model: str, api_key: str):
        if not api_key:
            yield "Error: OpenAI API key is required."; return
        payload = {"model": model, "stream": True, "temperature": 0.1,
                   "max_tokens": 4096,
                   "messages": [{"role": "system", "content": system},
                                 {"role": "user",   "content": user}]}
        async with httpx.AsyncClient(timeout=180.0) as client:
            try:
                async with client.stream("POST", "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}",
                             "Content-Type": "application/json"},
                    json=payload) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield f"Error: OpenAI {resp.status_code} — {body.decode()[:200]}"; return
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            chunk = line[6:]
                            if chunk.strip() == "[DONE]": break
                            try:
                                delta = json.loads(chunk)["choices"][0]["delta"].get("content","")
                                if delta: yield delta
                            except (json.JSONDecodeError, KeyError): continue
            except Exception as e:
                yield f"Error: {e}"

    async def _anthropic_msg(self, system: str, user: str, model: str, api_key: str):
        if not api_key:
            yield "Error: Anthropic API key is required."; return
        payload = {"model": model, "max_tokens": 4096, "stream": True,
                   "system": system,
                   "messages": [{"role": "user", "content": user}]}
        async with httpx.AsyncClient(timeout=180.0) as client:
            try:
                async with client.stream("POST", "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                             "Content-Type": "application/json"},
                    json=payload) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield f"Error: Anthropic {resp.status_code} — {body.decode()[:200]}"; return
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if data.get("type") == "content_block_delta":
                                    yield data["delta"].get("text", "")
                            except (json.JSONDecodeError, KeyError): continue
            except Exception as e:
                yield f"Error: {e}"

    async def _gemini_msg(self, system: str, user: str, model: str, api_key: str):
        if not api_key:
            yield "Error: Gemini API key is required."; return
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:streamGenerateContent?alt=sse&key={api_key}")
        payload = {"contents": [{"parts": [{"text": f"{system}\n\n{user}"}]}],
                   "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096}}
        async with httpx.AsyncClient(timeout=180.0) as client:
            try:
                async with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield f"Error: Gemini {resp.status_code} — {body.decode()[:200]}"; return
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                text = data["candidates"][0]["content"]["parts"][0].get("text","")
                                if text: yield text
                            except (json.JSONDecodeError, KeyError, IndexError): continue
            except Exception as e:
                yield f"Error: {e}"

    async def _groq_msg(self, system: str, user: str, model: str, api_key: str):
        """Groq uses an OpenAI-compatible API with streaming support."""
        if not api_key:
            yield "Error: Groq API key is required."; return
        payload = {
            "model": model,
            "stream": True,
            "temperature": 0.1,
            "max_tokens": 4096,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }
        async with httpx.AsyncClient(timeout=180.0) as client:
            try:
                async with client.stream(
                    "POST",
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield f"Error: Groq {resp.status_code} — {body.decode()[:200]}"; return
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            chunk = line[6:]
                            if chunk.strip() == "[DONE]":
                                break
                            try:
                                delta = json.loads(chunk)["choices"][0]["delta"].get("content", "")
                                if delta:
                                    yield delta
                            except (json.JSONDecodeError, KeyError):
                                continue
            except httpx.ConnectError:
                yield "Error: Cannot connect to Groq API."
            except Exception as e:
                yield f"Error: {e}"

    # ── Utility methods ───────────────────────────────────────────────────────

    async def get_ollama_models(self) -> List[str]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                if resp.status_code == 200:
                    return [m["name"] for m in resp.json().get("models", [])]
        except Exception: pass
        return []

    async def check_ollama(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                return (await client.get(f"{OLLAMA_BASE_URL}/api/tags")).status_code == 200
        except Exception: return False

    def has_documents(self) -> bool: return self.index.ntotal > 0
    def get_documents(self) -> List[Dict]: return self.documents

    def delete_document(self, doc_name: str) -> Dict:
        self.documents = [d for d in self.documents if d["name"] != doc_name]
        self.chunks    = [c for c in self.chunks    if c["source"] != doc_name]
        self._init_index()
        if self.chunks:
            emb = np.array(self.embedder.encode(
                [c["text"] for c in self.chunks], show_progress_bar=False), dtype=np.float32)
            self.index.add(emb)
            for i, c in enumerate(self.chunks): c["chunk_id"] = i
        return {"success": True, "message": f"Deleted '{doc_name}'"}
