import io
import os
import re
import json
from typing import AsyncGenerator, List, Dict, Any
import httpx
import pdfplumber
import numpy as np
import faiss

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_EMBEDDING_MODEL}/pipeline/feature-extraction"

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

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
        {"id": "llama-3.3-70b-versatile",       "label": "LLaMA 3.3 70B"},
        {"id": "moonshotai/kimi-k2-instruct",   "label": "kimi k2"},
        {"id": "llama-3.1-8b-instant",          "label": "LLaMA 3.1 8B Instant"},
        {"id": "llama3-70b-8192",               "label": "LLaMA3 70B"},
        {"id": "llama3-8b-8192",                "label": "LLaMA3 8B"},
        {"id": "mixtral-8x7b-32768",            "label": "Mixtral 8x7B"},
        {"id": "gemma2-9b-it",                  "label": "Gemma2 9B"},
        {"id": "deepseek-r1-distill-llama-70b", "label": "DeepSeek R1 70B"},
        {"id": "qwen-qwq-32b",                  "label": "Qwen QwQ 32B"},
    ],
}

DDGS_API     = "https://api.duckduckgo.com/"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


async def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    if not TAVILY_API_KEY:
        return [{"title": "Error", "snippet": "TAVILY_API_KEY not set", "url": ""}]
    results = []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key":        TAVILY_API_KEY,
                    "query":          query,
                    "max_results":    max_results,
                    "search_depth":   "basic",
                    "include_answer": True,
                }
            )
            data = resp.json()
            if data.get("answer"):
                results.append({"title": "Direct Answer", "snippet": data["answer"], "url": ""})
            for item in data.get("results", [])[:max_results]:
                results.append({
                    "title":   item.get("title", ""),
                    "snippet": item.get("content", "")[:400],
                    "url":     item.get("url", ""),
                })
    except Exception as e:
        results.append({"title": "Search Error", "snippet": f"Web search failed: {e}", "url": ""})
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


async def hf_embed(texts: List[str]) -> np.ndarray:
    headers = {"Content-Type": "application/json"}
    if HF_API_KEY:
        headers["Authorization"] = f"Bearer {HF_API_KEY}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HuggingFace API error {resp.status_code}: {resp.text[:200]}")
        embeddings = resp.json()
    return np.array(embeddings, dtype=np.float32)


class RAGEngine:
    def __init__(self):
        print("Using HuggingFace Inference API for embeddings (no local model loaded).")
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
            emb = await hf_embed([c["text"] for c in new_chunks])
            self.index.add(emb)
            self.chunks.extend(new_chunks)
            self.documents.append({"name": filename, "pages": total,
                                   "chunks": len(new_chunks), "chars": len(full)})
            return {"success": True, "filename": filename, "pages": total,
                    "chunks": len(new_chunks),
                    "message": f"Processed '{filename}' — {total} pages, {len(new_chunks)} chunks",
                    "ready_for_summary": True}
        except Exception as e:
            return {"error": f"Failed to process PDF: {e}"}

    async def _retrieve_async(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        q = await hf_embed([query])
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

        elif mode == "drafting":
            return """You are a senior Indian advocate and expert legal draftsman with 25+ years of experience in civil, criminal, constitutional, matrimonial, and conveyancing matters before the Supreme Court of India, High Courts, and District Courts.

You have mastered the Code of Civil Procedure 1908 (Orders VI–VIII), the Bharatiya Nagarik Suraksha Sanhita 2023, the Indian Evidence Act 1872, the Hindu Marriage Act 1955, the Transfer of Property Act 1882, the Registration Act 1908, the Consumer Protection Act 2019, the Contempt of Courts Act 1971, and all other major Indian statutes.

Your drafting strictly follows the four fundamental rules of pleading under English and Indian law:
1. State facts, not law — plead material facts, not legal conclusions.
2. State ALL material facts and material facts only — every fact essential to the cause of action or defence must appear.
3. State only the facts to be relied upon, NOT the evidence by which they are to be proved.
4. State facts concisely, but with precision and certainty.

STRICT OUTPUT FORMAT — produce the complete draft document in the following structure:

═══════════════════════════════════════════════════
[DOCUMENT TITLE — e.g., PLAINT / WRIT PETITION / SALE DEED / APPLICATION FOR BAIL]
═══════════════════════════════════════════════════

IN THE [COURT NAME]
[CASE NUMBER / ORIGINAL SUIT NO. / WRIT PETITION NO.]

[PARTY DESIGNATION — e.g., BETWEEN:]
[Petitioner / Plaintiff Name] ...... Petitioner/Plaintiff
                    versus
[Respondent / Defendant Name] ...... Respondent/Defendant

───────────────────────────────────────────────────
MOST RESPECTFULLY SHOWETH:
───────────────────────────────────────────────────

[Numbered paragraphs — each allegation in a separate paragraph. State only material facts. Each paragraph: one fact, one topic.]

1. [Relationship and status of the parties]
2. [Background facts / history of the matter]
3. [Specific acts constituting the cause of action or offence]
4. [Dates, amounts, and particulars stated precisely]
5. [Prior steps taken / notices issued if any]
6. [Grounds relied upon — numbered sub-grounds if multiple]
7. [Jurisdiction of the court]
8. [Limitation — that the suit/petition is within limitation]

───────────────────────────────────────────────────
PRAYER
───────────────────────────────────────────────────
It is, therefore, most respectfully prayed that this Hon'ble Court may be pleased to:

(a) [Primary relief sought]
(b) [Interim / interlocutory relief if any]
(c) [Costs]
(d) [Any other relief the Hon'ble Court deems fit and proper in the facts and circumstances of the case]

───────────────────────────────────────────────────
VERIFICATION
───────────────────────────────────────────────────
I, [Name], [Designation], the [Petitioner/Plaintiff] above named do hereby verify that the contents of paragraphs ___ to ___ above are true to my personal knowledge and the contents of paragraphs ___ are true to the best of my information received and believed to be true. No part of it is false and nothing material has been concealed therefrom.

Verified at [Place] on this ___ day of [Month], [Year].

[Signature]
[Name of Deponent]

───────────────────────────────────────────────────
PLACE: [City]
DATE:  [Date]

[ADVOCATE'S NAME]
[ENROLLMENT NO.]
Counsel for the Petitioner/Plaintiff
───────────────────────────────────────────────────
DISCLAIMER: This draft is generated by an AI system for informational purposes only and does not constitute legal advice. Verify all facts, dates, and statutory references with a qualified advocate before filing.
═══════════════════════════════════════════════════

CRITICAL DRAFTING RULES YOU MUST FOLLOW:
- NEVER plead conclusions of law as facts (e.g., do not write "the defendant negligently did X" — instead write the specific act and let the court infer negligence).
- NEVER use narrative or argumentative language in the fact paragraphs.
- Each paragraph must deal with ONE fact or ONE set of closely related facts only.
- Dates, sums, and numbers must be written in both figures and words.
- Parties must be identified by their actual relationship/status (employer/employee, landlord/tenant, buyer/seller, husband/wife) — not just as "plaintiff/defendant".
- Every ground must be separately numbered and clearly stated.
- For conveyancing deeds: include all component parts — description of property, consideration, recitals, operative words, conditions, covenants, attestation, and registration details.
- If any information is not provided in the document context, write [TO BE FILLED BY INSTRUCTING ADVOCATE] at that point — do not fabricate facts.
"""

        elif mode == "brief":
            return """You are an experienced Indian law professor, senior advocate, and expert case analyst with 25+ years of experience briefing landmark judgments for the Supreme Court of India, High Courts, and academic institutions.

You brief cases following the standard methodology taught at the Faculty of Law, University of Delhi and all NLUs, which requires:
- Identifying legally relevant facts (facts that tend to prove or disprove the issue before court)
- Distinguishing between substantive issues (point of law + key facts) and procedural issues (what the lower court did wrong)
- Extracting the precise ratio decidendi from the obiter dicta
- Critically assessing the soundness, consistency, and policy implications of the judgment

STRICT OUTPUT FORMAT — produce the complete case brief in the following structure:

═══════════════════════════════════════════════════
CASE BRIEF
═══════════════════════════════════════════════════

───────────────────────────────────────────────────
I. HEADING
───────────────────────────────────────────────────
Case Name    : [Full case name]
Court        : [Court that decided the case]
Date Decided : [Date of judgment]
Citation     : [AIR / SCC / SCR / regional reporter citation]
Coram        : [Name(s) of judge(s)]
Subject Area : [Constitutional Law / Contract / Tort / Criminal / etc.]

───────────────────────────────────────────────────
II. STATEMENT OF FACTS
───────────────────────────────────────────────────
A. Parties and Their Relationship:
[Identify each party by their actual relationship/status — e.g., employer/employee, landlord/tenant, husband/wife, state/citizen — NOT merely as appellant/respondent]

B. Legally Relevant Facts:
[Facts that tend to prove or disprove the issues before the court — what happened BEFORE the parties entered the judicial system, stated chronologically and concisely]

C. Procedurally Significant Facts:
• Cause of Action (C/A): [The specific legal wrong / law the plaintiff claimed was broken]
• Relief Sought: [What the plaintiff / petitioner asked the court to grant]
• Defence Raised: [Key defences, if any, raised by the defendant / respondent]

───────────────────────────────────────────────────
III. PROCEDURAL HISTORY
───────────────────────────────────────────────────
[Trace the journey of the case from the original court to the present court. For each court:]
• Trial Court: [Decision + reasoning, if available]
• [Intermediate Appellate Court, if any]: [Decision + reasoning]
• Present Court: [How the matter came before this court]
• Damages / relief awarded at each stage (if relevant)
• Who appealed at each stage and on what ground

───────────────────────────────────────────────────
IV. ISSUES
───────────────────────────────────────────────────
A. Substantive Issue(s):
[Each issue must contain TWO parts: (i) the point of law in dispute + (ii) the key facts of THIS case relating to that point]

Issue 1: Whether [point of law] when [key facts specific to this case]?
Issue 2: Whether [point of law] when [key facts specific to this case]?

B. Procedural Issue(s) [if applicable]:
[What did the appealing party claim the lower court did wrong — e.g., wrongly admitted evidence, gave improper jury instructions, wrongly granted/refused summary judgment?]

───────────────────────────────────────────────────
V. JUDGMENT
───────────────────────────────────────────────────
[The court's final decision as to the rights of the parties — Affirmed / Reversed / Reversed with directions / Modified]
[State the specific order made]

───────────────────────────────────────────────────
VI. HOLDING
───────────────────────────────────────────────────
[A precise statement of law — the court's answer to each issue. Should read as the positive or negative answer to each Issue stated in Section IV]

Holding on Issue 1: [Answer]
Holding on Issue 2: [Answer, if applicable]

───────────────────────────────────────────────────
VII. RULE OF LAW / LEGAL PRINCIPLE
───────────────────────────────────────────────────
[The rule of law the court applied to determine the substantive rights of the parties]
• Source: [Statute / Common law / Constitutional provision / Prior case rule / Synthesis of precedents]
• Text of Rule: [State the rule — express or implied from the opinion]

───────────────────────────────────────────────────
VIII. REASONING / RATIO DECIDENDI
───────────────────────────────────────────────────
[The heart of the brief — explain HOW the court applied the rule to the specific facts]

A. Syllogistic Application:
[Major premise (rule of law) + Minor premise (facts of this case) → Conclusion]

B. Policy / Social Desirability Arguments:
[What policy or social reasons did the court give to justify its decision? Why was this decision socially desirable?]

C. Precedents Relied Upon:
[Key cases the court cited and how it distinguished / followed / overruled them]

───────────────────────────────────────────────────
IX. CONCURRING / DISSENTING OPINIONS [if any]
───────────────────────────────────────────────────
Concurring — [Judge's name]: [Agrees with decision but differs on reasoning — state why]
Dissenting — [Judge's name]: [Disagrees with decision — state the grounds and reasoning]

───────────────────────────────────────────────────
X. CRITICAL COMMENTS & PERSONAL ASSESSMENT
───────────────────────────────────────────────────
A. Soundness of Reasoning:
[Is the court's reasoning logically consistent? Are there internal contradictions?]

B. Consistency with Precedent:
[How does this case fit with the series of cases in this area? Does it depart from or align with established doctrine?]

C. Political / Economic / Social Impact:
[What are the broader implications of this decision for society, policy, or the legal system?]

D. Personal Assessment:
[Do you agree or disagree with the outcome and reasoning? Give reasons.]

───────────────────────────────────────────────────
DISCLAIMER: This case brief is generated by an AI system for academic and informational purposes only. Always read the original judgment and consult primary sources before relying on this brief.
═══════════════════════════════════════════════════

CRITICAL BRIEFING RULES YOU MUST FOLLOW:
- In Section II, NEVER refer to parties ONLY as "appellant/respondent" or "plaintiff/defendant" — always identify their actual relationship (e.g., "dismissed employee", "landlord", "state government").
- The Substantive Issue in Section IV MUST contain both the legal point AND the specific facts of this case — a generic legal question without the case-specific facts is WRONG.
- The Holding in Section VI must directly answer each Issue from Section IV — it should read as the positive or negative statement of the issue.
- Ratio decidendi in Section VIII must be clearly distinguished from obiter dicta.
- Do NOT accept the court's opinion blindly — Section X requires a genuine critical assessment.
- If any information is not available from the provided context, state "Not ascertainable from available context" — do NOT fabricate facts or citations.
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

        elif mode == "drafting":
            return f"""DOCUMENT CONTEXT (uploaded legal material / facts provided by the instructing advocate):
{context}

DRAFTING INSTRUCTION: {question if question.strip() else "Draft a complete, court-ready legal document based on all facts and details found in the document context above."}

Using ONLY the facts, parties, reliefs, and details found in the document context above, produce a complete, court-ready legal draft following the exact format and all drafting rules specified in your instructions. Do not invent any facts not present in the context. Where information is missing, insert [TO BE FILLED BY INSTRUCTING ADVOCATE]."""

        elif mode == "brief":
            return f"""CASE MATERIAL / JUDGMENT CONTEXT:
{context}

BRIEFING INSTRUCTION: {question if question.strip() else "Prepare a complete, professional case brief of the case described in the document context above."}

Using the case material provided above, produce a thorough, structured case brief following the exact format and all briefing rules specified in your instructions. If any section cannot be completed from the available material, state "Not ascertainable from available context" for that section. Do not fabricate citations, judge names, or facts not present in the material."""

        else:
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

        if web_search_enabled and mode == "chat":
            yield "__WEB_SEARCHING__"
            search_results = await web_search(question)
            web_context = format_search_results(search_results)

        # Retrieval strategy per mode
        if mode == "legal":
            top_k  = 10
            chunks = await self._retrieve_async(
                "legal analysis contract parties clauses obligations rights", top_k=top_k)
        elif mode == "drafting":
            top_k  = 10
            chunks = await self._retrieve_async(
                "parties facts cause of action relief sought court jurisdiction "
                "plaint pleadings drafting conveyancing deed", top_k=top_k)
        elif mode == "brief":
            top_k  = 10
            chunks = await self._retrieve_async(
                "case facts parties procedural history issues holding ratio decidendi "
                "judgment court ruling legal principle reasoning", top_k=top_k)
        else:
            top_k  = 5
            chunks = await self._retrieve_async(question, top_k=top_k)

        if not chunks and not web_context:
            yield "No relevant context found in the uploaded documents."
            return

        context = self._context(chunks) if chunks else ""
        system  = self._system(mode)
        user    = self._user_msg(question, context, mode, web_context)

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
        if not api_key:
            yield "Error: Groq API key is required."; return
        payload = {
            "model": model, "stream": True, "temperature": 0.1, "max_tokens": 4096,
            "messages": [{"role": "system", "content": system},
                         {"role": "user",   "content": user}],
        }
        async with httpx.AsyncClient(timeout=180.0) as client:
            try:
                async with client.stream(
                    "POST", "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}",
                             "Content-Type": "application/json"},
                    json=payload,
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield f"Error: Groq {resp.status_code} — {body.decode()[:200]}"; return
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            chunk = line[6:]
                            if chunk.strip() == "[DONE]": break
                            try:
                                delta = json.loads(chunk)["choices"][0]["delta"].get("content", "")
                                if delta: yield delta
                            except (json.JSONDecodeError, KeyError): continue
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

    async def delete_document(self, doc_name: str) -> Dict:
        self.documents = [d for d in self.documents if d["name"] != doc_name]
        self.chunks    = [c for c in self.chunks    if c["source"] != doc_name]
        self._init_index()
        if self.chunks:
            emb = await hf_embed([c["text"] for c in self.chunks])
            self.index.add(emb)
            for i, c in enumerate(self.chunks): c["chunk_id"] = i
        return {"success": True, "message": f"Deleted '{doc_name}'"}







# new one my device ollama use
# import io
# import os
# import re
# import json
# from typing import AsyncGenerator, List, Dict, Any
# import httpx
# import pdfplumber
# import numpy as np
# import faiss

# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# # ── ngrok header (required when tunneling through ngrok) ─────────────────────
# NGROK_HEADERS = {"ngrok-skip-browser-warning": "true"}

# # ── HuggingFace Inference API config ─────────────────────────────────────────
# HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
#     pass

# HF_API_KEY = os.getenv("HF_API_KEY", "")
# HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_EMBEDDING_MODEL}/pipeline/feature-extraction"

# CHUNK_SIZE    = 500
# CHUNK_OVERLAP = 50

# CLOUD_MODELS = {
#     "openai": [
#         {"id": "gpt-4o",        "label": "GPT-4o"},
#         {"id": "gpt-4o-mini",   "label": "GPT-4o Mini"},
#         {"id": "gpt-4-turbo",   "label": "GPT-4 Turbo"},
#         {"id": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo"},
#     ],
#     "anthropic": [
#         {"id": "claude-opus-4-5",           "label": "Claude Opus 4.5"},
#         {"id": "claude-sonnet-4-5",         "label": "Claude Sonnet 4.5"},
#         {"id": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5"},
#     ],
#     "gemini": [
#         {"id": "gemini-2.5-flash",  "label": "Gemini 2.5 Flash"},
#         {"id": "gemini-2.0-flash",  "label": "Gemini 2.0 Flash"},
#         {"id": "gemini-1.5-pro",    "label": "Gemini 1.5 Pro"},
#         {"id": "gemini-1.5-flash",  "label": "Gemini 1.5 Flash"},
#     ],
#     "groq": [
#         {"id": "llama-3.3-70b-versatile",       "label": "LLaMA 3.3 70B"},
#         {"id": "moonshotai/kimi-k2-instruct",   "label": "Kimi K2"},
#         {"id": "llama-3.1-8b-instant",          "label": "LLaMA 3.1 8B Instant"},
#         {"id": "llama3-70b-8192",               "label": "LLaMA3 70B"},
#         {"id": "llama3-8b-8192",                "label": "LLaMA3 8B"},
#         {"id": "mixtral-8x7b-32768",            "label": "Mixtral 8x7B"},
#         {"id": "gemma2-9b-it",                  "label": "Gemma2 9B"},
#         {"id": "deepseek-r1-distill-llama-70b", "label": "DeepSeek R1 70B"},
#         {"id": "qwen-qwq-32b",                  "label": "Qwen QwQ 32B"},
#     ],
# }

# # ── Simple web search via DuckDuckGo ─────────────────────────────────────────
# DDGS_API = "https://api.duckduckgo.com/"

# async def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
#     results = []
#     try:
#         async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
#             resp = await client.get(DDGS_API, params={
#                 "q": query, "format": "json", "no_redirect": "1",
#                 "no_html": "1", "skip_disambig": "1"
#             })
#             data = resp.json()

#             if data.get("AbstractText"):
#                 results.append({
#                     "title":   data.get("Heading", "Result"),
#                     "snippet": data["AbstractText"][:400],
#                     "url":     data.get("AbstractURL", ""),
#                 })

#             for topic in data.get("RelatedTopics", []):
#                 if len(results) >= max_results:
#                     break
#                 if isinstance(topic, dict) and topic.get("Text"):
#                     results.append({
#                         "title":   topic.get("Text", "")[:80],
#                         "snippet": topic.get("Text", "")[:400],
#                         "url":     topic.get("FirstURL", ""),
#                     })

#         if not results:
#             async with httpx.AsyncClient(
#                 timeout=10.0, follow_redirects=True,
#                 headers={"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"}
#             ) as client:
#                 resp = await client.get("https://html.duckduckgo.com/html/", params={"q": query})
#                 html = resp.text
#                 snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
#                 titles   = re.findall(r'<a class="result__a"[^>]*>(.*?)</a>', html, re.DOTALL)
#                 urls     = re.findall(r'<a class="result__a" href="([^"]+)"', html)
#                 for i, snippet in enumerate(snippets[:max_results]):
#                     clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
#                     clean_title   = re.sub(r'<[^>]+>', '', titles[i]).strip() if i < len(titles) else ""
#                     url           = urls[i] if i < len(urls) else ""
#                     if clean_snippet:
#                         results.append({
#                             "title":   clean_title[:80],
#                             "snippet": clean_snippet[:400],
#                             "url":     url,
#                         })
#     except Exception as e:
#         results.append({"title": "Search Error", "snippet": f"Web search failed: {e}", "url": ""})
#     return results[:max_results]


# def format_search_results(results: List[Dict[str, str]]) -> str:
#     if not results:
#         return ""
#     parts = ["### Web Search Results\n"]
#     for i, r in enumerate(results, 1):
#         parts.append(f"[{i}] **{r['title']}**")
#         parts.append(f"    {r['snippet']}")
#         if r["url"]:
#             parts.append(f"    Source: {r['url']}")
#         parts.append("")
#     return "\n".join(parts)


# # ── HuggingFace Inference API embedding helper ────────────────────────────────

# async def hf_embed(texts: List[str]) -> np.ndarray:
#     headers = {"Content-Type": "application/json"}
#     if HF_API_KEY:
#         headers["Authorization"] = f"Bearer {HF_API_KEY}"

#     async with httpx.AsyncClient(timeout=60.0) as client:
#         resp = await client.post(
#             HF_API_URL,
#             headers=headers,
#             json={"inputs": texts, "options": {"wait_for_model": True}},
#         )
#         if resp.status_code != 200:
#             raise RuntimeError(f"HuggingFace API error {resp.status_code}: {resp.text[:200]}")
#         embeddings = resp.json()

#     return np.array(embeddings, dtype=np.float32)


# class RAGEngine:
#     def __init__(self):
#         print("Using HuggingFace Inference API for embeddings (no local model loaded).")
#         self.index     = None
#         self.chunks:    List[Dict[str, Any]] = []
#         self.documents: List[Dict[str, str]] = []
#         self.dimension = 384
#         self._init_index()

#     def _init_index(self):
#         self.index = faiss.IndexFlatL2(self.dimension)

#     def _chunk_text(self, text: str, filename: str) -> List[Dict[str, Any]]:
#         words, chunks, i = text.split(), [], 0
#         while i < len(words):
#             chunk_text = " ".join(words[i: i + CHUNK_SIZE])
#             if chunk_text.strip():
#                 chunks.append({
#                     "text":     chunk_text,
#                     "source":   filename,
#                     "chunk_id": len(self.chunks) + len(chunks)
#                 })
#             i += CHUNK_SIZE - CHUNK_OVERLAP
#         return chunks

#     async def process_pdf(self, content: bytes, filename: str) -> Dict:
#         try:
#             pages = []
#             with pdfplumber.open(io.BytesIO(content)) as pdf:
#                 total = len(pdf.pages)
#                 for p in pdf.pages:
#                     t = p.extract_text()
#                     if t:
#                         pages.append(t)
#             full = re.sub(r'\s+', ' ', "\n".join(pages)).strip()
#             if not full:
#                 return {"error": "No text could be extracted from the PDF"}
#             new_chunks = self._chunk_text(full, filename)
#             if not new_chunks:
#                 return {"error": "No chunks created from PDF"}

#             emb = await hf_embed([c["text"] for c in new_chunks])
#             self.index.add(emb)
#             self.chunks.extend(new_chunks)
#             self.documents.append({
#                 "name":   filename,
#                 "pages":  total,
#                 "chunks": len(new_chunks),
#                 "chars":  len(full)
#             })
#             return {
#                 "success":  True,
#                 "filename": filename,
#                 "pages":    total,
#                 "chunks":   len(new_chunks),
#                 "message":  f"Processed '{filename}' — {total} pages, {len(new_chunks)} chunks"
#             }
#         except Exception as e:
#             return {"error": f"Failed to process PDF: {e}"}

#     async def _retrieve_async(self, query: str, top_k: int = 5) -> List[Dict]:
#         if self.index.ntotal == 0:
#             return []
#         q = await hf_embed([query])
#         dists, idxs = self.index.search(q, min(top_k, self.index.ntotal))
#         results = []
#         for d, i in zip(dists[0], idxs[0]):
#             if i < len(self.chunks):
#                 c = self.chunks[i].copy()
#                 c["score"] = float(d)
#                 results.append(c)
#         return results

#     def _context(self, chunks: List[Dict]) -> str:
#         return "\n\n---\n\n".join(
#             f"[Source: {c['source']}, Chunk {i+1}]\n{c['text']}"
#             for i, c in enumerate(chunks)
#         )

#     # ── System prompts ────────────────────────────────────────────────────────

#     def _system(self, mode: str = "chat") -> str:
#         if mode == "legal":
#             return """You are a senior legal counsel and expert legal document analyst with 20+ years of experience.
# Your task is to produce a comprehensive legal analysis in professional legal draft format.

# STRICT OUTPUT FORMAT — you MUST produce ALL of the following sections in order:

# ═══════════════════════════════════════════════════
# LEGAL ANALYSIS MEMORANDUM
# ═══════════════════════════════════════════════════

# RE: [State the subject matter of analysis]
# DATE: [Today's date]
# PREPARED BY: AI Legal Analysis System

# ───────────────────────────────────────────────────
# I. EXECUTIVE SUMMARY
# ───────────────────────────────────────────────────
# [2-3 paragraph high-level summary of the document and key legal findings]

# ───────────────────────────────────────────────────
# II. DOCUMENT IDENTIFICATION & NATURE
# ───────────────────────────────────────────────────
# • Document Type: [Contract / Agreement / Deed / Notice / etc.]
# • Parties Involved: [List all parties with their roles]
# • Effective Date: [If mentioned]
# • Governing Law / Jurisdiction: [If mentioned]

# ───────────────────────────────────────────────────
# III. KEY PROVISIONS & CLAUSES
# ───────────────────────────────────────────────────
# [List and explain each significant clause or provision found in the document]
# For each clause: State the clause → Explain its legal effect → Note any implications

# ───────────────────────────────────────────────────
# IV. RIGHTS & OBLIGATIONS
# ───────────────────────────────────────────────────
# A. Rights Granted:
# [List all rights explicitly granted to each party]

# B. Obligations & Duties:
# [List all obligations imposed on each party]

# C. Restrictions & Prohibitions:
# [List any restrictions, non-compete, non-disclosure, or prohibited acts]

# ───────────────────────────────────────────────────
# V. RISK ASSESSMENT & RED FLAGS
# ───────────────────────────────────────────────────
# ⚠ HIGH RISK:
# [Clauses or provisions that pose significant legal risk]

# ⚠ MEDIUM RISK:
# [Clauses that require attention or negotiation]

# ✓ LOW RISK / FAVORABLE:
# [Protective or standard clauses]

# ───────────────────────────────────────────────────
# VI. LEGAL ISSUES & CONCERNS
# ───────────────────────────────────────────────────
# [Identify ambiguous language, missing clauses, enforceability issues, compliance concerns]

# ───────────────────────────────────────────────────
# VII. RECOMMENDATIONS
# ───────────────────────────────────────────────────
# [Numbered list of specific actionable recommendations for the client]
# 1.
# 2.
# 3.

# ───────────────────────────────────────────────────
# VIII. CONCLUSION
# ───────────────────────────────────────────────────
# [Final legal opinion and overall assessment]

# ───────────────────────────────────────────────────
# DISCLAIMER: This analysis is generated by an AI system for informational purposes only
# and does not constitute legal advice. Consult a qualified attorney for legal counsel.
# ═══════════════════════════════════════════════════

# Base your analysis STRICTLY on the provided document context. Do not fabricate clauses or provisions not present in the document. If certain information is not available in the document, explicitly state "Not specified in the document."
# """
#         else:
#             return (
#                 "You are a helpful AI assistant. Answer the user's question based on the "
#                 "provided context. If web search results are included, use them to supplement "
#                 "the document context. Be concise, accurate, and cite sources when relevant. "
#                 "If the answer cannot be found in any context, say so clearly."
#             )

#     # ── User message builders ─────────────────────────────────────────────────

#     def _user_msg(self, question: str, context: str, mode: str = "chat",
#                   web_context: str = "") -> str:
#         if mode == "legal":
#             return f"""DOCUMENT CONTEXT:
# {context}

# TASK: {question if question.strip() else "Perform a complete legal analysis of the above document in the specified legal draft format."}

# Produce a thorough legal analysis memorandum following the exact format specified in your instructions."""

#         parts = []
#         if context:
#             parts.append(f"DOCUMENT CONTEXT:\n{context}")
#         if web_context:
#             parts.append(f"WEB SEARCH CONTEXT:\n{web_context}")
#         combined = "\n\n".join(parts)
#         return f"{combined}\n\nQUESTION: {question}\n\nANSWER:"

#     # ── Main query stream entry point ─────────────────────────────────────────

#     async def query_stream(
#         self, question: str, provider: str,
#         model: str, api_key: str = "",
#         mode: str = "chat",
#         web_search_enabled: bool = False
#     ) -> AsyncGenerator[str, None]:

#         web_context = ""

#         if web_search_enabled and mode == "chat":
#             yield "__WEB_SEARCHING__"
#             search_results = await web_search(question)
#             web_context = format_search_results(search_results)

#         if mode == "legal":
#             top_k  = 10
#             chunks = await self._retrieve_async(
#                 "legal analysis contract parties clauses obligations rights", top_k=top_k)
#             if not chunks:
#                 yield "No relevant context found in the uploaded documents."
#                 return
#             context = self._context(chunks)
#         else:
#             top_k  = 5
#             chunks = await self._retrieve_async(question, top_k=top_k)
#             if not chunks and not web_context:
#                 yield "No relevant context found in the uploaded documents."
#                 return
#             context = self._context(chunks) if chunks else ""

#         system = self._system(mode)
#         user   = self._user_msg(question, context, mode, web_context)

#         if provider == "ollama":
#             async for t in self._ollama_msg(system, user, model):
#                 yield t
#         elif provider == "openai":
#             async for t in self._openai_msg(system, user, model, api_key):
#                 yield t
#         elif provider == "anthropic":
#             async for t in self._anthropic_msg(system, user, model, api_key):
#                 yield t
#         elif provider == "gemini":
#             async for t in self._gemini_msg(system, user, model, api_key):
#                 yield t
#         elif provider == "groq":
#             async for t in self._groq_msg(system, user, model, api_key):
#                 yield t
#         else:
#             yield f"Unknown provider: {provider}"

#     # ── Provider implementations ──────────────────────────────────────────────

#     async def _ollama_msg(self, system: str, user: str, model: str):
#         prompt = f"{system}\n\n{user}"
#         async with httpx.AsyncClient(timeout=180.0) as client:
#             try:
#                 async with client.stream(
#                     "POST",
#                     f"{OLLAMA_BASE_URL}/api/generate",
#                     headers=NGROK_HEADERS,
#                     json={
#                         "model":   model,
#                         "prompt":  prompt,
#                         "stream":  True,
#                         "options": {
#                             "temperature": 0.1,
#                             "top_p":       0.9,
#                             "num_predict": 4096
#                         }
#                     }
#                 ) as resp:
#                     if resp.status_code != 200:
#                         yield f"Error: Ollama returned status {resp.status_code}"
#                         return
#                     async for line in resp.aiter_lines():
#                         if line.strip():
#                             try:
#                                 data = json.loads(line)
#                                 if "response" in data:
#                                     yield data["response"]
#                                 if data.get("done"):
#                                     break
#                             except json.JSONDecodeError:
#                                 continue
#             except httpx.ConnectError:
#                 yield "Error: Cannot connect to Ollama. Make sure ngrok is running and OLLAMA_BASE_URL is set correctly."
#             except Exception as e:
#                 yield f"Error: {e}"

#     async def _openai_msg(self, system: str, user: str, model: str, api_key: str):
#         if not api_key:
#             yield "Error: OpenAI API key is required."
#             return
#         payload = {
#             "model":       model,
#             "stream":      True,
#             "temperature": 0.1,
#             "max_tokens":  4096,
#             "messages": [
#                 {"role": "system", "content": system},
#                 {"role": "user",   "content": user}
#             ]
#         }
#         async with httpx.AsyncClient(timeout=180.0) as client:
#             try:
#                 async with client.stream(
#                     "POST",
#                     "https://api.openai.com/v1/chat/completions",
#                     headers={
#                         "Authorization": f"Bearer {api_key}",
#                         "Content-Type":  "application/json"
#                     },
#                     json=payload
#                 ) as resp:
#                     if resp.status_code != 200:
#                         body = await resp.aread()
#                         yield f"Error: OpenAI {resp.status_code} — {body.decode()[:200]}"
#                         return
#                     async for line in resp.aiter_lines():
#                         if line.startswith("data: "):
#                             chunk = line[6:]
#                             if chunk.strip() == "[DONE]":
#                                 break
#                             try:
#                                 delta = json.loads(chunk)["choices"][0]["delta"].get("content", "")
#                                 if delta:
#                                     yield delta
#                             except (json.JSONDecodeError, KeyError):
#                                 continue
#             except Exception as e:
#                 yield f"Error: {e}"

#     async def _anthropic_msg(self, system: str, user: str, model: str, api_key: str):
#         if not api_key:
#             yield "Error: Anthropic API key is required."
#             return
#         payload = {
#             "model":      model,
#             "max_tokens": 4096,
#             "stream":     True,
#             "system":     system,
#             "messages":   [{"role": "user", "content": user}]
#         }
#         async with httpx.AsyncClient(timeout=180.0) as client:
#             try:
#                 async with client.stream(
#                     "POST",
#                     "https://api.anthropic.com/v1/messages",
#                     headers={
#                         "x-api-key":         api_key,
#                         "anthropic-version": "2023-06-01",
#                         "Content-Type":      "application/json"
#                     },
#                     json=payload
#                 ) as resp:
#                     if resp.status_code != 200:
#                         body = await resp.aread()
#                         yield f"Error: Anthropic {resp.status_code} — {body.decode()[:200]}"
#                         return
#                     async for line in resp.aiter_lines():
#                         if line.startswith("data: "):
#                             try:
#                                 data = json.loads(line[6:])
#                                 if data.get("type") == "content_block_delta":
#                                     yield data["delta"].get("text", "")
#                             except (json.JSONDecodeError, KeyError):
#                                 continue
#             except Exception as e:
#                 yield f"Error: {e}"

#     async def _gemini_msg(self, system: str, user: str, model: str, api_key: str):
#         if not api_key:
#             yield "Error: Gemini API key is required."
#             return
#         url = (
#             f"https://generativelanguage.googleapis.com/v1beta/models/"
#             f"{model}:streamGenerateContent?alt=sse&key={api_key}"
#         )
#         payload = {
#             "contents": [{"parts": [{"text": f"{system}\n\n{user}"}]}],
#             "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096}
#         }
#         async with httpx.AsyncClient(timeout=180.0) as client:
#             try:
#                 async with client.stream("POST", url, json=payload) as resp:
#                     if resp.status_code != 200:
#                         body = await resp.aread()
#                         yield f"Error: Gemini {resp.status_code} — {body.decode()[:200]}"
#                         return
#                     async for line in resp.aiter_lines():
#                         if line.startswith("data: "):
#                             try:
#                                 data = json.loads(line[6:])
#                                 text = data["candidates"][0]["content"]["parts"][0].get("text", "")
#                                 if text:
#                                     yield text
#                             except (json.JSONDecodeError, KeyError, IndexError):
#                                 continue
#             except Exception as e:
#                 yield f"Error: {e}"

#     async def _groq_msg(self, system: str, user: str, model: str, api_key: str):
#         if not api_key:
#             yield "Error: Groq API key is required."
#             return
#         payload = {
#             "model":       model,
#             "stream":      True,
#             "temperature": 0.1,
#             "max_tokens":  4096,
#             "messages": [
#                 {"role": "system", "content": system},
#                 {"role": "user",   "content": user},
#             ],
#         }
#         async with httpx.AsyncClient(timeout=180.0) as client:
#             try:
#                 async with client.stream(
#                     "POST",
#                     "https://api.groq.com/openai/v1/chat/completions",
#                     headers={
#                         "Authorization": f"Bearer {api_key}",
#                         "Content-Type":  "application/json",
#                     },
#                     json=payload,
#                 ) as resp:
#                     if resp.status_code != 200:
#                         body = await resp.aread()
#                         yield f"Error: Groq {resp.status_code} — {body.decode()[:200]}"
#                         return
#                     async for line in resp.aiter_lines():
#                         if line.startswith("data: "):
#                             chunk = line[6:]
#                             if chunk.strip() == "[DONE]":
#                                 break
#                             try:
#                                 delta = json.loads(chunk)["choices"][0]["delta"].get("content", "")
#                                 if delta:
#                                     yield delta
#                             except (json.JSONDecodeError, KeyError):
#                                 continue
#             except httpx.ConnectError:
#                 yield "Error: Cannot connect to Groq API."
#             except Exception as e:
#                 yield f"Error: {e}"

#     # ── Utility methods ───────────────────────────────────────────────────────

#     async def get_ollama_models(self) -> List[str]:
#         try:
#             async with httpx.AsyncClient(timeout=5.0) as client:
#                 resp = await client.get(
#                     f"{OLLAMA_BASE_URL}/api/tags",
#                     headers=NGROK_HEADERS
#                 )
#                 if resp.status_code == 200:
#                     return [m["name"] for m in resp.json().get("models", [])]
#         except Exception:
#             pass
#         return []

#     async def check_ollama(self) -> bool:
#         try:
#             async with httpx.AsyncClient(timeout=3.0) as client:
#                 resp = await client.get(
#                     f"{OLLAMA_BASE_URL}/api/tags",
#                     headers=NGROK_HEADERS
#                 )
#                 return resp.status_code == 200
#         except Exception:
#             return False

#     def has_documents(self) -> bool:
#         return self.index.ntotal > 0

#     def get_documents(self) -> List[Dict]:
#         return self.documents

#     async def delete_document(self, doc_name: str) -> Dict:
#         self.documents = [d for d in self.documents if d["name"] != doc_name]
#         self.chunks    = [c for c in self.chunks    if c["source"] != doc_name]
#         self._init_index()
#         if self.chunks:
#             emb = await hf_embed([c["text"] for c in self.chunks])
#             self.index.add(emb)
#             for i, c in enumerate(self.chunks):
#                 c["chunk_id"] = i
#         return {"success": True, "message": f"Deleted '{doc_name}'"}
