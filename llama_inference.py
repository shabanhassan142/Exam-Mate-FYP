"""
TinyLlama + FAISS RAG -- Question Generator
============================================
Model      : tinyllama_model (1)/
Embeddings : vector_db/faiss_index.bin  (all-MiniLM-L6-v2)

Public API:
    generate_questions_with_tinyllama(topic, course_code, num_questions,
                                      question_type, chapter_text="")
    is_tinyllama_available()
"""

import re
import time
from pathlib import Path

from rag_retriever import retriever

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "tinyllama_model (1)"
MODEL_DIR  = Path(MODEL_PATH)
# Context fed to the model — keep short so generation is fast
CONTEXT_CHARS = 800
# Tokens to generate — enough for all questions in one pass
MAX_NEW_TOKENS = 400
# ─────────────────────────────────────────────────────────────────────────────

_pipe    = None   # HuggingFace pipeline (loaded once)
_backend = None   # "transformers" | None


# ── Model loader ──────────────────────────────────────────────────────────────

def _load_model():
    global _pipe, _backend
    if _pipe is not None:
        return True
    if not MODEL_DIR.exists():
        print(f"  TinyLlama folder not found: {MODEL_PATH!r}")
        return False
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch
        print("  Loading TinyLlama (transformers)...")
        tok  = AutoTokenizer.from_pretrained(MODEL_PATH)
        mdl  = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.float32, low_cpu_mem_usage=True
        )
        # Do NOT pass max_new_tokens here — pass it at call time instead
        _pipe    = pipeline("text-generation", model=mdl, tokenizer=tok, device=-1)
        _backend = "transformers"
        print("  TinyLlama loaded (transformers backend)")
        return True
    except Exception as e:
        print(f"  TinyLlama load failed: {e}")
        return False


# ── Prompt ────────────────────────────────────────────────────────────────────

def _build_prompt(context: str, q_type: str, num: int) -> str:
    """Compact, numbered-list prompt — easier for TinyLlama to follow."""
    if q_type == "mcq":
        fmt = (
            "Write {n} MCQ questions. Format each as:\n"
            "{i}. <question>\n"
            "A) <option>  B) <option>  C) <option>  D) <option>\n"
            "Answer: <letter>\n"
        )
    elif q_type == "short_answer":
        fmt = (
            "Write {n} short-answer questions (one sentence each). Format:\n"
            "{i}. <question>\n"
        )
    else:
        fmt = (
            "Write {n} long-answer questions. Format:\n"
            "{i}. <question>\n"
        )

    header = fmt.format(n=num, i="1")
    return (
        f"Context (use only this):\n{context}\n\n"
        f"{header}\n"
    )


# ── Raw generation ────────────────────────────────────────────────────────────

def _raw_generate(prompt: str) -> str:
    out  = _pipe(
        prompt,
        max_new_tokens    = MAX_NEW_TOKENS,
        do_sample         = True,
        temperature       = 0.7,
        top_p             = 0.9,
        repetition_penalty= 1.1,
    )
    full = out[0]["generated_text"]
    return full[len(prompt):].strip()


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_questions(raw: str, q_type: str, num: int) -> list:
    """
    Robust parser — handles:
      1. Question text  /  Q1. text  /  1) text  /  1. text
    """
    questions = []

    # Split on any numbered marker: "1.", "1)", "Q1.", "Q1)", "Question 1."
    blocks = re.split(
        r'(?:^|\n)\s*(?:Q(?:uestion)?\s*)?\d+[\.\)]\s*',
        raw, flags=re.IGNORECASE
    )
    blocks = [b.strip() for b in blocks if b.strip()]

    for i, block in enumerate(blocks[:num]):
        q = {
            "question_number": i + 1,
            "question_text"  : "",
            "question_type"  : q_type,
            "marks"          : 5 if q_type in ("mcq", "short_answer") else 10,
            "answer_hint"    : "",
            "options"        : [],
            "correct_answer" : "",
        }

        if q_type == "mcq":
            lines = block.splitlines()
            q_lines, opt_lines, ans_line = [], [], ""
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if re.match(r'^[A-D][\)\.\s]', line):
                    opt_lines.append(re.sub(r'^[A-D][\)\.\s]+', '', line).strip())
                elif re.match(r'(?i)answer\s*:', line):
                    ans_line = line
                else:
                    q_lines.append(line)
            q["question_text"] = " ".join(q_lines).strip()
            q["options"]       = opt_lines[:4]
            while len(q["options"]) < 4:
                q["options"].append(f"Option {chr(65 + len(q['options']))}")
            m = re.search(r'\b([A-D])\b', ans_line)
            q["correct_answer"] = m.group(1) if m else "A"
        else:
            # For short/long: first non-empty lines are the question
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            q["question_text"] = " ".join(lines[:3])  # take up to 3 lines

        if q["question_text"]:
            questions.append(q)

    return questions


def _fallback_questions(raw: str, q_type: str, num: int) -> list:
    """
    If the parser finds nothing, split on sentence boundaries and
    turn each sentence into a question.
    """
    sentences = [s.strip() for s in re.split(r'[.!?]\s+', raw) if len(s.strip()) > 20]
    questions = []
    for i, sent in enumerate(sentences[:num]):
        q = {
            "question_number": i + 1,
            "question_text"  : sent[:300],
            "question_type"  : q_type,
            "marks"          : 5 if q_type in ("mcq", "short_answer") else 10,
            "answer_hint"    : "",
            "options"        : (["Option A", "Option B", "Option C", "Option D"]
                                if q_type == "mcq" else []),
            "correct_answer" : "A" if q_type == "mcq" else "",
        }
        questions.append(q)
    return questions


# ── Public API ────────────────────────────────────────────────────────────────

def generate_questions_with_tinyllama(
    topic: str,
    course_code: str,
    num_questions: int,
    question_type: str,
    chapter_text: str = "",
) -> list:
    """
    Generate exam questions using TinyLlama + FAISS RAG.
    Returns a list of question dicts (may be fewer than num_questions if
    the model output is short — caller pads/trims as needed).
    """
    if not _load_model():
        return []

    # ── Get context ───────────────────────────────────────────────────────────
    if chapter_text.strip():
        context = chapter_text[:CONTEXT_CHARS]
    else:
        if not retriever.is_ready():
            try:
                retriever.load()
            except FileNotFoundError:
                print("  Vector DB missing — run build_vector_db.py first")
                return []
        context = retriever.get_context(
            topic, course_code=course_code, top_k=3, max_chars=CONTEXT_CHARS
        )

    if not context.strip():
        print(f"  No context for: {topic}")
        return []

    # ── Generate ──────────────────────────────────────────────────────────────
    print(f"  Generating {num_questions}x {question_type} with TinyLlama...")
    t0     = time.time()
    prompt = _build_prompt(context, question_type, num_questions)
    raw    = _raw_generate(prompt)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  |  raw output: {len(raw)} chars")

    questions = _parse_questions(raw, question_type, num_questions)

    if not questions:
        print("  Parser found 0 questions — using sentence fallback")
        questions = _fallback_questions(raw, question_type, num_questions)

    print(f"  Parsed {len(questions)} question(s)")
    return questions


def is_tinyllama_available() -> bool:
    return MODEL_DIR.exists()
