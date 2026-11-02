# AI Exam Management System

A Flask-based web application for AI-powered exam management, supporting both students and teachers with intelligent tools for exam generation, practice, analysis, and learning assistance.

---

## Tech Stack

- **Backend**: Python / Flask
- **Database**: SQLite (`users.db`)
- **AI Generation**: Groq API (`llama-3.3-70b-versatile`) + Google Gemini (fallback)
- **Vector Search**: FAISS + `all-MiniLM-L6-v2` sentence embeddings (RAG)
- **PDF Export**: ReportLab
- **Frontend**: Vanilla JS + Chart.js (analytics) + Font Awesome

---

## Project Structure

```
app.py                          # Main Flask application (5468 lines)
config.py                       # API keys, SMTP, secret key
rag_retriever.py                # FAISS-based RAG retriever singleton
build_vector_db.py              # Script to build the FAISS vector index
ingest_textbooks.py             # Textbook ingestion pipeline
ingest_past_papers.py           # Past paper ingestion pipeline
llama_inference.py              # TinyLlama inference wrapper (legacy)
similarity_semantic.py          # Semantic similarity matcher
requirements.txt                # Python dependencies

templates/
  login.html                    # Login page
  signup.html                   # Signup page
  student_dashboard.html        # Student module hub
  teacher_dashboard.html        # Teacher module hub
  past_papers.html              # Past papers browser
  similarity_checker.html       # Similarity checker (teacher)
  similarity_results.html       # Similarity results view
  difficulty_checker.html       # Difficulty analyzer (teacher)
  ai_paper_generator.html       # AI paper generator (teacher)
  mock_exam_generator.html      # Mock exam generator (student)
  attempt_mock_exam.html        # Mock exam attempt interface
  solution_explainer.html       # AI solution explainer (student)
  exam_bee.html                 # Exam Bee MCQ generator (student)
  chatbot.html                  # AI chatbot (student)
  analytics_dashboard.html      # Analytics dashboard (student)

textbooks/                      # Uploaded textbook PDFs by course
uploads/                        # Uploaded files (past papers, checks)
vector_db/                      # Pre-built FAISS index files
vector_indexes/                 # Per-textbook FAISS indexes
```

---

## Database Tables (`app.py` lines 128–400)

| Table | Purpose |
|---|---|
| `users` | Student and teacher accounts |
| `courses` | Course codes and names |
| `exam_types` | sessional1, sessional2, final |
| `past_papers` | Uploaded past paper metadata |
| `similarity_checks` | Teacher similarity check records |
| `similarity_results` | Per-paper similarity results |
| `question_difficulty_labels` | Teacher-labeled training data |
| `textbooks` | Uploaded textbook metadata |
| `chapters` | Extracted chapter content |
| `text_chunks` | Chunked text for vector search |
| `chunk_embeddings` | FAISS embeddings per chunk |
| `vector_indexes` | FAISS index file paths |
| `generated_papers` | AI-generated exam papers |
| `generated_questions` | Questions in generated papers |
| `mock_papers` | Student mock exam papers |
| `mock_questions` | Questions in mock papers |
| `used_questions` | Hash-based duplicate prevention |
| `mock_exam_attempts` | Attempt scores (new) |
| `activity_log` | Student activity tracking (new) |

---

## Authentication & User Management

| Route | Lines | Description |
|---|---|---|
| `GET/POST /signup` | 1631–1696 | Student/teacher registration |
| `GET /verify/<token>` | 1697–1716 | Email verification for teachers |
| `GET/POST /login` | 1717–1755 | Login with role selection |
| `GET /logout` | 1846–1851 | Session clear and redirect |

**Rules:**
- Students: no email verification required
- Teachers: must have `edu.pk` email, email verification required
- Predefined teacher emails: `i221935@nu.edu.pk`, `i221927@nu.edu.pk`

---

## Student Dashboard

| Route | Lines | Template |
|---|---|---|
| `GET /student-dashboard` | 1756–1763 | `student_dashboard.html` |

Displays 6 module cards: Past Papers, Mock Exam Generator, AI Solution Explainer, Exam Bee, AI Chatbot, Analytics Dashboard.

---

## Teacher Dashboard

| Route | Lines | Template |
|---|---|---|
| `GET /teacher-dashboard` | 1764–1771 | `teacher_dashboard.html` |

---

## Module 1 — Past Papers

| Route | Lines | Description |
|---|---|---|
| `GET /past-papers` | 1772–1809 | Browse and filter past papers |
| `GET /files/<paper_id>` | 1810–1845 | Download a past paper PDF (logs activity) |

Students can filter by course and exam type. Downloads are logged to `activity_log`.

---

## Module 2 — Similarity Checker (Teacher)

| Route | Lines | Description |
|---|---|---|
| `GET /similarity-checker` | 1901–1925 | Similarity checker page |
| `POST /check-similarity` | 1926–2139 | Upload paper and run similarity check |
| `GET /similarity-results/<id>` | 2140–2181 | View detailed results |

**Logic:** Extracts questions from uploaded PDF, compares against past papers using lexical similarity + optional semantic embeddings. Detects same-paper uploads (≥80% question match → 100% similarity).

---

## Module 3 — Difficulty Analyzer (Teacher)

| Route | Lines | Description |
|---|---|---|
| `GET /difficulty-checker` | 2182–2257 | Difficulty analyzer page |
| `POST /api/fetch-questions-from-papers` | 2258–2328 | Extract questions for labeling |
| `POST /difficulty-labels` | 2329–2385 | Add manual difficulty label |
| `POST /difficulty-analyze` | 2386–2500 | Analyze uploaded paper difficulty |

**Logic:** Three-tier prediction — ML model (scikit-learn LogisticRegression + TF-IDF) if ≥9 labeled samples, zero-shot classifier (BART-MNLI) as fallback, heuristic (word count + cognitive keyword scoring) as last resort.

---

## Module 4 — AI Paper Generator (Teacher)

| Route | Lines | Description |
|---|---|---|
| `GET /ai-paper-generator` | 3281–3325 | Paper generator page |
| `POST /api/upload-textbook` | 3795–3922 | Upload and index a textbook PDF |
| `POST /api/generate-paper` | 3564–3794 | Generate exam paper with AI |
| `GET /api/generated-papers` | 5031–5061 | List teacher's generated papers |
| `GET /api/paper/<id>` | 5062–5098 | Get paper questions |
| `GET /api/export-paper/<id>` | 5099–5228 | Export paper as PDF |

**Generation pipeline** (`app.py` lines 2647–3563):
- `detect_chapters_in_pdf()` — line 2647: multi-pattern chapter extraction from PDF
- `generate_questions_with_gemini_rag()` — primary generator using Gemini API
- `generate_questions_with_groq()` — line 2971: Groq fallback
- `generate_questions_with_openai()` — OpenAI fallback
- `generate_questions_with_huggingface()` — template-based last resort
- `generate_questions_hybrid()` — line 3480: dispatcher (Gemini → Groq → OpenAI → template)

---

## Module 5 — Mock Exam Generator (Student)

| Route | Lines | Description |
|---|---|---|
| `GET /mock-exam-generator` | 3983–3992 | Mock exam generator page |
| `POST /api/generate-mock-paper` | 4311–4534 | Generate mock exam with AI |
| `GET /api/get-mock-papers` | 4535–4565 | List student's mock papers |
| `GET /api/download-mock-paper/<id>` | 4566–4667 | Download mock paper as PDF |
| `GET /attempt-mock-exam/<id>` | 4668–4726 | Attempt a mock exam |
| `POST /api/submit-mock-exam` | 4727–4800 | Submit answers and get score |

**Question types supported:** MCQ, Short Answer, Long Answer, Case Study, Analytical Reasoning

**Generation pipeline:** `generate_questions_hybrid_mock()` — line 3207 (Gemini → Groq → OpenAI → template)

**Scoring:** MCQ answers auto-graded. Score saved to `mock_exam_attempts` table and logged to `activity_log`.

---

## Module 6 — AI Solution Explainer (Student)

| Route | Lines | Description |
|---|---|---|
| `GET /solution-explainer` | 4802–4856 | Solution explainer page |
| `POST /api/solve-paper` | 4858–5029 | Upload exam PDF and get solutions |

**Logic:**
1. Extract text from uploaded PDF
2. Parse questions using multi-strategy splitter (`_extract_questions_for_solver()`)
3. For each question: retrieve relevant textbook chunks via FAISS RAG
4. Generate solution + explanation using Groq (primary) or Gemini (fallback)
5. Return structured JSON with question, solution, and explanation per item

---

## Module 7 — Exam Bee (Student) ⭐ New

| Route | Lines | Description |
|---|---|---|
| `GET /exam-bee` | 3994–4001 | Exam Bee page |
| `POST /api/exam-bee/generate` | 4003–4217 | Generate Bloom-aligned MCQs |
| `GET /api/exam-bee/download/<id>` | 4219–4309 | Download generated MCQs as PDF |

**Template:** `templates/exam_bee.html`

**What it does:** Generates MCQs from selected textbook chapters, aligned to a chosen Bloom's taxonomy level and difficulty. Supports three MCQ types:

| MCQ Type | Prompt Logic |
|---|---|
| Standard | Classic knowledge-check — one concept, direct question |
| Case Study | 2–4 sentence scenario first, question tests application within that scenario |
| Analytical Reasoning | Multi-step reasoning required — no direct recall, logically tempting distractors |

**User controls:**
- Course → Textbook → Chapters (multi-select)
- Bloom's Level: Remember / Understand / Apply / Analyze
- Difficulty: Easy / Medium / Hard
- MCQ Type: Standard / Case Study / Analytical Reasoning
- Number of questions: 1–20

**Generation:** Groq primary (`llama-3.3-70b-versatile`), Gemini fallback. Results saved to `generated_papers` + `generated_questions` tables. Activity logged.

---

## Module 8 — AI Chatbot (Student) ⭐ New

| Route | Lines | Description |
|---|---|---|
| `GET /chatbot` | 5387–5398 | Chatbot page |
| `POST /api/chatbot` | 5400–5440 | Handle chat message and return answer |

**Template:** `templates/chatbot.html`

**What it does:** A multi-turn conversational AI tutor that answers student questions grounded in uploaded textbook content.

**Flow:**
1. Student types a question (optionally filtered by course)
2. Backend retrieves top-5 relevant textbook chunks via FAISS RAG (`rag_retriever.py`)
3. Chunks injected into system prompt as context
4. Last 6 conversation turns included for multi-turn memory
5. Groq generates the answer
6. Response rendered with basic markdown support (bold, code, lists)

**UI features:**
- Course filter dropdown (focuses RAG retrieval to one course)
- Suggestion chips for quick-start questions
- Typing indicator
- New Chat button (clears history)
- Markdown rendering in bot responses

---

## Module 9 — Analytics Dashboard (Student) ⭐ New

| Route | Lines | Description |
|---|---|---|
| `GET /analytics` | 5230–5237 | Analytics dashboard page |
| `GET /api/analytics/data` | 5239–5385 | Return all analytics data as JSON |

**Template:** `templates/analytics_dashboard.html`

**What it tracks:**

| Metric | Source Table |
|---|---|
| Total mock exams generated | `mock_papers` |
| Total Exam Bee sessions | `generated_papers` (mode=exam_bee) |
| Total exams attempted | `mock_exam_attempts` |
| Total past papers downloaded | `activity_log` |
| Average score | `mock_exam_attempts` |
| Score trend (last 10) | `mock_exam_attempts` |
| Mock exams by course | `mock_papers` + `courses` |
| Average score by course | `mock_exam_attempts` + `courses` |
| Question type distribution | `mock_questions` |
| Difficulty distribution | `mock_questions` |
| Bloom's level usage | `generated_papers` (exam_bee titles) |
| Recent activity feed | `activity_log` |

**Charts (Chart.js):**
- Score trend — line chart
- Mock exams by course — bar chart
- Average score by course — horizontal bar (red < 50%, amber < 70%, green ≥ 70%)
- Question types attempted — doughnut chart
- Difficulty distribution — pie chart
- Bloom's levels used — radar chart

**Recommendations engine (rule-based):**
- Courses with avg score < 50% → "Focus recommended"
- Bloom's levels with 0 sessions → "Haven't tried X level yet"
- No attempts yet → nudge to attempt a mock exam
- No Exam Bee sessions → nudge to try Exam Bee

**Activity logging** — events written to `activity_log`:
- Mock exam generated → `generate / mock_exam` (line 4497)
- Mock exam attempted + scored → `attempt / mock_exam` (line 4762)
- Exam Bee session → `generate / exam_bee` (line 4210)
- Past paper downloaded → `download / past_papers` (line 1826)

---

## Shared API Endpoints

| Route | Lines | Description |
|---|---|---|
| `GET /api/courses` | 3923–3934 | List all courses |
| `GET /api/exam-types` | 3935–3946 | List all exam types |
| `GET /api/textbooks` | 3947–3964 | List textbooks (filter by course) |
| `GET /api/chapters` | 3965–3982 | List chapters (filter by textbook) |
| `GET /api/chapters/<id>` | 3346–3405 | Get chapters for a textbook (auto-extracts) |
| `POST /api/chapters/<id>/refresh` | 3406–3430 | Force re-extract chapters from PDF |

---

## RAG System (`rag_retriever.py`)

Singleton class `RAGRetriever` — loads once, queries many times.

- **Index:** `vector_db/faiss_index.bin` (pre-built by `build_vector_db.py`)
- **Metadata:** `vector_db/metadata.pkl`
- **Embedding model:** `all-MiniLM-L6-v2` (384 dimensions)
- **`retrieve(query, course_code, top_k)`** — returns top-k chunks with similarity scores
- **`get_context(query, course_code, top_k, max_chars)`** — returns concatenated string ready for prompt injection

Used by: Chatbot (line 5408), Solution Explainer (line 4893).

---

## Configuration (`config.py`)

| Key | Purpose |
|---|---|
| `SECRET_KEY` | Flask session encryption |
| `SMTP_*` | Gmail SMTP for teacher email verification |
| `GROQ_API_KEY` | Primary AI generation (free tier) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` |
| `GEMINI_API_KEY` | Fallback AI generation |
| `GEMINI_MODEL` | `gemini-2.5-flash` |
| `OPENAI_API_KEY` | Optional second fallback |

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Build vector database from textbooks
python build_vector_db.py

# Run the application
python app.py
```

App runs at `http://localhost:5000` with `debug=True, use_reloader=False`.

---

## AI Generation Priority Order

All question generation follows this fallback chain:

```
Gemini (gemini-2.5-flash)
    ↓ (quota exceeded / error)
Groq (llama-3.3-70b-versatile)
    ↓ (unavailable)
OpenAI (gpt-4-turbo-preview)
    ↓ (unavailable)
Template-based fallback (no AI)
```

Chatbot and Solution Explainer use: **Groq → Gemini** (no template fallback).
