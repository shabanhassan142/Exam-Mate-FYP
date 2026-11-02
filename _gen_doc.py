"""Generate FYP project documentation as a .docx file."""
try:
    from docx import Document
    from docx.shared import Pt, RGBColor, Inches, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.style import WD_STYLE_TYPE
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    import datetime
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'python-docx', '-q'])
    from docx import Document
    from docx.shared import Pt, RGBColor, Inches, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    import datetime

doc = Document()

# ── Page margins ──────────────────────────────────────────────────────────────
for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(3.0)
    section.right_margin  = Cm(2.5)

# ── Helper: set paragraph shading ────────────────────────────────────────────
def shade_paragraph(para, hex_color):
    pPr = para._p.get_or_add_pPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    pPr.append(shd)

# ── Helper: add horizontal rule ───────────────────────────────────────────────
def add_hr(doc):
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), '14B8A6')
    pBdr.append(bottom)
    pPr.append(pBdr)
    p.paragraph_format.space_after = Pt(4)

# ── Helper: styled heading ────────────────────────────────────────────────────
def add_heading(doc, text, level=1, color='0F1117'):
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in h.runs:
        run.font.color.rgb = RGBColor.from_string(color)
    return h

# ── Helper: body paragraph ────────────────────────────────────────────────────
def add_body(doc, text, bold=False, italic=False, color=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(11)
    run.bold   = bold
    run.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    p.paragraph_format.space_after = Pt(6)
    return p

# ── Helper: bullet ────────────────────────────────────────────────────────────
def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(style='List Bullet')
    run = p.add_run(text)
    run.font.size = Pt(10.5)
    p.paragraph_format.left_indent = Inches(0.25 * (level + 1))
    p.paragraph_format.space_after = Pt(3)
    return p

# ── Helper: info box ──────────────────────────────────────────────────────────
def add_info_box(doc, label, value):
    p = doc.add_paragraph()
    r1 = p.add_run(f'{label}: ')
    r1.bold = True
    r1.font.size = Pt(10.5)
    r1.font.color.rgb = RGBColor.from_string('0D9488')
    r2 = p.add_run(value)
    r2.font.size = Pt(10.5)
    p.paragraph_format.space_after = Pt(4)
    shade_paragraph(p, 'F0FDFA')

# ── Helper: section box (teal left border via shading) ───────────────────────
def add_section_box(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(10.5)
    run.italic = True
    shade_paragraph(p, 'F0FDFA')
    p.paragraph_format.left_indent  = Inches(0.2)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(8)

# ═══════════════════════════════════════════════════════════════════════════════
# COVER PAGE
# ═══════════════════════════════════════════════════════════════════════════════
cover = doc.add_paragraph()
cover.alignment = WD_ALIGN_PARAGRAPH.CENTER
cover.paragraph_format.space_before = Pt(60)
r = cover.add_run('AI EXAM MANAGEMENT SYSTEM')
r.font.size  = Pt(28)
r.font.bold  = True
r.font.color.rgb = RGBColor.from_string('14B8A6')

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r2 = sub.add_run('Final Year Project — Complete Technical Documentation')
r2.font.size  = Pt(14)
r2.font.color.rgb = RGBColor.from_string('64748B')

doc.add_paragraph()
doc.add_paragraph()

meta_items = [
    ('Project Title',    'AI Exam Management System'),
    ('Project Type',     'Final Year Project (FYP)'),
    ('Technology Stack', 'Python · Flask · SQLite · FAISS · TinyLlama · Groq API · Gemini API'),
    ('Document Date',    datetime.date.today().strftime('%B %d, %Y')),
]
for label, val in meta_items:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f'{label}:  ')
    r.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor.from_string('0D9488')
    r2 = p.add_run(val)
    r2.font.size = Pt(11)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PROJECT OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, '1. Project Overview', 1, '14B8A6')
add_hr(doc)
add_body(doc,
    'The AI Exam Management System is a full-stack web application built as a Final Year Project. '
    'It serves two user roles — Students and Teachers — and provides an end-to-end intelligent '
    'platform for exam preparation, paper generation, difficulty analysis, and similarity checking. '
    'The system integrates large language models (LLMs), Retrieval-Augmented Generation (RAG), '
    'FAISS vector search, and machine learning classifiers to deliver AI-powered academic tools.')

add_heading(doc, '1.1 Problem Statement', 2, '0D9488')
add_body(doc,
    'Traditional exam management is manual, time-consuming, and prone to repetition. Teachers '
    'spend hours creating balanced papers, checking for similarity with past papers, and evaluating '
    'question difficulty. Students lack personalized practice tools that align with their course '
    'syllabus. This system addresses all these pain points through automation and AI.')

add_heading(doc, '1.2 Objectives', 2, '0D9488')
objectives = [
    'Automate exam paper generation using AI with proper difficulty distribution.',
    'Enable students to generate personalized mock exams from course textbooks.',
    'Provide AI-powered solutions and explanations for uploaded exam papers.',
    'Detect similarity between new exam papers and existing past papers.',
    'Analyze and predict question difficulty using ML classifiers.',
    'Generate Bloom\'s taxonomy-aligned MCQs for targeted cognitive practice.',
    'Maintain a structured past papers library accessible to all users.',
]
for obj in objectives:
    add_bullet(doc, obj)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SYSTEM ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, '2. System Architecture', 1, '14B8A6')
add_hr(doc)
add_body(doc,
    'The system follows a monolithic Flask architecture with a SQLite database, a FAISS vector '
    'index for semantic retrieval, and multiple AI backends. The application is structured as a '
    'single Python module (app.py) with supporting scripts for data ingestion and model management.')

add_heading(doc, '2.1 Technology Stack', 2, '0D9488')
tech = [
    ('Backend Framework',   'Flask 2.3.3 (Python)'),
    ('Database',            'SQLite via sqlite3 — stores users, courses, papers, questions, labels'),
    ('Vector Database',     'FAISS (faiss-cpu) — 109,635 vectors, all-MiniLM-L6-v2 embeddings'),
    ('Primary LLM',         'Groq API — llama-3.3-70b-versatile (free tier, fast inference)'),
    ('Fallback LLM',        'Google Gemini API — gemini-2.0-flash'),
    ('Embeddings',          'sentence-transformers/all-MiniLM-L6-v2 (384-dim)'),
    ('ML Classifier',       'scikit-learn — TF-IDF + Logistic Regression for difficulty prediction'),
    ('PDF Processing',      'PyPDF2 3.0.1 — text extraction from uploaded PDFs'),
    ('PDF Generation',      'ReportLab — generates downloadable exam paper PDFs'),
    ('Frontend',            'Jinja2 templates + vanilla JS + Font Awesome 6 + Inter font'),
    ('Authentication',      'Werkzeug password hashing + Flask sessions + email verification'),
]
for label, val in tech:
    add_info_box(doc, label, val)

add_heading(doc, '2.2 Project File Structure', 2, '0D9488')
files = [
    ('app.py',                  'Main Flask application — all routes, business logic, AI calls (~1100 lines)'),
    ('config.py',               'API keys, SMTP config, Flask secret key'),
    ('rag_retriever.py',        'FAISS vector index loader and semantic retrieval engine'),
    ('build_vector_db.py',      'One-time script to build the FAISS index from textbook PDFs'),
    ('ingest_past_papers.py',   'Script to ingest past papers into the SQLite database'),
    ('ingest_textbooks.py',     'Script to extract and store textbook chapters'),
    ('llama_inference.py',      'TinyLlama local inference wrapper (legacy)'),
    ('similarity_semantic.py',  'Semantic similarity matcher using sentence embeddings'),
    ('finetune.py',             'Fine-tuning script for custom model training'),
    ('export_db.py',            'Database export utility'),
    ('install_dependencies.py', 'Dependency installer helper'),
    ('intents.json',            'Intent definitions for chatbot module'),
    ('QA_dataset.csv',          'Question-Answer dataset for training'),
    ('templates/',              '13 Jinja2 HTML templates for all pages'),
    ('textbooks/',              '26 course textbook PDFs organized by subject code'),
    ('uploads/',                'User-uploaded files — past papers, similarity checks, solutions'),
    ('tinyllama_model (1)/',    'Local TinyLlama model weights (safetensors format)'),
]
for fname, desc in files:
    p = doc.add_paragraph()
    r1 = p.add_run(f'{fname:<30}')
    r1.font.name = 'Courier New'
    r1.font.size = Pt(9.5)
    r1.font.color.rgb = RGBColor.from_string('0D9488')
    r2 = p.add_run(desc)
    r2.font.size = Pt(10)
    p.paragraph_format.space_after = Pt(3)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATABASE SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, '3. Database Schema', 1, '14B8A6')
add_hr(doc)
add_body(doc,
    'The system uses a single SQLite database (users.db) with 18 tables. All tables are created '
    'automatically on first run via the init_db() function in app.py.')

tables = [
    ('users',                    'id, username, email, password (hashed), role, is_verified, verification_token, created_at'),
    ('courses',                  'id, code, name'),
    ('exam_types',               'id, name  [sessional1 | sessional2 | final]'),
    ('past_papers',              'id, course_id, exam_type_id, title, original_filename, storage_path, mime_type, size_bytes, year, created_at'),
    ('similarity_checks',        'id, teacher_id, course_id, exam_type_id, uploaded_file_path, original_filename, overall_similarity_percentage, created_at'),
    ('similarity_results',       'id, check_id, compared_paper_id, similarity_percentage, similar_questions (JSON), analysis_details, created_at'),
    ('question_difficulty_labels','id, course_id, exam_type_id, question_text, difficulty_label, created_by, created_at'),
    ('textbooks',                'id, course_id, title, file_path, total_pages, uploaded_at'),
    ('chapters',                 'id, textbook_id, chapter_number, chapter_title, start_page, end_page, content_text, extracted_at'),
    ('topics',                   'id, chapter_id, topic_name, content_text'),
    ('text_chunks',              'id, chapter_id, chunk_text, chunk_index, start_char, end_char, metadata (JSON), created_at'),
    ('chunk_embeddings',         'id, chunk_id, embedding_model, embedding_vector (BLOB), embedding_dimension, created_at'),
    ('vector_indexes',           'id, textbook_id, index_type, index_path, embedding_model, dimension, total_chunks, created_at'),
    ('generated_papers',         'id, course_id, exam_type_id, teacher_id, paper_title, generation_mode, selected_chapters (JSON), total_questions, difficulty_distribution (JSON), generated_at'),
    ('generated_questions',      'id, paper_id, question_number, question_text, question_type, difficulty_level, marks, answer_hint, mcq_options (JSON), correct_answer, source_chapter_id'),
    ('mock_papers',              'id, course_id, exam_type_id, student_id, paper_title, generation_mode, selected_chapters (JSON), total_questions, difficulty_distribution (JSON), generated_at'),
    ('mock_questions',           'id, mock_paper_id, question_number, question_text, question_type, difficulty_level, marks, answer_hint, mcq_options (JSON), correct_answer, source_chapter_id'),
    ('used_questions',           'id, question_text_hash (MD5), question_type, course_id, used_in_type, created_at'),
]

for tname, cols in tables:
    p = doc.add_paragraph()
    r1 = p.add_run(f'  {tname}')
    r1.bold = True
    r1.font.size = Pt(10.5)
    r1.font.color.rgb = RGBColor.from_string('14B8A6')
    p.paragraph_format.space_after = Pt(1)
    p2 = doc.add_paragraph()
    r2 = p2.add_run(f'    {cols}')
    r2.font.size = Pt(9.5)
    r2.font.name = 'Courier New'
    r2.font.color.rgb = RGBColor.from_string('64748B')
    p2.paragraph_format.space_after = Pt(5)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════════════════
# 4. USER ROLES & AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, '4. User Roles & Authentication', 1, '14B8A6')
add_hr(doc)

add_heading(doc, '4.1 Roles', 2, '0D9488')
add_body(doc, 'The system supports two roles:')
add_bullet(doc, 'Student — Can access past papers, generate mock exams, use the AI solution explainer, and use Exam Bee.')
add_bullet(doc, 'Teacher — Can upload past papers, generate AI exam papers, check similarity, and analyze difficulty.')

add_heading(doc, '4.2 Authentication Flow', 2, '0D9488')
steps = [
    'User registers via /signup with username, email, password, and role.',
    'Password is hashed using Werkzeug\'s generate_password_hash (PBKDF2-SHA256).',
    'Teachers must use a pre-authorized @nu.edu.pk email address.',
    'Teachers receive an email verification link via Gmail SMTP (configured in config.py).',
    'Students are verified immediately without email confirmation.',
    'Login at /login validates credentials and role, then creates a Flask session.',
    'All protected routes check session[\'user_id\'] and session[\'role\'] before serving content.',
]
for i, step in enumerate(steps, 1):
    add_bullet(doc, f'Step {i}: {step}')

add_heading(doc, '4.3 Predefined Teacher Emails', 2, '0D9488')
add_body(doc, 'The following emails are pre-authorized as teacher accounts:')
add_bullet(doc, 'i221935@nu.edu.pk')
add_bullet(doc, 'i221927@nu.edu.pk')
add_body(doc, 'Additional teachers can be added by modifying the PREDEFINED_TEACHER_EMAILS list in app.py.', italic=True)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MODULES — STUDENT SIDE
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, '5. Student Modules', 1, '14B8A6')
add_hr(doc)

# ── 5.1 Past Papers ──────────────────────────────────────────────────────────
add_heading(doc, '5.1 Past Papers Library', 2, '0D9488')
add_body(doc,
    'Students can browse and download past exam papers organized by course code and exam type '
    '(Sessional 1, Sessional 2, Final). Papers are stored as PDFs in the uploads/past_papers/ '
    'directory and served via the /files/<paper_id> route.')
add_bullet(doc, 'Route: GET /past-papers?course_id=&exam_type_id=')
add_bullet(doc, 'Filter by course and exam type using dropdown selectors.')
add_bullet(doc, 'Papers are downloaded directly as PDF attachments.')
add_bullet(doc, '26 courses are supported, covering CS, AI, Math, and Engineering subjects.')

# ── 5.2 Mock Exam Generator ──────────────────────────────────────────────────
add_heading(doc, '5.2 Mock Exam Generator', 2, '0D9488')
add_body(doc,
    'Students select a course, textbook, and chapters, then specify how many MCQ, short answer, '
    'and long answer questions they want. The system generates a complete mock exam using AI '
    'and the textbook content as context.')

add_body(doc, 'Generation Pipeline:', bold=True)
flow = [
    'Student submits course, textbook, chapters, and question distribution via POST /api/generate-mock-paper.',
    'Chapter content is fetched from the chapters table in SQLite.',
    'A structured prompt is built with up to 6000 characters of chapter text.',
    'Groq API (llama-3.3-70b-versatile) generates all questions in one batch call.',
    'If Groq fails, Gemini API (gemini-2.0-flash) is used as fallback.',
    'If both fail, OpenAI GPT-4 is tried as a second fallback.',
    'As a last resort, template-based questions are generated from sentence extraction.',
    'Questions are distributed into type buckets (MCQ, short, long) and numbered.',
    'The paper is saved to mock_papers and mock_questions tables.',
    'Student can download the paper as a PDF or attempt it interactively.',
]
for i, step in enumerate(flow, 1):
    add_bullet(doc, f'{i}. {step}')

add_body(doc, 'Mock Exam Attempt:', bold=True)
add_body(doc,
    'Students can attempt the generated mock exam at /attempt-mock-exam/<paper_id>. '
    'The page features a countdown timer (2 minutes per question), a progress bar, '
    'MCQ radio buttons, and text areas for written answers. On submission, MCQ answers '
    'are auto-graded and a score breakdown is shown in a results overlay.')

# ── 5.3 AI Solution Explainer ────────────────────────────────────────────────
add_heading(doc, '5.3 AI Solution Explainer', 2, '0D9488')
add_body(doc,
    'Students upload any exam PDF and select their course. The system extracts questions '
    'from the PDF, retrieves relevant textbook passages using FAISS semantic search, and '
    'generates complete solutions with step-by-step explanations.')

add_body(doc, 'RAG Pipeline:', bold=True)
rag_steps = [
    'PDF uploaded to uploads/solution_explainer/ and text extracted via PyPDF2.',
    'Questions parsed using multi-pattern regex (numbered lists, Q1., Question 1:, etc.).',
    'FAISS index loaded from vector_db/faiss_index.bin (109,635 vectors).',
    'Each question is embedded into a 384-dim vector using all-MiniLM-L6-v2.',
    'Top-4 most semantically similar textbook chunks retrieved per question.',
    'Retrieved context (max 2000 chars) injected into the solution prompt.',
    'Groq API generates solution; Gemini used as fallback.',
    'Results returned as JSON with question, solution, and explanation fields.',
    'Temporary PDF file deleted after processing.',
]
for i, step in enumerate(rag_steps, 1):
    add_bullet(doc, f'{i}. {step}')

add_body(doc, 'Two modes are available:', bold=True)
add_bullet(doc, 'Solution Only — returns just the answer.')
add_bullet(doc, 'Solution + Explanation — returns answer with step-by-step reasoning referencing textbook concepts.')

# ── 5.4 Exam Bee ─────────────────────────────────────────────────────────────
add_heading(doc, '5.4 Exam Bee — Bloom\'s MCQ Generator', 2, '0D9488')
add_body(doc,
    'Exam Bee generates Bloom\'s taxonomy-aligned MCQs from selected course chapters. '
    'Students choose a cognitive level (Remember, Understand, Apply, Analyze) and a '
    'difficulty level (Easy, Medium, Hard), then specify how many questions to generate.')

add_body(doc, 'Bloom\'s Level Mapping:', bold=True)
bloom = [
    ('Remember',    'Factual recall — definitions, names, dates, basic facts.'),
    ('Understand',  'Concept explanation — summarize, interpret, explain ideas.'),
    ('Apply',       'Problem-solving — use formulas, procedures in new situations.'),
    ('Analyze',     'Reasoning — break down, compare, contrast, infer from content.'),
]
for level, desc in bloom:
    p = doc.add_paragraph()
    r1 = p.add_run(f'  {level}: ')
    r1.bold = True
    r1.font.size = Pt(10.5)
    r1.font.color.rgb = RGBColor.from_string('14B8A6')
    r2 = p.add_run(desc)
    r2.font.size = Pt(10.5)
    p.paragraph_format.space_after = Pt(3)

add_body(doc,
    'Generation uses Groq as primary engine and Gemini as fallback. Each MCQ includes '
    '4 options (A/B/C/D), the correct answer label, and an explanation. Results can be '
    'downloaded as a PDF.')

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════════════════
# 6. MODULES — TEACHER SIDE
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, '6. Teacher Modules', 1, '14B8A6')
add_hr(doc)

# ── 6.1 AI Paper Generator ───────────────────────────────────────────────────
add_heading(doc, '6.1 AI Paper Generator', 2, '0D9488')
add_body(doc,
    'Teachers can generate complete exam papers with a specified distribution of MCQ, '
    'short answer, and long answer questions. The system uses textbook content as the '
    'knowledge base and ensures no repeated questions across papers.')

add_body(doc, 'Configuration Options:', bold=True)
add_bullet(doc, 'Course and exam type selection.')
add_bullet(doc, 'Generation mode: Chapters (select specific chapters) or Topics (enter topic keywords).')
add_bullet(doc, 'Question type distribution: MCQ count, Short Answer count, Long Answer count.')
add_bullet(doc, 'Maximum 50 questions per paper.')

add_body(doc, 'Generation Flow:', bold=True)
gen_flow = [
    'Teacher selects course → system loads textbooks for that course via /api/textbooks/<course_id>.',
    'Chapters are extracted from the textbook PDF using detect_chapters_in_pdf() with multi-pattern detection.',
    'Teacher selects chapters and question counts, then submits via POST /api/generate-paper.',
    'Chapter content fetched from SQLite chapters table.',
    'Groq API generates all questions in one batch (fallback: Gemini → OpenAI → template).',
    'Questions saved to generated_papers and generated_questions tables.',
    'Teacher can view the paper and download it as a formatted PDF via /api/export-paper/<paper_id>.',
]
for i, step in enumerate(gen_flow, 1):
    add_bullet(doc, f'{i}. {step}')

# ── 6.2 Similarity Checker ───────────────────────────────────────────────────
add_heading(doc, '6.2 Similarity Checker', 2, '0D9488')
add_body(doc,
    'Teachers upload a new exam paper PDF and select the course and exam type. The system '
    'compares it against all existing past papers of the same course and exam type, '
    'identifying similar questions and calculating an overall similarity percentage.')

add_body(doc, 'Similarity Algorithm:', bold=True)
sim_steps = [
    'Text extracted from uploaded PDF using PyPDF2.',
    'Questions parsed using multi-pattern regex (numbered lists, Q1., etc.).',
    'For each existing past paper: text extracted and questions parsed.',
    'Same-paper detection: if 80%+ of questions are 90%+ similar → 100% similarity reported.',
    'For different papers: semantic matcher (sentence embeddings) used if available.',
    'Fallback: lexical similarity using SequenceMatcher + Jaccard word overlap + concept extraction.',
    'Concept extraction identifies specific algorithms, data structures, DB concepts, math topics.',
    'Overall similarity = average of per-paper similarities.',
    'Results saved to similarity_checks and similarity_results tables.',
    'Detailed report shows matched questions, similarity percentages, and matched concepts.',
]
for i, step in enumerate(sim_steps, 1):
    add_bullet(doc, f'{i}. {step}')

add_body(doc, 'Similarity Score Interpretation:', bold=True)
add_bullet(doc, '0–30%: Low similarity — paper is largely original.')
add_bullet(doc, '30–70%: Moderate similarity — some questions overlap, review recommended.')
add_bullet(doc, '70–100%: High similarity — significant overlap, revision strongly advised.')

# ── 6.3 Difficulty Analyzer ──────────────────────────────────────────────────
add_heading(doc, '6.3 Difficulty Analyzer', 2, '0D9488')
add_body(doc,
    'The difficulty analyzer predicts whether exam questions are Easy, Medium, or Hard. '
    'It uses a three-tier approach: trained ML model (when sufficient labeled data exists), '
    'zero-shot AI classification (BART MNLI), or heuristic analysis.')

add_body(doc, 'Training Data Collection:', bold=True)
add_bullet(doc, 'Teachers label questions manually via the labeling form.')
add_bullet(doc, 'Questions can be auto-fetched from uploaded past papers via /api/fetch-questions-from-papers.')
add_bullet(doc, 'Minimum 9 labeled samples required (mixed easy/medium/hard) to train the ML model.')
add_bullet(doc, 'Labels stored in question_difficulty_labels table.')

add_body(doc, 'Prediction Pipeline:', bold=True)
diff_steps = [
    'Teacher uploads a new exam paper PDF.',
    'Questions extracted and cleaned.',
    'If ≥9 labeled samples exist for the course: TF-IDF vectorizer + Logistic Regression classifier used.',
    'Extra features added: word count, lexical diversity, avg word length, Bloom\'s cognitive scores.',
    'If insufficient labels: zero-shot BART MNLI classifier attempted.',
    'Final fallback: heuristic rules based on word count and Bloom\'s keyword density.',
    'Results show each question with predicted label (Easy/Medium/Hard) and confidence score.',
    'Summary counts and dominant difficulty level displayed.',
]
for i, step in enumerate(diff_steps, 1):
    add_bullet(doc, f'{i}. {step}')

add_body(doc, 'Bloom\'s Cognitive Keywords Used in Heuristics:', bold=True)
add_bullet(doc, 'Recall: define, list, state, name, identify, what, when')
add_bullet(doc, 'Understand: explain, describe, summarize, compare, differentiate, why')
add_bullet(doc, 'Apply: solve, calculate, design, implement, derive, prove, analyze, evaluate')

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════════════════
# 7. AI & RAG ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, '7. AI & RAG Architecture', 1, '14B8A6')
add_hr(doc)

add_heading(doc, '7.1 LLM Backend Priority Chain', 2, '0D9488')
add_body(doc,
    'All AI generation tasks follow a priority chain to maximize availability and quality:')
chain = [
    ('1st — Groq API',    'llama-3.3-70b-versatile', 'Free tier, fast (< 3s), high quality. Primary for all generation.'),
    ('2nd — Gemini API',  'gemini-2.0-flash',         'Google\'s fast model. Used when Groq quota is exceeded.'),
    ('3rd — OpenAI API',  'gpt-4-turbo-preview',      'Optional fallback. Requires paid API key.'),
    ('4th — Templates',   'Rule-based',               'Last resort. Generates basic questions from sentence extraction.'),
]
for priority, model, desc in chain:
    p = doc.add_paragraph()
    r1 = p.add_run(f'  {priority} ({model}): ')
    r1.bold = True
    r1.font.size = Pt(10.5)
    r1.font.color.rgb = RGBColor.from_string('14B8A6')
    r2 = p.add_run(desc)
    r2.font.size = Pt(10.5)
    p.paragraph_format.space_after = Pt(4)

add_heading(doc, '7.2 FAISS Vector Database', 2, '0D9488')
add_body(doc,
    'The FAISS vector database is the core of the RAG system. It enables semantic search '
    'across all course textbooks to retrieve relevant context for question solving.')

add_info_box(doc, 'Index File',       'vector_db/faiss_index.bin')
add_info_box(doc, 'Metadata File',    'vector_db/metadata.pkl')
add_info_box(doc, 'Total Vectors',    '109,635')
add_info_box(doc, 'Vector Dimension', '384 (all-MiniLM-L6-v2)')
add_info_box(doc, 'Index Type',       'IndexFlatIP (inner product / cosine similarity)')
add_info_box(doc, 'Courses Covered',  '26 subjects')

add_body(doc, 'How the Index Was Built (build_vector_db.py):', bold=True)
build_steps = [
    'All PDFs in the textbooks/ directory are processed.',
    'Each PDF is read page-by-page using PyPDF2.',
    'Text is cleaned: whitespace normalized, noise removed, short lines filtered.',
    'Text is split into overlapping chunks (~500 words, 50-word overlap) using smart_chunk_text().',
    'Each chunk is embedded using sentence-transformers/all-MiniLM-L6-v2.',
    'Embeddings are normalized (L2) and added to the FAISS IndexFlatIP.',
    'Metadata (course code, source file, chunk index, page number) saved to metadata.pkl.',
    'Final index written to vector_db/faiss_index.bin.',
]
for i, step in enumerate(build_steps, 1):
    add_bullet(doc, f'{i}. {step}')

add_heading(doc, '7.3 Retrieval Process (rag_retriever.py)', 2, '0D9488')
add_body(doc,
    'The RAGRetriever class handles all semantic retrieval operations:')
retrieval = [
    'Query text is embedded into a 384-dim vector.',
    'L2 normalization applied to enable cosine similarity via inner product.',
    'FAISS searches top-K candidates (default K=20 before filtering).',
    'Results filtered by course code to ensure subject relevance.',
    'Top-4 chunks returned, joined with separators, capped at 2000 characters.',
    'Context injected into the LLM prompt before the question.',
]
for i, step in enumerate(retrieval, 1):
    add_bullet(doc, f'{i}. {step}')

add_heading(doc, '7.4 Chapter Detection Algorithm', 2, '0D9488')
add_body(doc,
    'The detect_chapters_in_pdf() function uses a multi-pass algorithm to extract chapter '
    'structure from textbook PDFs:')
add_bullet(doc, 'Pass 1: Scan all pages for lines matching chapter heading patterns (8 regex patterns).')
add_bullet(doc, 'Patterns include: "Chapter N: Title", "CHAPTER N TITLE", "N. Title", "Unit N:", "Module N:", etc.')
add_bullet(doc, 'Pass 2: Deduplicate candidates and validate chapter number sequence (no wild jumps > 5).')
add_bullet(doc, 'Pass 3: Extract content between chapter boundaries.')
add_bullet(doc, 'Fallback: If no chapters detected, split textbook into ~10 equal page groups.')

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════════════════
# 8. API ROUTES REFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, '8. API Routes Reference', 1, '14B8A6')
add_hr(doc)

routes = [
    # Auth
    ('GET/POST', '/login',                          'Login page and form submission'),
    ('GET/POST', '/signup',                         'Registration page and form submission'),
    ('GET',      '/verify/<token>',                 'Email verification for teacher accounts'),
    ('GET',      '/logout',                         'Clear session and redirect to login'),
    # Dashboards
    ('GET',      '/student-dashboard',              'Student dashboard (requires student role)'),
    ('GET',      '/teacher-dashboard',              'Teacher dashboard (requires teacher role)'),
    # Past Papers
    ('GET',      '/past-papers',                    'Browse past papers with course/type filter'),
    ('GET',      '/files/<paper_id>',               'Download a past paper PDF'),
    # Similarity
    ('GET',      '/similarity-checker',             'Similarity checker form (teacher only)'),
    ('POST',     '/check-similarity',               'Process similarity check, return JSON results'),
    ('GET',      '/similarity-results/<check_id>',  'View detailed similarity results page'),
    # Difficulty
    ('GET',      '/difficulty-checker',             'Difficulty analyzer page (teacher only)'),
    ('POST',     '/difficulty-labels',              'Add a labeled training sample'),
    ('POST',     '/api/fetch-questions-from-papers','Auto-fetch questions from past papers for labeling'),
    ('POST',     '/api/analyze-difficulty',         'Run difficulty prediction on uploaded paper'),
    # AI Paper Generator
    ('GET',      '/ai-paper-generator',             'AI paper generator page (teacher only)'),
    ('POST',     '/api/generate-paper',             'Generate exam paper, return questions JSON'),
    ('GET',      '/api/export-paper/<paper_id>',    'Download generated paper as PDF'),
    ('GET',      '/api/generated-papers',           'List teacher\'s generated papers'),
    ('GET',      '/api/paper/<paper_id>',           'Get questions for a specific paper'),
    # Textbooks & Chapters
    ('POST',     '/api/upload-textbook',            'Upload a textbook PDF for a course'),
    ('GET',      '/api/textbooks',                  'List textbooks (optional ?course_id filter)'),
    ('GET',      '/api/textbooks/<course_id>',      'List textbooks for a specific course'),
    ('GET',      '/api/chapters/<textbook_id>',     'Get chapters for a textbook'),
    ('POST',     '/api/chapters/<textbook_id>/refresh', 'Re-extract chapters from textbook PDF'),
    # Mock Exam
    ('GET',      '/mock-exam-generator',            'Mock exam generator page (student only)'),
    ('POST',     '/api/generate-mock-paper',        'Generate mock exam, return questions JSON'),
    ('GET',      '/api/get-mock-papers',            'List student\'s generated mock papers'),
    ('GET',      '/api/download-mock-paper/<id>',   'Download mock paper as PDF'),
    ('GET',      '/attempt-mock-exam/<paper_id>',   'Interactive exam attempt page'),
    ('POST',     '/api/submit-mock-exam',           'Submit answers, return score and breakdown'),
    # Solution Explainer
    ('GET',      '/solution-explainer',             'Solution explainer page (student only)'),
    ('POST',     '/api/solve-paper',                'Process uploaded exam PDF, return solutions JSON'),
    # Exam Bee
    ('GET',      '/exam-bee',                       'Exam Bee MCQ generator page (student only)'),
    ('POST',     '/api/exam-bee/generate',          'Generate Bloom-aligned MCQs, return JSON'),
    ('GET',      '/api/exam-bee/download/<id>',     'Download Exam Bee paper as PDF'),
    # Utility
    ('GET',      '/api/courses',                    'List all courses'),
    ('GET',      '/api/exam-types',                 'List all exam types'),
]

for method, route, desc in routes:
    p = doc.add_paragraph()
    r1 = p.add_run(f'  {method:<8}')
    r1.font.name = 'Courier New'
    r1.font.size = Pt(9)
    r1.font.color.rgb = RGBColor.from_string('F59E0B')
    r2 = p.add_run(f'{route:<45}')
    r2.font.name = 'Courier New'
    r2.font.size = Pt(9)
    r2.font.color.rgb = RGBColor.from_string('14B8A6')
    r3 = p.add_run(desc)
    r3.font.size = Pt(9.5)
    r3.font.color.rgb = RGBColor.from_string('94A3B8')
    p.paragraph_format.space_after = Pt(2)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════════════════
# 9. COURSES SUPPORTED
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, '9. Supported Courses', 1, '14B8A6')
add_hr(doc)
add_body(doc, 'The system supports 26 courses with textbooks and past papers:')

courses = [
    ('AI',        'Artificial Intelligence',                    'Peter Norvig — AI: A Modern Approach'),
    ('ALGO',      'Design & Analysis of Algorithms',          
# ═══════════════════════════════════════════════════════════════════════════════
# 9. CONFIGURATION & DEPLOYMENT
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, '9. Configuration & Deployment', 1, '14B8A6')
add_hr(doc)

add_heading(doc, '9.1 config.py Settings', 2, '0D9488')
config_items = [
    ('SMTP_SERVER / SMTP_PORT',  'Gmail SMTP — smtp.gmail.com:587 for email verification'),
    ('SMTP_USERNAME',            'Gmail address used to send verification emails'),
    ('SMTP_PASSWORD',            'Gmail App Password (16-char, not account password)'),
    ('SECRET_KEY',               'Flask session secret — 64-char hex string'),
    ('GROQ_API_KEY',             'Groq API key — get free at console.groq.com/keys'),
    ('GROQ_MODEL',               'llama-3.3-70b-versatile (default)'),
    ('GEMINI_API_KEY',           'Google Gemini API key — get at aistudio.google.com'),
    ('GEMINI_MODEL',             'gemini-2.0-flash (default)'),
    ('OPENAI_API_KEY',           'Optional OpenAI key for GPT-4 fallback'),
    ('APP_URL',                  'Base URL for email verification links (default: http://localhost:5000)'),
]
for label, val in config_items:
    add_info_box(doc, label, val)

add_heading(doc, '9.2 Running the Application', 2, '0D9488')
add_body(doc, 'Prerequisites:', bold=True)
add_bullet(doc, 'Python 3.9+ installed')
add_bullet(doc, 'pip install -r requirements.txt')
add_bullet(doc, 'Valid Groq API key in config.py')
add_bullet(doc, 'FAISS index built (run build_vector_db.py once)')

add_body(doc, 'Start the server:', bold=True)
p = doc.add_paragraph()
r = p.add_run('    python app.py')
r.font.name = 'Courier New'
r.font.size = Pt(11)
r.font.color.rgb = RGBColor.from_string('14B8A6')
shade_paragraph(p, 'F0FDFA')
p.paragraph_format.space_after = Pt(8)

add_body(doc, 'The application starts on http://localhost:5000 with debug=True and use_reloader=False '
              '(reloader disabled to prevent DB lock issues during long AI generation requests).', italic=True)

add_heading(doc, '9.3 First-Time Setup', 2, '0D9488')
setup_steps = [
    'Run python app.py — this auto-creates users.db with all 18 tables.',
    'Run python ingest_past_papers.py to populate the past papers library.',
    'Run python build_vector_db.py to build the FAISS index from textbooks/ PDFs.',
    'Register a teacher account using a pre-authorized @nu.edu.pk email.',
    'Verify the teacher email via the link sent to your inbox.',
    'Upload textbooks via the AI Paper Generator → Upload Textbook.',
    'Label at least 9 questions per course for the difficulty analyzer ML model.',
]
for i, step in enumerate(setup_steps, 1):
    add_bullet(doc, f'{i}. {step}')

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════════════════
# 10. SUPPORTED COURSES
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, '10. Supported Courses & Textbooks', 1, '14B8A6')
add_hr(doc)
add_body(doc, 'The following 26 courses are supported with dedicated textbooks in the textbooks/ directory:')

courses = [
    ('AI',        'Artificial Intelligence',          'Artificial Intelligence — Peter Norvig'),
    ('ALGO',      'Design & Analysis of Algorithms',  'Introduction to the Design and Analysis of Algorithms — Levitin'),
    ('calculus',  'Calculus',                         'Thomas Calculus'),
    ('CNET',      'Computer Networks',                'Computer Networking: A Top-Down Approach — Kurose & Ross'),
    ('coal',      'Computer Organization & Architecture', 'Computer Organization & Architecture — Stallings'),
    ('CYBER_SEC', 'Cyber Security',                   'Computer Security: Principles — Stallings'),
    ('DATA MINING','Data Mining',                     'Data Mining: Concepts and Techniques — Han et al.'),
    ('discrete',  'Discrete Mathematics',             'Discrete Mathematics — Susanna Epp'),
    ('dld',       'Digital Logic Design',             'Digital Design — M. Morris Mano'),
    ('DSA',       'Data Structures & Algorithms',     'Data Structures Using C++ — D.S. Malik'),
    ('DWH',       'Data Warehousing',                 'The Data Warehouse Toolkit — Ralph Kimball'),
    ('english',   'English',                          'ENGLISH.pdf'),
    ('FSE',       'Fundamentals of Software Engineering', 'Software Engineering: A Practitioner\'s Approach — Pressman'),
    ('IDS',       'Introduction to Data Science',     'Python for Everybody — Severance'),
    ('INFO SEC',  'Information Security',             'Information Security: Principles and Practice — Mark Stamp'),
    ('ISE',       'Information Systems Engineering',  'ISE.pdf'),
    ('LA',        'Linear Algebra',                   'Linear Algebra and Its Applications — David Lay'),
    ('OOP',       'Object-Oriented Programming',      'Starting Out with C++ — Tony Gaddis'),
    ('OS',        'Operating Systems',                'Operating System Concepts — Silberschatz'),
    ('PF',        'Programming Fundamentals',         'Book-1-CPP PF.pdf'),
    ('PPIT',      'Professional Practices in IT',     'Ethics for the Information Age — Michael Quinn'),
    ('SDA',       'Software Design & Architecture',   'SDA.pdf'),
    ('SPM',       'Software Project Management',      'SPM.pdf'),
    ('SQE',       'Software Quality Engineering',     'SQE.pdf'),
    ('SREQ',      'Software Requirements Engineering','sreq.pdf'),
]

for code, name, textbook in courses:
    p = doc.add_paragraph()
    r1 = p.add_run(f'  {code:<12}')
    r1.font.name = 'Courier New'
    r1.font.size = Pt(10)
    r1.font.color.rgb = RGBColor.from_string('14B8A6')
    r1.bold = True
    r2 = p.add_run(f'{name:<40}')
    r2.font.size = Pt(10)
    r2.font.color.rgb = RGBColor.from_string('E2E8F0')
    r3 = p.add_run(textbook)
    r3.font.size = Pt(9.5)
    r3.font.color.rgb = RGBColor.from_string('64748B')
    p.paragraph_format.space_after = Pt(3)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════
output_path = 'AI_Exam_Management_System_Documentation.docx'
doc.save(output_path)
print(f'Document saved: {output_path}')
