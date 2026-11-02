"""
Read _gen_doc.py, append the missing sections + doc.save(), then run it.
"""
import subprocess, sys, os

extra = '''

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

add_body(doc, 'The application starts on http://localhost:5000 with debug=True and use_reloader=False.', italic=True)

add_heading(doc, '9.3 First-Time Setup', 2, '0D9488')
for i, step in enumerate([
    'Run python app.py — auto-creates users.db with all 18 tables.',
    'Run python ingest_past_papers.py to populate the past papers library.',
    'Run python build_vector_db.py to build the FAISS index from textbooks/ PDFs.',
    'Register a teacher account using a pre-authorized @nu.edu.pk email.',
    'Verify the teacher email via the link sent to your inbox.',
    'Upload textbooks via the AI Paper Generator → Upload Textbook.',
    'Label at least 9 questions per course for the difficulty analyzer ML model.',
], 1):
    add_bullet(doc, f'{i}. {step}')

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════════════════
# 10. SUPPORTED COURSES
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, '10. Supported Courses & Textbooks', 1, '14B8A6')
add_hr(doc)
add_body(doc, 'The following 26 courses are supported with dedicated textbooks:')

courses = [
    ('AI',         'Artificial Intelligence',              'Artificial Intelligence — Peter Norvig'),
    ('ALGO',       'Design & Analysis of Algorithms',      'Introduction to the Design and Analysis of Algorithms'),
    ('calculus',   'Calculus',                             'Thomas Calculus'),
    ('CNET',       'Computer Networks',                    'Computer Networking: A Top-Down Approach — Kurose & Ross'),
    ('coal',       'Computer Organization & Architecture', 'Computer Organization & Architecture — Stallings'),
    ('CYBER_SEC',  'Cyber Security',                       'Computer Security: Principles — Stallings'),
    ('DATA MINING','Data Mining',                          'Data Mining: Concepts and Techniques — Han et al.'),
    ('discrete',   'Discrete Mathematics',                 'Discrete Mathematics — Susanna Epp'),
    ('dld',        'Digital Logic Design',                 'Digital Design — M. Morris Mano'),
    ('DSA',        'Data Structures & Algorithms',         'Data Structures Using C++ — D.S. Malik'),
    ('DWH',        'Data Warehousing',                     'The Data Warehouse Toolkit — Ralph Kimball'),
    ('english',    'English',                              'ENGLISH.pdf'),
    ('FSE',        'Fundamentals of Software Engineering', 'Software Engineering: A Practitioner Approach'),
    ('IDS',        'Introduction to Data Science',         'Python for Everybody — Severance'),
    ('INFO SEC',   'Information Security',                 'Information Security: Principles and Practice — Mark Stamp'),
    ('ISE',        'Information Systems Engineering',      'ISE.pdf'),
    ('LA',         'Linear Algebra',                       'Linear Algebra and Its Applications — David Lay'),
    ('OOP',        'Object-Oriented Programming',          'Starting Out with C++ — Tony Gaddis'),
    ('OS',         'Operating Systems',                    'Operating System Concepts — Silberschatz'),
    ('PF',         'Programming Fundamentals',             'Book-1-CPP PF.pdf'),
    ('PPIT',       'Professional Practices in IT',         'Ethics for the Information Age — Michael Quinn'),
    ('SDA',        'Software Design & Architecture',       'SDA.pdf'),
    ('SPM',        'Software Project Management',          'SPM.pdf'),
    ('SQE',        'Software Quality Engineering',         'SQE.pdf'),
    ('SREQ',       'Software Requirements Engineering',    'sreq.pdf'),
]
for code, name, textbook in courses:
    p = doc.add_paragraph()
    r1 = p.add_run(f'  {code:<14}')
    r1.font.name = 'Courier New'
    r1.font.size = Pt(10)
    r1.font.color.rgb = RGBColor.from_string('14B8A6')
    r1.bold = True
    r2 = p.add_run(f'{name:<42}')
    r2.font.size = Pt(10)
    r3 = p.add_run(textbook)
    r3.font.size = Pt(9.5)
    r3.font.color.rgb = RGBColor.from_string('64748B')
    p.paragraph_format.space_after = Pt(3)

# ── SAVE ──────────────────────────────────────────────────────────────────────
output_path = 'AI_Exam_Management_System_Documentation.docx'
doc.save(output_path)
import os as _os
print(f'[OK] Saved: {output_path}  ({_os.path.getsize(output_path):,} bytes)')
'''

# Write a combined script
combined = open('_gen_doc.py', encoding='utf-8').read() + extra
tmp = '_combined_doc.py'
open(tmp, 'w', encoding='utf-8').write(combined)

# Run it
result = subprocess.run([sys.executable, tmp], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print('STDERR:', result.stderr[:2000])

# Cleanup
os.remove(tmp)
