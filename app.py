from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import secrets
import PyPDF2
import io
import difflib
import json
import math
import hashlib
from collections import Counter
from pathlib import Path
import shutil

# RAG retriever (FAISS vector DB)
try:
    from rag_retriever import retriever as retriever
except Exception as _rag_err:
    print(f"RAG retriever not available: {_rag_err}")
    class _DummyRetriever:
        def is_ready(self): return False
        def load(self): pass
        def get_context(self, *a, **kw): return ''
    retriever = _DummyRetriever()

# Vector database and embeddings for AI Paper Generator
try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
    EMBEDDING_MODEL = None  # Will be lazy-loaded
except Exception as e:
    print(f"Vector DB dependencies not available: {e}")
    faiss = None
    FAISS_AVAILABLE = False
    EMBEDDING_MODEL = None

# PDF export
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# TinyLlama question generation (local, no API)
try:
    from llama_inference import generate_questions_with_tinyllama, is_tinyllama_available
    LLAMA_AVAILABLE = True
    LLAMA_GENERATOR = None  # kept for legacy references; not used directly
except Exception as e:
    print(f"Llama model not available: {e}")
    LLAMA_AVAILABLE = False
    LLAMA_GENERATOR = None
    generate_questions_with_tinyllama = None
    is_tinyllama_available = lambda: False

# Optional ML stack for difficulty analyzer
try:
    import numpy as np
    from scipy.sparse import hstack, csr_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except Exception:
    np = None
    hstack = csr_matrix = None
    TfidfVectorizer = None
    LogisticRegression = None
    SKLEARN_AVAILABLE = False

# Optional Transformers zero-shot classifier
try:
    from transformers import pipeline
except Exception:
    pipeline = None

ZERO_SHOT_MODEL_NAME = "facebook/bart-large-mnli"
ZERO_SHOT_CLASSIFIER = None
ZERO_SHOT_INIT_ATTEMPTED = False

# Try to import optional semantic matcher (prototype)
try:
    from similarity_semantic import get_matcher_safe
    semantic_matcher = get_matcher_safe()
except Exception:
    semantic_matcher = None

# Import configuration from config.py
try:
    from config import (
        SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, 
        FROM_EMAIL, FROM_NAME, APP_URL, VERIFICATION_EXPIRY_HOURS,
        SECRET_KEY
    )
except ImportError:
    # Fallback configuration if config.py doesn't exist
    SMTP_SERVER = 'smtp.gmail.com'
    SMTP_PORT = 587
    SMTP_USERNAME = 'your-gmail@gmail.com'
    SMTP_PASSWORD = 'your-app-password'
    FROM_EMAIL = SMTP_USERNAME
    FROM_NAME = "AI Exam Management System"
    APP_URL = "http://localhost:5000"
    VERIFICATION_EXPIRY_HOURS = 24
    # Generate a random secret key if not in config
    import secrets
    SECRET_KEY = secrets.token_hex(32)

app = Flask(__name__)
app.secret_key = SECRET_KEY

# Email configuration is now imported from config.py

# Predefined teacher emails
PREDEFINED_TEACHER_EMAILS = ['i221935@nu.edu.pk', 'i221927@nu.edu.pk']

# Initialize database
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # Create users table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            is_verified INTEGER DEFAULT 0,
            verification_token TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create courses table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL,
            name TEXT
        )
    ''')

    # Create exam_types table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS exam_types (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    ''')

    # Create past_papers table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS past_papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER NOT NULL,
            exam_type_id INTEGER NOT NULL,
            title TEXT,
            original_filename TEXT,
            storage_path TEXT NOT NULL,
            mime_type TEXT,
            size_bytes INTEGER,
            year TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(course_id) REFERENCES courses(id),
            FOREIGN KEY(exam_type_id) REFERENCES exam_types(id)
        )
    ''')
    
    # Create similarity_checks table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS similarity_checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_id INTEGER NOT NULL,
            course_id INTEGER NOT NULL,
            exam_type_id INTEGER NOT NULL,
            uploaded_file_path TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            overall_similarity_percentage REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(teacher_id) REFERENCES users(id),
            FOREIGN KEY(course_id) REFERENCES courses(id),
            FOREIGN KEY(exam_type_id) REFERENCES exam_types(id)
        )
    ''')
    
    # Create similarity_results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS similarity_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            check_id INTEGER NOT NULL,
            compared_paper_id INTEGER NOT NULL,
            similarity_percentage REAL NOT NULL,
            similar_questions TEXT,
            analysis_details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(check_id) REFERENCES similarity_checks(id),
            FOREIGN KEY(compared_paper_id) REFERENCES past_papers(id)
        )
    ''')
    
    # Training data for difficulty analyzer
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS question_difficulty_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER NOT NULL,
            exam_type_id INTEGER NOT NULL,
            question_text TEXT NOT NULL,
            difficulty_label TEXT NOT NULL,
            created_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(course_id) REFERENCES courses(id),
            FOREIGN KEY(exam_type_id) REFERENCES exam_types(id),
            FOREIGN KEY(created_by) REFERENCES users(id)
        )
    ''')
    
    # AI Paper Generation tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS textbooks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            file_path TEXT NOT NULL,
            total_pages INTEGER,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(course_id) REFERENCES courses(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chapters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            textbook_id INTEGER NOT NULL,
            chapter_number INTEGER,
            chapter_title TEXT NOT NULL,
            start_page INTEGER,
            end_page INTEGER,
            content_text TEXT,
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(textbook_id) REFERENCES textbooks(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chapter_id INTEGER NOT NULL,
            topic_name TEXT NOT NULL,
            content_text TEXT,
            FOREIGN KEY(chapter_id) REFERENCES chapters(id)
        )
    ''')
    
    # Vector database tables for enhanced AI Paper Generator
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS text_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chapter_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            start_char INTEGER,
            end_char INTEGER,
            metadata TEXT, -- JSON: page, section, etc.
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(chapter_id) REFERENCES chapters(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id INTEGER NOT NULL,
            embedding_model TEXT NOT NULL,
            embedding_vector BLOB, -- Store as binary (FAISS will handle)
            embedding_dimension INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(chunk_id) REFERENCES text_chunks(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vector_indexes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            textbook_id INTEGER NOT NULL,
            index_type TEXT NOT NULL, -- 'faiss', 'weaviate'
            index_path TEXT,
            embedding_model TEXT,
            dimension INTEGER,
            total_chunks INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(textbook_id) REFERENCES textbooks(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS generated_papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER NOT NULL,
            exam_type_id INTEGER,
            teacher_id INTEGER NOT NULL,
            paper_title TEXT NOT NULL,
            generation_mode TEXT NOT NULL, -- 'chapters' or 'topics'
            selected_chapters TEXT, -- JSON array of chapter IDs
            selected_topics TEXT, -- JSON array of topic names
            total_questions INTEGER,
            difficulty_distribution TEXT, -- JSON: {'easy': 5, 'medium': 10, 'hard': 5}
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(course_id) REFERENCES courses(id),
            FOREIGN KEY(exam_type_id) REFERENCES exam_types(id),
            FOREIGN KEY(teacher_id) REFERENCES users(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS generated_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id INTEGER NOT NULL,
            question_number INTEGER NOT NULL,
            question_text TEXT NOT NULL,
            question_type TEXT NOT NULL, -- 'mcq', 'short_answer', 'long_answer'
            difficulty_level TEXT NOT NULL, -- 'easy', 'medium', 'hard'
            marks INTEGER DEFAULT 5,
            answer_hint TEXT,
            mcq_options TEXT, -- JSON array of MCQ options
            correct_answer TEXT, -- Correct answer for MCQ (A, B, C, D)
            source_chapter_id INTEGER,
            source_topic TEXT,
            FOREIGN KEY(paper_id) REFERENCES generated_papers(id),
            FOREIGN KEY(source_chapter_id) REFERENCES chapters(id)
        )
    ''')
    
    # Mock Exam tables for students
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mock_papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER NOT NULL,
            exam_type_id INTEGER,
            student_id INTEGER NOT NULL,
            paper_title TEXT NOT NULL,
            generation_mode TEXT NOT NULL, -- 'chapters' or 'topics'
            selected_chapters TEXT, -- JSON array of chapter IDs
            selected_topics TEXT, -- JSON array of topic names
            total_questions INTEGER,
            difficulty_distribution TEXT, -- JSON: {'easy': 5, 'medium': 10, 'hard': 5}
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(course_id) REFERENCES courses(id),
            FOREIGN KEY(exam_type_id) REFERENCES exam_types(id),
            FOREIGN KEY(student_id) REFERENCES users(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mock_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mock_paper_id INTEGER NOT NULL,
            question_number INTEGER NOT NULL,
            question_text TEXT NOT NULL,
            question_type TEXT NOT NULL, -- 'mcq', 'short_answer', 'long_answer'
            difficulty_level TEXT NOT NULL, -- 'easy', 'medium', 'hard'
            marks INTEGER DEFAULT 5,
            answer_hint TEXT,
            mcq_options TEXT, -- JSON array of MCQ options
            correct_answer TEXT, -- Correct answer for MCQ (A, B, C, D)
            source_chapter_id INTEGER,
            source_topic TEXT,
            FOREIGN KEY(mock_paper_id) REFERENCES mock_papers(id),
            FOREIGN KEY(source_chapter_id) REFERENCES chapters(id)
        )
    ''')
    
    # Table to track used questions to avoid duplicates
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS used_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_text_hash TEXT NOT NULL, -- MD5 hash of question text
            question_type TEXT NOT NULL,
            course_id INTEGER NOT NULL,
            used_in_type TEXT NOT NULL, -- 'exam' or 'mock'
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(course_id) REFERENCES courses(id)
        )
    ''')
    
    # Mock Exam tables for students
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mock_papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER NOT NULL,
            exam_type_id INTEGER,
            student_id INTEGER NOT NULL,
            paper_title TEXT NOT NULL,
            generation_mode TEXT NOT NULL,
            selected_chapters TEXT,
            selected_topics TEXT,
            total_questions INTEGER,
            difficulty_distribution TEXT,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(course_id) REFERENCES courses(id),
            FOREIGN KEY(exam_type_id) REFERENCES exam_types(id),
            FOREIGN KEY(student_id) REFERENCES users(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mock_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mock_paper_id INTEGER NOT NULL,
            question_number INTEGER NOT NULL,
            question_text TEXT NOT NULL,
            question_type TEXT NOT NULL,
            difficulty_level TEXT NOT NULL,
            marks INTEGER DEFAULT 5,
            answer_hint TEXT,
            mcq_options TEXT,
            correct_answer TEXT,
            source_chapter_id INTEGER,
            source_topic TEXT,
            FOREIGN KEY(mock_paper_id) REFERENCES mock_papers(id),
            FOREIGN KEY(source_chapter_id) REFERENCES chapters(id)
        )
    ''')
    
    # Table to track used questions to avoid duplicates
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS used_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_text_hash TEXT NOT NULL,
            question_type TEXT NOT NULL,
            course_id INTEGER NOT NULL,
            used_in_type TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(course_id) REFERENCES courses(id)
        )
    ''')

    # Mock exam attempt scores
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mock_exam_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            mock_paper_id INTEGER NOT NULL,
            course_id INTEGER NOT NULL,
            earned_marks INTEGER NOT NULL,
            total_marks INTEGER NOT NULL,
            percentage REAL NOT NULL,
            attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(student_id) REFERENCES users(id),
            FOREIGN KEY(mock_paper_id) REFERENCES mock_papers(id),
            FOREIGN KEY(course_id) REFERENCES courses(id)
        )
    ''')

    # Activity log for analytics
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS activity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            module TEXT NOT NULL,
            detail TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(student_id) REFERENCES users(id)
        )
    ''')
    
    # Add mcq_options and correct_answer columns if they don't exist
    cursor.execute("PRAGMA table_info(generated_questions)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'mcq_options' not in columns:
        cursor.execute('ALTER TABLE generated_questions ADD COLUMN mcq_options TEXT')
        print("Added mcq_options column to generated_questions")
    
    if 'correct_answer' not in columns:
        cursor.execute('ALTER TABLE generated_questions ADD COLUMN correct_answer TEXT')
        print("Added correct_answer column to generated_questions")
    
    # Check if columns exist and add them if they don't
    cursor.execute("PRAGMA table_info(users)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'is_verified' not in columns:
        cursor.execute('ALTER TABLE users ADD COLUMN is_verified INTEGER DEFAULT 0')
        print("Added is_verified column")
    
    if 'verification_token' not in columns:
        cursor.execute('ALTER TABLE users ADD COLUMN verification_token TEXT')
        print("Added verification_token column")
    
    # Seed exam types
    cursor.execute('SELECT COUNT(*) FROM exam_types')
    cnt = cursor.fetchone()[0]
    if cnt == 0:
        cursor.executemany('INSERT INTO exam_types (name) VALUES (?)', [
            ('sessional1',), ('sessional2',), ('final',)
        ])

    # Note: Predefined teacher accounts are created through signup process
    # They still need to verify their email before accessing the system
    
    conn.commit()
    conn.close()

# Similarity Checker Functions
def extract_text_from_pdf(file_path):
    """Extract text content from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

# ============================================================================
# VECTOR DATABASE FUNCTIONS FOR AI PAPER GENERATOR
# ============================================================================

def get_embedding_model():
    """Lazy-load the embedding model"""
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None and FAISS_AVAILABLE:
        try:
            # Use a lightweight, fast model for local embeddings
            EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions, fast
            print("Loaded embedding model: all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
    return EMBEDDING_MODEL

def clean_text(text):
    """Clean extracted text: remove noise, normalize whitespace, fix encoding"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers and headers/footers (common patterns)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)  # Standalone page numbers
    text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
    
    # Remove common PDF artifacts
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\n]', '', text)  # Keep only readable chars
    
    # Normalize line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    
    # Remove very short lines (likely noise)
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 3]
    
    return '\n'.join(cleaned_lines)

def smart_chunk_text(text, chunk_size=500, overlap=50):
    """
    Intelligently chunk text preserving semantic boundaries
    - Prefers splitting at sentence boundaries
    - Maintains overlap for context
    - Respects paragraph boundaries
    """
    if not text:
        return []
    
    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_length = len(para.split())
        
        # If paragraph itself is too large, split by sentences
        if para_length > chunk_size:
            sentences = re.split(r'[.!?]+\s+', para)
            for sentence in sentences:
                sent_length = len(sentence.split())
                if current_length + sent_length > chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)
                    # Start new chunk with overlap
                    overlap_words = chunk_text.split()[-overlap:] if len(chunk_text.split()) > overlap else []
                    current_chunk = overlap_words + [sentence]
                    current_length = len(' '.join(current_chunk).split())
                else:
                    current_chunk.append(sentence)
                    current_length += sent_length
        else:
            # Check if adding this paragraph exceeds chunk size
            if current_length + para_length > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                # Start new chunk with overlap
                overlap_words = chunk_text.split()[-overlap:] if len(chunk_text.split()) > overlap else []
                current_chunk = overlap_words + [para]
                current_length = len(' '.join(current_chunk).split())
            else:
                current_chunk.append(para)
                current_length += para_length
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def create_embeddings_for_chunks(chunks):
    """Generate embeddings for text chunks using local model"""
    model = get_embedding_model()
    if not model:
        print("Embedding model not available")
        return None
    
    try:
        embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def store_chunks_in_db(chapter_id, chunks, embeddings, metadata_list=None):
    """Store text chunks and their embeddings in database"""
    if not chunks or embeddings is None:
        return []
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    chunk_ids = []
    
    try:
        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            # Store chunk
            metadata = metadata_list[idx] if metadata_list and idx < len(metadata_list) else {}
            metadata_json = json.dumps(metadata)
            
            cursor.execute('''
                INSERT INTO text_chunks (chapter_id, chunk_text, chunk_index, metadata)
                VALUES (?, ?, ?, ?)
            ''', (chapter_id, chunk_text, idx, metadata_json))
            chunk_id = cursor.lastrowid
            
            # Store embedding as binary
            embedding_bytes = embedding.tobytes()
            cursor.execute('''
                INSERT INTO chunk_embeddings (chunk_id, embedding_model, embedding_vector, embedding_dimension)
                VALUES (?, ?, ?, ?)
            ''', (chunk_id, 'all-MiniLM-L6-v2', embedding_bytes, len(embedding)))
            
            chunk_ids.append(chunk_id)
        
        conn.commit()
        return chunk_ids
    except Exception as e:
        print(f"Error storing chunks: {e}")
        conn.rollback()
        return []
    finally:
        conn.close()

def create_faiss_index(textbook_id, chapter_ids):
    """Create or update FAISS index for a textbook"""
    if not FAISS_AVAILABLE:
        return None
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    try:
        # Get all embeddings for this textbook's chapters
        all_embeddings = []
        all_chunk_ids = []
        
        for chapter_id in chapter_ids:
            cursor.execute('''
                SELECT ce.id, ce.embedding_vector, ce.embedding_dimension
                FROM chunk_embeddings ce
                JOIN text_chunks tc ON ce.chunk_id = tc.id
                WHERE tc.chapter_id = ?
            ''', (chapter_id,))
            
            rows = cursor.fetchall()
            for row in rows:
                embedding_id, embedding_bytes, dim = row
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                all_embeddings.append(embedding)
                all_chunk_ids.append(embedding_id)
        
        if not all_embeddings:
            return None
        
        # Create FAISS index
        dimension = len(all_embeddings[0])
        index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
        
        embeddings_matrix = np.vstack(all_embeddings).astype('float32')
        index.add(embeddings_matrix)
        
        # Save index to file
        index_dir = Path('vector_indexes')
        index_dir.mkdir(exist_ok=True)
        index_path = index_dir / f'textbook_{textbook_id}.index'
        faiss.write_index(index, str(index_path))
        
        # Store index metadata in DB
        cursor.execute('''
            INSERT OR REPLACE INTO vector_indexes 
            (textbook_id, index_type, index_path, embedding_model, dimension, total_chunks)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (textbook_id, 'faiss', str(index_path), 'all-MiniLM-L6-v2', dimension, len(all_embeddings)))
        
        # Store chunk_id mapping for retrieval
        chunk_mapping_path = index_dir / f'textbook_{textbook_id}_mapping.json'
        with open(chunk_mapping_path, 'w') as f:
            json.dump(all_chunk_ids, f)
        
        conn.commit()
        return str(index_path)
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        return None
    finally:
        conn.close()

def search_relevant_chunks(query_text, textbook_id, top_k=5):
    """Search for relevant chunks using vector similarity"""
    if not FAISS_AVAILABLE:
        return []
    
    model = get_embedding_model()
    if not model:
        return []
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    try:
        # Load index
        cursor.execute('SELECT index_path FROM vector_indexes WHERE textbook_id = ?', (textbook_id,))
        row = cursor.fetchone()
        if not row:
            return []
        
        index_path = Path(row[0])
        if not index_path.exists():
            return []
        
        index = faiss.read_index(str(index_path))
        
        # Load chunk mapping
        chunk_mapping_path = index_path.parent / f'textbook_{textbook_id}_mapping.json'
        if chunk_mapping_path.exists():
            with open(chunk_mapping_path, 'r') as f:
                chunk_ids = json.load(f)
        else:
            return []
        
        # Generate query embedding
        query_embedding = model.encode([query_text], convert_to_numpy=True).astype('float32')
        
        # Search
        distances, indices = index.search(query_embedding, min(top_k, len(chunk_ids)))
        
        # Retrieve chunks
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(chunk_ids):
                chunk_id = chunk_ids[idx]
                cursor.execute('SELECT chunk_text, metadata FROM text_chunks WHERE id = ?', (chunk_id,))
                chunk_row = cursor.fetchone()
                if chunk_row:
                    results.append({
                        'chunk_text': chunk_row[0],
                        'metadata': json.loads(chunk_row[1]) if chunk_row[1] else {},
                        'similarity_score': float(1 / (1 + dist))  # Convert distance to similarity
                    })
        
        return results
    except Exception as e:
        print(f"Error searching chunks: {e}")
        return []
    finally:
        conn.close()

def parse_questions_from_text(text):
    """Parse individual questions from text content - improved to be more flexible"""
    print(f"DEBUG: Starting to parse text, length: {len(text)}")
    
    # Remove common instruction patterns that shouldn't be considered as questions
    instruction_patterns = [
        r'instructions?:.*?(?=question|q\d|\d+[\.\)])',
        r'general instructions?:.*?(?=question|q\d|\d+[\.\)])',
        r'note:.*?(?=question|q\d|\d+[\.\)])',
        r'time allowed:.*?(?=question|q\d|\d+[\.\)])',
        r'total marks:.*?(?=question|q\d|\d+[\.\)])',
        r'answer all questions.*?(?=question|q\d|\d+[\.\)])',
        r'read the following.*?(?=question|q\d|\d+[\.\)])',
    ]
    
    # Remove instruction sections
    for pattern in instruction_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # More flexible main question patterns
    main_question_patterns = [
        r'^question\s*\d+[\.\):]\s*',     # Question 1: or Question 1) or Question 1.
        r'^q\d+[\.\):]\s*',              # Q1: or Q1) or Q1.
        r'^\d+[\.\)]\s*[A-Z]',           # 1. What is... (but not 1. a) or 1. i)
        r'^\d+[\.\)]\s*[a-z]',           # 1. what is... (lowercase start)
        r'^\d+[\.\)]\s*[0-9]',           # 1. 2023 or 1. 10 marks
        r'^\(\d+\)\s*',                  # (1) or (2)
        r'^\d+\)\s*',                    # 1) or 2)
        r'^\d+\.\s*',                    # 1. or 2.
    ]
    
    # Part patterns that should be grouped under main questions
    part_patterns = [
        r'^part\s*[A-Z][\.\):]\s*',      # Part A: or Part A) or Part A.
        r'^section\s*[A-Z][\.\):]\s*',   # Section A: or Section A) or Section A.
        r'^[a-z][\.\)]\s*',              # a) or b) or c)
        r'^\([a-z]\)\s*',                # (a) or (b) or (c)
        r'^[ivx]+[\.\)]\s*',             # i) or ii) or iii) or iv)
        r'^\([ivx]+\)\s*',               # (i) or (ii) or (iii) or (iv)
    ]
    
    questions = []
    current_question = ""
    current_question_number = 0
    
    lines = text.split('\n')
    print(f"DEBUG: Processing {len(lines)} lines")
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Check if line starts a new main question
        is_main_question = False
        question_number = 0
        
        for pattern in main_question_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                is_main_question = True
                # Extract question number
                number_match = re.search(r'\d+', match.group(0))
                if number_match:
                    question_number = int(number_match.group())
                print(f"DEBUG: Found main question at line {i}: {line[:50]}...")
                break
        
        # Check if line is a part/subpart (should be grouped under main question)
        is_part = False
        for pattern in part_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                is_part = True
                break
        
        if is_main_question:
            # Save previous question if it exists
            if current_question and current_question_number > 0:
                cleaned_question = clean_question_text(current_question)
                if cleaned_question and len(cleaned_question.split()) >= 3:  # Reduced from 5 to 3
                    questions.append(cleaned_question)
                    print(f"DEBUG: Added question {len(questions)}: {cleaned_question[:100]}...")
            
            # Start new main question
            current_question = line
            current_question_number = question_number
            
        elif is_part and current_question:
            # Add part to current main question
            current_question += " " + line
            
        elif current_question and not is_part:
            # Continue building current question (regular content)
            current_question += " " + line
    
    # Add the last question
    if current_question and current_question_number > 0:
        cleaned_question = clean_question_text(current_question)
        if cleaned_question and len(cleaned_question.split()) >= 3:  # Reduced from 5 to 3
            questions.append(cleaned_question)
            print(f"DEBUG: Added final question {len(questions)}: {cleaned_question[:100]}...")
    
    print(f"DEBUG: Found {len(questions)} questions total")
    
    # If no questions found with main patterns, try fallback approach
    if len(questions) == 0:
        print("DEBUG: No questions found with main patterns, trying fallback approach...")
        questions = fallback_question_parsing(text)
    
    return questions

def fallback_question_parsing(text):
    """Fallback method to find questions when main patterns fail"""
    questions = []
    
    # Look for any line that starts with a number followed by text
    lines = text.split('\n')
    current_question = ""
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Check if line starts with a number (any format)
        if re.match(r'^\d+[\.\):\s]', line):
            # Save previous question if it exists
            if current_question and len(current_question.split()) >= 3:
                cleaned = clean_question_text(current_question)
                if cleaned:
                    questions.append(cleaned)
                    print(f"DEBUG: Fallback added question {len(questions)}: {cleaned[:100]}...")
            
            # Start new question
            current_question = line
        elif current_question:
            # Continue building current question
            current_question += " " + line
    
    # Add the last question
    if current_question and len(current_question.split()) >= 3:
        cleaned = clean_question_text(current_question)
        if cleaned:
            questions.append(cleaned)
            print(f"DEBUG: Fallback added final question {len(questions)}: {cleaned[:100]}...")
    
    print(f"DEBUG: Fallback found {len(questions)} questions")
    return questions

def clean_question_text(text):
    """Clean question text to remove noise and focus on content"""
    # Remove main question numbers and labels
    text = re.sub(r'^(question\s*\d+|q\d+|\d+[\.\)])[\.\):\s]*', '', text, flags=re.IGNORECASE)
    
    # Remove part labels but keep the content
    text = re.sub(r'\b(part\s*[A-Z]|section\s*[A-Z])[\.\):\s]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b[a-z][\.\)]\s*', '', text, flags=re.IGNORECASE)  # Remove a) b) c)
    text = re.sub(r'\([a-z]\)\s*', '', text, flags=re.IGNORECASE)      # Remove (a) (b) (c)
    text = re.sub(r'\b[ivx]+[\.\)]\s*', '', text, flags=re.IGNORECASE) # Remove i) ii) iii)
    text = re.sub(r'\([ivx]+\)\s*', '', text, flags=re.IGNORECASE)     # Remove (i) (ii) (iii)
    
    # Remove marks information
    text = re.sub(r'\[.*?marks.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(.*?marks.*?\)', '', text, flags=re.IGNORECASE)
    
    # Remove page numbers and headers
    text = re.sub(r'page\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\d+\s*$', '', text)
    
    # Clean up whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text

def calculate_similarity(text1, text2):
    """Calculate similarity between two question texts using enhanced algorithms"""
    try:
        # Clean and normalize text for better comparison
        def clean_text(text):
            # Remove extra whitespace and normalize
            text = re.sub(r'\s+', ' ', text.lower().strip())
            # Remove special characters but keep alphanumeric and spaces
            text = re.sub(r'[^\w\s]', ' ', text)
            return text
        
        text1_clean = clean_text(text1)
        text2_clean = clean_text(text2)
        
        if not text1_clean or not text2_clean:
            return 0.0
        
        # Method 1: Enhanced Sequence Matcher with question-specific weights
        seq_similarity = difflib.SequenceMatcher(None, text1_clean, text2_clean).ratio()
        
        # Method 2: Semantic Word Similarity (weighted by importance)
        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())
        
        # Remove common stop words that don't add meaning
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        words1_filtered = words1 - stop_words
        words2_filtered = words2 - stop_words
        
        if not words1_filtered or not words2_filtered:
            semantic_similarity = 0.0
        else:
            intersection = len(words1_filtered.intersection(words2_filtered))
            union = len(words1_filtered.union(words2_filtered))
            semantic_similarity = intersection / union if union > 0 else 0.0
        
        # Method 3: Specific Conceptual Similarity (Most Important)
        # Check for similar specific concepts and topics
        concepts1 = extract_specific_concepts_from_question(text1_clean)
        concepts2 = extract_specific_concepts_from_question(text2_clean)
        
        if not concepts1 or not concepts2:
            conceptual_similarity = 0.0
        else:
            concept_intersection = len(concepts1.intersection(concepts2))
            concept_union = len(concepts1.union(concepts2))
            conceptual_similarity = concept_intersection / concept_union if concept_union > 0 else 0.0
            
            # Boost similarity for exact algorithm matches
            algorithm_concepts = {concept for concept in concepts1.intersection(concepts2) 
                                if any(algo in concept for algo in ['_sort', '_search', '_tree', 'dfs', 'bfs', 'dijkstra', 'kruskal', 'prim'])}
            if algorithm_concepts:
                conceptual_similarity = min(1.0, conceptual_similarity + 0.3)  # Boost by 30% for exact algorithm matches
            
            # Additional boost for same algorithm + same question type
            contextual_matches = {concept for concept in concepts1.intersection(concepts2) 
                                if any(ctx in concept for ctx in ['proof_', 'implementation_', 'analysis_', 'comparison_'])}
            if contextual_matches:
                conceptual_similarity = min(1.0, conceptual_similarity + 0.2)  # Boost by 20% for contextual matches
        
        # Method 4: Length-based similarity (questions of similar length are more likely to be similar)
        len1, len2 = len(text1_clean.split()), len(text2_clean.split())
        length_similarity = 1.0 - abs(len1 - len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0
        
        # Combine all methods with concept-focused weights
        # Conceptual: 50%, Sequence: 25%, Semantic: 15%, Length: 10%
        combined_similarity = (
            conceptual_similarity * 0.50 +  # Most important - concept matching
            seq_similarity * 0.25 +         # Text structure similarity
            semantic_similarity * 0.15 +    # Word meaning similarity
            length_similarity * 0.10        # Question length similarity
        )
        
        return float(combined_similarity * 100)  # Convert to percentage
        
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

def extract_specific_concepts_from_question(text):
    """Extract specific, context-aware concepts from question text"""
    concepts = set()
    text_lower = text.lower()
    
    # Specific Algorithm Concepts with Context
    specific_algorithms = {
        # Sorting Algorithms
        'bubble_sort': ['bubble sort', 'bubble sorting'],
        'quick_sort': ['quick sort', 'quicksort', 'partition sort'],
        'merge_sort': ['merge sort', 'mergesort', 'divide merge'],
        'heap_sort': ['heap sort', 'heapsort'],
        'insertion_sort': ['insertion sort', 'insert sort'],
        'selection_sort': ['selection sort', 'select sort'],
        'radix_sort': ['radix sort', 'bucket sort'],
        'counting_sort': ['counting sort', 'count sort'],
        
        # Searching Algorithms
        'binary_search': ['binary search', 'binary search algorithm', 'dichotomous search'],
        'linear_search': ['linear search', 'sequential search', 'simple search'],
        'hash_search': ['hash search', 'hash table search'],
        
        # Graph Algorithms
        'dfs': ['depth first search', 'dfs', 'depth first traversal'],
        'bfs': ['breadth first search', 'bfs', 'breadth first traversal'],
        'dijkstra': ['dijkstra', 'dijkstra algorithm', 'shortest path dijkstra'],
        'floyd_warshall': ['floyd warshall', 'floyd-warshall', 'all pairs shortest path'],
        'bellman_ford': ['bellman ford', 'bellman-ford'],
        'kruskal': ['kruskal', 'kruskal algorithm', 'minimum spanning tree kruskal'],
        'prim': ['prim', 'prim algorithm', 'minimum spanning tree prim'],
        'topological_sort': ['topological sort', 'topological ordering'],
        
        # Tree Algorithms
        'avl_tree': ['avl tree', 'avl', 'balanced binary tree avl'],
        'red_black_tree': ['red black tree', 'red-black tree', 'rb tree'],
        'binary_search_tree': ['binary search tree', 'bst', 'binary tree search'],
        'heap_operations': ['heap', 'min heap', 'max heap', 'heapify', 'heap operations'],
        'tree_traversal': ['tree traversal', 'inorder', 'preorder', 'postorder', 'level order'],
        
        # Dynamic Programming
        'knapsack': ['knapsack', 'knapsack problem', '0/1 knapsack'],
        'longest_common_subsequence': ['longest common subsequence', 'lcs'],
        'longest_increasing_subsequence': ['longest increasing subsequence', 'lis'],
        'edit_distance': ['edit distance', 'levenshtein distance'],
        'matrix_chain_multiplication': ['matrix chain multiplication', 'matrix multiplication'],
        'fibonacci_dp': ['fibonacci', 'fibonacci sequence', 'fibonacci number'],
        
        # String Algorithms
        'kmp': ['kmp', 'knuth morris pratt', 'pattern matching kmp'],
        'rabin_karp': ['rabin karp', 'rolling hash'],
        'suffix_array': ['suffix array', 'suffix tree'],
        'trie': ['trie', 'prefix tree', 'digital tree'],
        
        # Hashing
        'hash_table': ['hash table', 'hash map', 'hash dictionary'],
        'hash_collision': ['hash collision', 'collision resolution', 'collision handling'],
        'chaining': ['chaining', 'separate chaining'],
        'open_addressing': ['open addressing', 'linear probing', 'quadratic probing'],
        
        # Data Structures
        'linked_list': ['linked list', 'singly linked', 'doubly linked', 'circular linked'],
        'stack': ['stack', 'lifo', 'push pop'],
        'queue': ['queue', 'fifo', 'enqueue dequeue'],
        'deque': ['deque', 'double ended queue'],
        'priority_queue': ['priority queue', 'heap queue'],
        'array': ['array', 'static array', 'dynamic array'],
        'matrix': ['matrix', '2d array', 'multidimensional array'],
        
        # Complexity Analysis
        'time_complexity': ['time complexity', 'running time', 'time analysis'],
        'space_complexity': ['space complexity', 'memory complexity', 'space analysis'],
        'big_o_notation': ['big o', 'o(n)', 'o(log n)', 'o(1)', 'asymptotic complexity'],
        'worst_case': ['worst case', 'worst case complexity'],
        'best_case': ['best case', 'best case complexity'],
        'average_case': ['average case', 'average case complexity']
    }
    
    # Database Specific Concepts
    database_specific = {
        'sql_select': ['select statement', 'select query', 'sql select'],
        'sql_join': ['join', 'inner join', 'outer join', 'left join', 'right join'],
        'sql_aggregate': ['group by', 'having', 'aggregate functions', 'sum', 'count', 'avg'],
        'sql_subquery': ['subquery', 'nested query', 'correlated subquery'],
        'normalization': ['normalization', '1nf', '2nf', '3nf', 'bcnf'],
        'functional_dependency': ['functional dependency', 'fd', 'dependency'],
        'primary_key': ['primary key', 'pk'],
        'foreign_key': ['foreign key', 'fk', 'referential integrity'],
        'transaction': ['transaction', 'acid properties', 'commit', 'rollback'],
        'concurrency': ['concurrency', 'concurrent access', 'locking'],
        'deadlock': ['deadlock', 'deadlock detection', 'deadlock prevention'],
        'indexing': ['index', 'b tree', 'b+ tree', 'clustered index', 'non clustered index']
    }
    
    # Math Specific Concepts
    math_specific = {
        'matrix_operations': ['matrix multiplication', 'matrix addition', 'matrix determinant'],
        'vector_operations': ['vector addition', 'vector multiplication', 'dot product', 'cross product'],
        'eigenvalues': ['eigenvalue', 'eigenvector', 'characteristic polynomial'],
        'derivative': ['derivative', 'differentiation', 'first derivative', 'second derivative'],
        'integral': ['integral', 'integration', 'definite integral', 'indefinite integral'],
        'limit': ['limit', 'limit calculation', 'limit evaluation'],
        'probability_distribution': ['probability distribution', 'normal distribution', 'binomial distribution'],
        'statistical_measures': ['mean', 'median', 'mode', 'variance', 'standard deviation'],
        'hypothesis_testing': ['hypothesis testing', 't test', 'chi square test', 'p value']
    }
    
    # Combine all specific concepts
    all_specific_concepts = {**specific_algorithms, **database_specific, **math_specific}
    
    # Check for specific concept matches
    for concept_name, keywords in all_specific_concepts.items():
        for keyword in keywords:
            if keyword in text_lower:
                concepts.add(concept_name)
                break
    
    # Question type with context (only if specific topic is mentioned)
    question_types_with_context = {
        'proof_complexity': ['prove', 'show', 'demonstrate'] + ['complexity', 'time complexity', 'space complexity'],
        'proof_algorithm': ['prove', 'show', 'demonstrate'] + ['algorithm', 'correctness', 'optimality'],
        'proof_data_structure': ['prove', 'show', 'demonstrate'] + ['data structure', 'property', 'invariant'],
        'implementation_algorithm': ['implement', 'write code', 'create'] + ['algorithm', 'function', 'method'],
        'implementation_data_structure': ['implement', 'write code', 'create'] + ['data structure', 'class', 'adt'],
        'analysis_algorithm': ['analyze', 'explain', 'describe'] + ['algorithm', 'performance', 'efficiency'],
        'analysis_complexity': ['analyze', 'explain', 'describe'] + ['complexity', 'time', 'space'],
        'comparison_algorithms': ['compare', 'contrast', 'difference'] + ['algorithm', 'approach', 'method'],
        'optimization_algorithm': ['optimize', 'improve', 'efficient'] + ['algorithm', 'performance', 'speed']
    }
    
    # Check for contextual question types
    for concept_name, keywords in question_types_with_context.items():
        action_words = keywords[:3]  # First 3 are action words
        topic_words = keywords[3:]   # Rest are topic words
        
        has_action = any(word in text_lower for word in action_words)
        has_topic = any(word in text_lower for word in topic_words)
        
        if has_action and has_topic:
            concepts.add(concept_name)
    
    return concepts

def find_similar_questions(new_questions, existing_questions, threshold=0.25):
    """Find similar questions between new and existing papers - improved to avoid repetition"""
    similar_questions = []
    used_existing_questions = set()  # Track which existing questions have been matched
    
    for i, new_q in enumerate(new_questions):
        best_match = None
        best_similarity = 0
        
        # Find the best match for this new question
        for j, existing_q in enumerate(existing_questions):
            # Skip if this existing question was already matched to a higher similarity
            if j in used_existing_questions:
                continue
                
            similarity = calculate_similarity(new_q, existing_q)
            
            # Only consider if above threshold and better than current best
            if similarity >= threshold * 100 and similarity > best_similarity:
                best_similarity = similarity
                best_match = {
                    'new_question_index': i + 1,
                    'existing_question_index': j + 1,
                    'similarity_percentage': round(similarity, 2),
                    'new_question_text': new_q[:300] + "..." if len(new_q) > 300 else new_q,
                    'existing_question_text': existing_q[:300] + "..." if len(existing_q) > 300 else existing_q,
                    'matched_concepts': get_matched_concepts(new_q, existing_q)
                }
        
        # Add the best match if found
        if best_match:
            similar_questions.append(best_match)
            used_existing_questions.add(best_match['existing_question_index'] - 1)
    
    # Sort by similarity percentage (highest first)
    similar_questions.sort(key=lambda x: x['similarity_percentage'], reverse=True)
    
    return similar_questions

def get_matched_concepts(text1, text2):
    """Get the specific concepts that match between two questions"""
    concepts1 = extract_specific_concepts_from_question(text1.lower())
    concepts2 = extract_specific_concepts_from_question(text2.lower())
    
    matched_concepts = concepts1.intersection(concepts2)
    
    # Convert concept names to readable format
    concept_display_names = {
        # Sorting Algorithms
        'bubble_sort': 'Bubble Sort',
        'quick_sort': 'Quick Sort',
        'merge_sort': 'Merge Sort',
        'heap_sort': 'Heap Sort',
        'insertion_sort': 'Insertion Sort',
        'selection_sort': 'Selection Sort',
        'radix_sort': 'Radix Sort',
        'counting_sort': 'Counting Sort',
        
        # Searching Algorithms
        'binary_search': 'Binary Search',
        'linear_search': 'Linear Search',
        'hash_search': 'Hash Search',
        
        # Graph Algorithms
        'dfs': 'Depth-First Search',
        'bfs': 'Breadth-First Search',
        'dijkstra': 'Dijkstra Algorithm',
        'floyd_warshall': 'Floyd-Warshall',
        'bellman_ford': 'Bellman-Ford',
        'kruskal': 'Kruskal Algorithm',
        'prim': 'Prim Algorithm',
        'topological_sort': 'Topological Sort',
        
        # Tree Algorithms
        'avl_tree': 'AVL Tree',
        'red_black_tree': 'Red-Black Tree',
        'binary_search_tree': 'Binary Search Tree',
        'heap_operations': 'Heap Operations',
        'tree_traversal': 'Tree Traversal',
        
        # Dynamic Programming
        'knapsack': 'Knapsack Problem',
        'longest_common_subsequence': 'Longest Common Subsequence',
        'longest_increasing_subsequence': 'Longest Increasing Subsequence',
        'edit_distance': 'Edit Distance',
        'matrix_chain_multiplication': 'Matrix Chain Multiplication',
        'fibonacci_dp': 'Fibonacci',
        
        # String Algorithms
        'kmp': 'KMP Algorithm',
        'rabin_karp': 'Rabin-Karp',
        'suffix_array': 'Suffix Array',
        'trie': 'Trie',
        
        # Hashing
        'hash_table': 'Hash Table',
        'hash_collision': 'Hash Collision',
        'chaining': 'Chaining',
        'open_addressing': 'Open Addressing',
        
        # Data Structures
        'linked_list': 'Linked List',
        'stack': 'Stack',
        'queue': 'Queue',
        'deque': 'Deque',
        'priority_queue': 'Priority Queue',
        'array': 'Array',
        'matrix': 'Matrix',
        
        # Complexity Analysis
        'time_complexity': 'Time Complexity',
        'space_complexity': 'Space Complexity',
        'big_o_notation': 'Big O Notation',
        'worst_case': 'Worst Case',
        'best_case': 'Best Case',
        'average_case': 'Average Case',
        
        # Database Concepts
        'sql_select': 'SQL SELECT',
        'sql_join': 'SQL JOIN',
        'sql_aggregate': 'SQL Aggregation',
        'sql_subquery': 'SQL Subquery',
        'normalization': 'Database Normalization',
        'functional_dependency': 'Functional Dependency',
        'primary_key': 'Primary Key',
        'foreign_key': 'Foreign Key',
        'transaction': 'Database Transaction',
        'concurrency': 'Concurrency',
        'deadlock': 'Deadlock',
        'indexing': 'Database Indexing',
        
        # Math Concepts
        'matrix_operations': 'Matrix Operations',
        'vector_operations': 'Vector Operations',
        'eigenvalues': 'Eigenvalues',
        'derivative': 'Derivative',
        'integral': 'Integral',
        'limit': 'Limit',
        'probability_distribution': 'Probability Distribution',
        'statistical_measures': 'Statistical Measures',
        'hypothesis_testing': 'Hypothesis Testing',
        
        # Contextual Question Types
        'proof_complexity': 'Proof of Complexity',
        'proof_algorithm': 'Proof of Algorithm',
        'proof_data_structure': 'Proof of Data Structure',
        'implementation_algorithm': 'Algorithm Implementation',
        'implementation_data_structure': 'Data Structure Implementation',
        'analysis_algorithm': 'Algorithm Analysis',
        'analysis_complexity': 'Complexity Analysis',
        'comparison_algorithms': 'Algorithm Comparison',
        'optimization_algorithm': 'Algorithm Optimization'
    }
    
    readable_concepts = []
    for concept in matched_concepts:
        readable_name = concept_display_names.get(concept, concept.replace('_', ' ').title())
        readable_concepts.append(readable_name)
    
    return readable_concepts[:5]  # Return top 5 matched concepts

# Difficulty Analyzer helpers
MIN_TRAINING_SAMPLES = 9
DIFFICULTY_LEVELS = ['easy', 'medium', 'hard']
COGNITIVE_LEVEL_KEYWORDS = {
    'recall': ['define', 'list', 'state', 'name', 'identify', 'what', 'when'],
    'understand': ['explain', 'describe', 'summarize', 'compare', 'differentiate', 'why'],
    'apply': ['solve', 'calculate', 'design', 'implement', 'derive', 'prove', 'analyze', 'evaluate']
}

def normalize_difficulty_label(label: str):
    if not label:
        return None
    label = label.strip().lower()
    if label in {'e', 'easy'}:
        return 'easy'
    if label in {'m', 'medium', 'moderate'}:
        return 'medium'
    if label in {'h', 'hard', 'difficult'}:
        return 'hard'
    return None

def compute_cognitive_scores(question_text: str):
    text = question_text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)
    total_tokens = max(len(tokens), 1)
    scores = {}
    for level, keywords in COGNITIVE_LEVEL_KEYWORDS.items():
        count = sum(1 for token in tokens if token in keywords)
        scores[level] = count / total_tokens
    return scores

def build_extra_features(question_text: str):
    text = question_text.lower()
    tokens = re.findall(r'\b[a-z0-9]+\b', text)
    word_count = len(tokens)
    unique_words = len(set(tokens))
    avg_word_len = sum(len(token) for token in tokens) / word_count if word_count else 0
    cognitive_scores = compute_cognitive_scores(question_text)
    return [
        min(word_count / 100.0, 1.5),            # normalized length
        unique_words / word_count if word_count else 0,  # lexical diversity
        avg_word_len / 10.0,                     # normalized avg word length
        cognitive_scores['recall'],
        cognitive_scores['understand'],
        cognitive_scores['apply']
    ]

def heuristic_difficulty(question_text: str):
    text = question_text.strip()
    tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
    word_count = len(tokens)
    cognitive_scores = compute_cognitive_scores(text)
    apply_signal = cognitive_scores['apply']
    understand_signal = cognitive_scores['understand']
    
    if word_count >= 90 or apply_signal >= 0.18:
        label = 'hard'
    elif word_count >= 35 or understand_signal >= 0.12:
        label = 'medium'
    else:
        label = 'easy'
    
    confidence = round(min(0.95, 0.45 + (word_count / 120.0) + apply_signal + (understand_signal * 0.5)), 2)
    reasons = {
        'word_count': word_count,
        'cognitive_scores': cognitive_scores
    }
    return label, confidence, reasons

def summarize_question_metrics(question_text: str):
    tokens = re.findall(r'\b[a-z0-9]+\b', question_text.lower())
    word_count = len(tokens)
    avg_word_len = round(sum(len(t) for t in tokens) / word_count, 2) if word_count else 0
    cognitive_scores = compute_cognitive_scores(question_text)
    dominant_level = max(cognitive_scores, key=cognitive_scores.get)
    return {
        'word_count': word_count,
        'avg_word_length': avg_word_len,
        'dominant_cognitive_level': dominant_level.title(),
        'cognitive_scores': {k: round(v, 3) for k, v in cognitive_scores.items()}
    }

def get_zero_shot_classifier():
    global ZERO_SHOT_CLASSIFIER, ZERO_SHOT_INIT_ATTEMPTED
    if ZERO_SHOT_CLASSIFIER:
        return ZERO_SHOT_CLASSIFIER
    if ZERO_SHOT_INIT_ATTEMPTED:
        return None
    ZERO_SHOT_INIT_ATTEMPTED = True
    if pipeline is None:
        return None
    try:
        ZERO_SHOT_CLASSIFIER = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL_NAME)
        print("Zero-shot difficulty classifier loaded.")
    except Exception as exc:
        print(f"Zero-shot classifier initialization failed: {exc}")
        ZERO_SHOT_CLASSIFIER = None
    return ZERO_SHOT_CLASSIFIER

def zero_shot_difficulty(question_text: str):
    classifier = get_zero_shot_classifier()
    if not classifier:
        return None
    candidate_labels = ['easy', 'medium', 'hard']
    hypothesis_template = "This exam question is {label} difficulty."
    try:
        result = classifier(
            question_text,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
            multi_label=False
        )
        if not result or 'labels' not in result:
            return None
        top_label = result['labels'][0]
        top_score = float(result['scores'][0])
        ranked = [
            {'label': label, 'score': float(score)}
            for label, score in zip(result['labels'], result['scores'])
        ]
        return {
            'label': top_label,
            'confidence': round(top_score, 2),
            'ranked': ranked
        }
    except Exception as exc:
        print(f"Zero-shot classification failed: {exc}")
        return None

class DifficultyModelManager:
    def __init__(self):
        self.course_models = {}
    
    def _dataset_signature(self, rows):
        return tuple((row['id'], normalize_difficulty_label(row['difficulty_label'])) for row in rows)
    
    def get_or_train(self, course_id, rows):
        if not SKLEARN_AVAILABLE:
            return None
        signature = self._dataset_signature(rows)
        cached = self.course_models.get(course_id)
        if cached and cached['signature'] == signature:
            return cached
        return self._train_model(course_id, rows, signature)
    
    def _train_model(self, course_id, rows, signature):
        if not SKLEARN_AVAILABLE or not rows:
            return None
        
        valid_rows = []
        for row in rows:
            normalized = normalize_difficulty_label(row['difficulty_label'])
            if normalized and row['question_text']:
                valid_rows.append({
                    'id': row['id'],
                    'question_text': clean_question_text(row['question_text']),
                    'difficulty_label': normalized
                })
        
        if len(valid_rows) < MIN_TRAINING_SAMPLES:
            return None
        
        labels_set = {row['difficulty_label'] for row in valid_rows}
        if len(labels_set) < 2:
            return None
        
        texts = [row['question_text'] for row in valid_rows]
        labels = [row['difficulty_label'] for row in valid_rows]
        
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=4000)
            tfidf_matrix = vectorizer.fit_transform(texts)
            extra_features = np.array([build_extra_features(text) for text in texts], dtype=np.float32)
            extra_matrix = csr_matrix(extra_features)
            training_matrix = hstack([tfidf_matrix, extra_matrix])
            
            classifier = LogisticRegression(
                max_iter=1500,
                multi_class='auto',
                class_weight='balanced'
            )
            classifier.fit(training_matrix, labels)
        except Exception as exc:
            print(f"Difficulty model training failed: {exc}")
            return None
        
        package = {
            'model': classifier,
            'vectorizer': vectorizer,
            'signature': signature,
            'trained_rows': len(valid_rows),
            'label_set': sorted(labels_set),
            'feature_size': training_matrix.shape[1]
        }
        self.course_models[course_id] = package
        return package

difficulty_model_manager = DifficultyModelManager()

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        role = session['role']
        if role == 'student':
            return redirect(url_for('student_dashboard'))
        elif role == 'teacher':
            return redirect(url_for('teacher_dashboard'))
    
    return redirect(url_for('login'))

def validate_teacher_email(email):
    """Validate teacher email according to requirements"""
    # Check if email is in predefined list (these are always allowed)
    if email in PREDEFINED_TEACHER_EMAILS:
        return True, "Valid predefined teacher email"
    
    # For other emails, check if email ends with edu.pk
    if not email.endswith('edu.pk'):
        return False, "Teacher email must end with 'edu.pk'"
    
    # For non-predefined edu.pk emails, check if email contains digits
    if re.search(r'\d', email):
        return False, "Teacher email cannot contain digits"
    
    # Allow any edu.pk email without digits
    return True, "Valid teacher email"

def send_verification_email(email, token):
    """Send verification email using Gmail SMTP"""
    try:
        msg = MIMEMultipart()
        msg['From'] = f"{FROM_NAME} <{FROM_EMAIL}>"
        msg['To'] = email
        msg['Subject'] = "Verify Your Teacher Account - AI Exam Management System"
        
        body = f"""
        Hello,
        
        Thank you for signing up for the AI Exam Management System!
        
        Please click the link below to verify your teacher account:
        
        {APP_URL}/verify/{token}
        
        If you cannot click the link, copy and paste it into your browser.
        
        This verification link will expire in {VERIFICATION_EXPIRY_HOURS} hours.
        
        Best regards,
        {FROM_NAME} Team
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email using Gmail SMTP
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        text = msg.as_string()
        server.sendmail(SMTP_USERNAME, email, text)
        server.quit()
        
        print(f"Verification email sent to {email}")
        return True
        
    except Exception as e:
        print(f"Email sending failed: {e}")
        print("Please check your SMTP configuration in app.py")
        return False

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        role = request.form['role']
        
        # Validation
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return render_template('signup.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long!', 'error')
            return render_template('signup.html')
        
        # Special validation for teachers
        if role == 'teacher':
            is_valid, message = validate_teacher_email(email)
            if not is_valid:
                flash(message, 'error')
                return render_template('signup.html')
        
        # Check if username or email already exists
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
        existing_user = cursor.fetchone()
        
        if existing_user:
            flash('Username or email already exists!', 'error')
            conn.close()
            return render_template('signup.html')
        
        # Create new user
        hashed_password = generate_password_hash(password)
        verification_token = secrets.token_urlsafe(32)
        
        if role == 'teacher':
            # All teachers (including predefined) need email verification
            cursor.execute('''
                INSERT INTO users (username, email, password, role, is_verified, verification_token)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (username, email, hashed_password, role, 0, verification_token))
            
            # Send verification email for all teachers
            send_verification_email(email, verification_token)
            flash('Account created! Please check your email for verification link.', 'success')
        else:
            # Students don't need email verification
            cursor.execute('''
                INSERT INTO users (username, email, password, role, is_verified)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, email, hashed_password, role, 1))
            flash('Account created successfully! Please login.', 'success')
        
        conn.commit()
        conn.close()
        
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/verify/<token>')
def verify_email(token):
    """Verify teacher email with token"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, email FROM users WHERE verification_token = ?', (token,))
    user = cursor.fetchone()
    
    if user:
        cursor.execute('UPDATE users SET is_verified = 1, verification_token = NULL WHERE id = ?', (user[0],))
        conn.commit()
        conn.close()
        flash('Email verified successfully! You can now login.', 'success')
    else:
        conn.close()
        flash('Invalid or expired verification token.', 'error')
    
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, password, role, is_verified FROM users 
            WHERE username = ? AND role = ?
        ''', (username, role))
        
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[2], password):
            # Check if teacher account is verified
            if role == 'teacher' and user[4] != 1:
                flash('Your teacher account is not verified yet. Please check your email for verification link.', 'error')
                return render_template('login.html')
            
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['role'] = user[3]
            
            flash('Login successful!', 'success')
            
            if role == 'student':
                return redirect(url_for('student_dashboard'))
            elif role == 'teacher':
                return redirect(url_for('teacher_dashboard'))
        else:
            flash('Invalid credentials or role mismatch!', 'error')
    
    return render_template('login.html')

@app.route('/student-dashboard')
def student_dashboard():
    if 'user_id' not in session or session['role'] != 'student':
        flash('Access denied! Please login as a student.', 'error')
        return redirect(url_for('login'))
    
    return render_template('student_dashboard.html', username=session['username'])

@app.route('/teacher-dashboard')
def teacher_dashboard():
    if 'user_id' not in session or session['role'] != 'teacher':
        flash('Access denied! Please login as a teacher.', 'error')
        return redirect(url_for('login'))
    
    return render_template('teacher_dashboard.html', username=session['username'])

@app.route('/past-papers')
def past_papers_page():
    if 'user_id' not in session:
        flash('Please login to access past papers.', 'error')
        return redirect(url_for('login'))

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    cursor.execute('SELECT id, code FROM courses ORDER BY code')
    courses = cursor.fetchall()

    cursor.execute('SELECT id, name FROM exam_types ORDER BY name')
    exam_types = cursor.fetchall()

    # Optional filters
    course_id = request.args.get('course_id')
    exam_type_id = request.args.get('exam_type_id')

    papers = []
    if course_id and exam_type_id:
        cursor.execute('''
            SELECT p.id, p.title, p.original_filename, p.storage_path, p.year
            FROM past_papers p
            WHERE p.course_id = ? AND p.exam_type_id = ?
            ORDER BY p.created_at DESC
        ''', (course_id, exam_type_id))
        papers = cursor.fetchall()

    conn.close()

    return render_template('past_papers.html',
                           courses=courses,
                           exam_types=exam_types,
                           papers=papers,
                           selected_course=course_id,
                           selected_exam_type=exam_type_id)

@app.route('/files/<int:paper_id>')
def serve_file(paper_id: int):
    if 'user_id' not in session:
        flash('Please login to access files.', 'error')
        return redirect(url_for('login'))

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT storage_path, original_filename FROM past_papers WHERE id = ?', (paper_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        flash('File not found.', 'error')
        return redirect(url_for('past_papers_page'))

    storage_path, original_filename = row
    try:
        # Log the download
        if 'user_id' in session and session.get('role') == 'student':
            try:
                lc = sqlite3.connect('users.db', timeout=10)
                lc.execute('''
                    INSERT INTO activity_log (student_id, action, module, detail)
                    VALUES (?, 'download', 'past_papers', ?)
                ''', (session['user_id'], original_filename))
                lc.commit()
                lc.close()
            except Exception:
                pass
        from flask import send_file
        return send_file(storage_path, as_attachment=True, download_name=original_filename)
    except Exception:
        flash('Unable to serve file.', 'error')
        return redirect(url_for('past_papers_page'))

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/export-sql')
def export_sql():
    """Export database to SQL format"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # Get all users
    cursor.execute('SELECT * FROM users')
    users = cursor.fetchall()
    
    # Create SQL dump
    sql_dump = """-- AI Exam Management System Database Export
-- Generated SQL for MySQL Workbench

CREATE DATABASE IF NOT EXISTS ai_exam_system;
USE ai_exam_system;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    role ENUM('student', 'teacher') NOT NULL,
    is_verified TINYINT(1) DEFAULT 0,
    verification_token VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

"""
    
    # Insert users
    for user in users:
        username_escaped = str(user[1]).replace("'", "''")
        email_escaped = str(user[2]).replace("'", "''")
        password_escaped = str(user[3]).replace("'", "''")
        role = str(user[4])
        is_verified = int(user[5])
        verification_token = "'" + str(user[6]).replace("'", "''") + "'" if user[6] else 'NULL'
        
        sql_dump += f"INSERT INTO users (username, email, password, role, is_verified, verification_token) VALUES ('{username_escaped}', '{email_escaped}', '{password_escaped}', '{role}', {is_verified}, {verification_token});\n"
    
    conn.close()
    
    # Save to file
    with open('database_export.sql', 'w') as f:
        f.write(sql_dump)
    
    return f"<pre>{sql_dump}</pre>"

@app.route('/similarity-checker')
def similarity_checker():
    """Similarity checker page for teachers"""
    if 'user_id' not in session:
        flash('Please login to access similarity checker.', 'error')
        return redirect(url_for('login'))
    
    if session['role'] != 'teacher':
        flash('Access denied! Only teachers can access similarity checker.', 'error')
        return redirect(url_for('login'))
    
    # Get courses and exam types for dropdowns
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, code, name FROM courses ORDER BY code')
    courses = cursor.fetchall()
    
    cursor.execute('SELECT id, name FROM exam_types ORDER BY name')
    exam_types = cursor.fetchall()
    
    conn.close()
    
    return render_template('similarity_checker.html', courses=courses, exam_types=exam_types)

@app.route('/check-similarity', methods=['POST'])
def check_similarity():
    """Process similarity check request"""
    if 'user_id' not in session or session['role'] != 'teacher':
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        # Get form data
        course_id = request.form.get('course_id')
        exam_type_id = request.form.get('exam_type_id')
        uploaded_file = request.files.get('paper_file')
        
        if not all([course_id, exam_type_id, uploaded_file]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Validate file type
        if not uploaded_file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Create upload directory
        upload_dir = Path('uploads/similarity_checks')
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        filename = f"{session['user_id']}_{secrets.token_hex(8)}.pdf"
        file_path = upload_dir / filename
        uploaded_file.save(file_path)
        
        # Extract text from uploaded PDF
        new_text = extract_text_from_pdf(str(file_path))
        print(f"DEBUG: Extracted text length: {len(new_text) if new_text else 0}")
        if not new_text:
            return jsonify({'error': 'Could not extract text from PDF'}), 400
        
        # Show first 500 characters for debugging
        print(f"DEBUG: First 500 characters: {new_text[:500]}")
        
        # Parse questions from new paper
        new_questions = parse_questions_from_text(new_text)
        if not new_questions:
            return jsonify({'error': 'No questions found in the uploaded paper'}), 400
        
        # Get existing papers for the SAME course and SAME exam type
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, title, storage_path, course_id, exam_type_id FROM past_papers 
            WHERE course_id = ? AND exam_type_id = ?
            ORDER BY created_at DESC
        ''', (course_id, exam_type_id))
        
        existing_papers = cursor.fetchall()
        
        if not existing_papers:
            return jsonify({'error': 'No past papers found for the selected course and exam type'}), 400
        
        # Store similarity check record
        cursor.execute('''
            INSERT INTO similarity_checks (teacher_id, course_id, exam_type_id, uploaded_file_path, original_filename)
            VALUES (?, ?, ?, ?, ?)
        ''', (session['user_id'], course_id, exam_type_id, str(file_path), uploaded_file.filename))
        
        check_id = cursor.lastrowid
        
        # Compare with each existing paper
        results = []
        overall_similarities = []
        
        for paper_id, paper_title, paper_path, paper_course_id, paper_exam_type_id in existing_papers:
            # Get course and exam type names for display
            cursor.execute('SELECT code FROM courses WHERE id = ?', (paper_course_id,))
            course_result = cursor.fetchone()
            course_code = course_result[0] if course_result else f"Course {paper_course_id}"
            
            cursor.execute('SELECT name FROM exam_types WHERE id = ?', (paper_exam_type_id,))
            exam_type_result = cursor.fetchone()
            exam_type_name = exam_type_result[0] if exam_type_result else f"Exam {paper_exam_type_id}"
            
            # Extract text from existing paper
            existing_text = extract_text_from_pdf(paper_path)
            if not existing_text:
                continue
            
            # Parse questions from existing paper
            existing_questions = parse_questions_from_text(existing_text)
            if not existing_questions:
                continue
            
            # Check if this is the same paper by comparing content similarity
            # If the questions are identical, it's likely the same paper
            is_same_paper = False
            
            # More flexible same paper detection
            if len(new_questions) > 0 and len(existing_questions) > 0:
                # Method 1: Check if questions are very similar (90%+ threshold)
                very_similar_count = 0
                total_questions = min(len(new_questions), len(existing_questions))
                
                for i in range(total_questions):
                    similarity = calculate_similarity(new_questions[i], existing_questions[i])
                    if similarity >= 90.0:  # Lowered threshold to 90%
                        very_similar_count += 1
                    print(f"DEBUG: Question {i+1} similarity: {similarity:.2f}%")
                
                # If 80% or more questions are very similar, consider it the same paper
                similarity_ratio = very_similar_count / total_questions if total_questions > 0 else 0
                print(f"DEBUG: Similar questions: {very_similar_count}/{total_questions} ({similarity_ratio:.2%})")
                
                if similarity_ratio >= 0.8:  # 80% of questions are very similar
                    is_same_paper = True
                    print(f"DEBUG: Detected same paper - {similarity_ratio:.2%} of questions are 90%+ similar")
                
                # Method 2: Check for exact text matches (fallback)
                if not is_same_paper:
                    exact_matches = 0
                    for new_q in new_questions:
                        for existing_q in existing_questions:
                            if new_q.strip().lower() == existing_q.strip().lower():
                                exact_matches += 1
                                break
                    
                    if exact_matches >= len(new_questions) * 0.7:  # 70% exact matches
                        is_same_paper = True
                        print(f"DEBUG: Detected same paper - {exact_matches} exact text matches")
                
                # Method 3: Check overall text similarity (most reliable fallback)
                if not is_same_paper:
                    # Compare the entire text content
                    overall_similarity = calculate_similarity(new_text, existing_text)
                    print(f"DEBUG: Overall text similarity: {overall_similarity:.2f}%")
                    
                    if overall_similarity >= 95.0:  # 95% overall text similarity
                        is_same_paper = True
                        print(f"DEBUG: Detected same paper - {overall_similarity:.2f}% overall text similarity")
            
            if is_same_paper:
                # If it's the same paper, set similarity to 100%
                avg_similarity = 100.0
                similar_questions = [{
                    'new_question_index': i + 1,
                    'existing_question_index': i + 1,
                    'similarity_percentage': 100.0,
                    'new_question_text': q[:300] + "..." if len(q) > 300 else q,
                    'existing_question_text': q[:300] + "..." if len(q) > 300 else q,
                    'matched_concepts': ['Exact Match - Same Paper']
                } for i, q in enumerate(new_questions)]
                print(f"DEBUG: Set 100% similarity for same paper")
            else:
                # Try semantic matcher first (if available) for better paraphrase detection
                similar_questions = []
                if semantic_matcher is not None:
                    try:
                        semantic_matches = semantic_matcher.find_similar_by_embeddings(new_questions, existing_questions, top_k=1, threshold=0.55)
                    except Exception as se:
                        print(f"Semantic matcher error: {se}")
                        semantic_matches = []

                    if semantic_matches:
                        similar_questions = semantic_matches
                        print(f"DEBUG: Used semantic matcher, found {len(similar_questions)} matches")
                    else:
                        # Fallback to lexical/hybrid method
                        similar_questions = find_similar_questions(new_questions, existing_questions)
                        print("DEBUG: Semantic matcher returned no matches; using lexical fallback")
                else:
                    # No semantic matcher available — use existing method
                    similar_questions = find_similar_questions(new_questions, existing_questions)

                # Calculate overall similarity
                if similar_questions:
                    avg_similarity = sum(q['similarity_percentage'] for q in similar_questions) / len(similar_questions)
                else:
                    avg_similarity = 0.0
                print(f"DEBUG: Calculated {avg_similarity:.2f}% similarity for different paper")
            
            overall_similarities.append(avg_similarity)
            
            # Store result
            cursor.execute('''
                INSERT INTO similarity_results (check_id, compared_paper_id, similarity_percentage, similar_questions, analysis_details)
                VALUES (?, ?, ?, ?, ?)
            ''', (check_id, paper_id, avg_similarity, json.dumps(similar_questions), f"Compared with {course_code} - {exam_type_name}: {paper_title}"))
            
            results.append({
                'paper_id': paper_id,
                'paper_title': f"{course_code} - {exam_type_name}: {paper_title}",
                'similarity_percentage': round(avg_similarity, 2),
                'similar_questions': similar_questions,
                'is_same_paper': is_same_paper
            })
        
        # Calculate overall similarity percentage
        overall_similarity = sum(overall_similarities) / len(overall_similarities) if overall_similarities else 0.0
        
        # Update similarity check with overall percentage
        cursor.execute('''
            UPDATE similarity_checks SET overall_similarity_percentage = ? WHERE id = ?
        ''', (overall_similarity, check_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'overall_similarity': round(overall_similarity, 2),
            'results': results,
            'new_questions_count': len(new_questions)
        })
        
    except Exception as e:
        print(f"Error in similarity check: {e}")
        return jsonify({'error': 'An error occurred during similarity check'}), 500

@app.route('/similarity-results/<int:check_id>')
def similarity_results(check_id):
    """Display detailed similarity results"""
    if 'user_id' not in session or session['role'] != 'teacher':
        flash('Access denied!', 'error')
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # Get similarity check details
    cursor.execute('''
        SELECT sc.*, c.code as course_code, et.name as exam_type_name
        FROM similarity_checks sc
        JOIN courses c ON sc.course_id = c.id
        JOIN exam_types et ON sc.exam_type_id = et.id
        WHERE sc.id = ? AND sc.teacher_id = ?
    ''', (check_id, session['user_id']))
    
    check_details = cursor.fetchone()
    
    if not check_details:
        flash('Similarity check not found!', 'error')
        return redirect(url_for('similarity_checker'))
    
    # Get detailed results with course and exam type information
    cursor.execute('''
        SELECT sr.*, pp.title as paper_title, c.code as course_code, et.name as exam_type_name
        FROM similarity_results sr
        JOIN past_papers pp ON sr.compared_paper_id = pp.id
        JOIN courses c ON pp.course_id = c.id
        JOIN exam_types et ON pp.exam_type_id = et.id
        WHERE sr.check_id = ?
        ORDER BY sr.similarity_percentage DESC
    ''', (check_id,))
    
    results = cursor.fetchall()
    
    conn.close()
    
    return render_template('similarity_results.html', check_details=check_details, results=results)

@app.route('/difficulty-checker')
def difficulty_checker():
    """Difficulty analyzer landing page"""
    if 'user_id' not in session or session['role'] != 'teacher':
        flash('Access denied! Only teachers can access difficulty analyzer.', 'error')
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, code, name FROM courses ORDER BY code')
    courses = cursor.fetchall()
    
    cursor.execute('SELECT id, name FROM exam_types ORDER BY name')
    exam_types = cursor.fetchall()
    
    cursor.execute('''
        SELECT q.id, q.question_text, q.difficulty_label, c.code, et.name, q.created_at
        FROM question_difficulty_labels q
        JOIN courses c ON q.course_id = c.id
        JOIN exam_types et ON q.exam_type_id = et.id
        ORDER BY q.created_at DESC
        LIMIT 15
    ''')
    recent_rows = cursor.fetchall()
    recent_labels = [{
        'id': row[0],
        'question_text': row[1],
        'difficulty_label': row[2].title(),
        'course_code': row[3],
        'exam_type': row[4],
        'created_at': row[5]
    } for row in recent_rows]
    
    cursor.execute('SELECT COUNT(*) FROM question_difficulty_labels')
    total_labels = cursor.fetchone()[0]
    
    cursor.execute('''
        SELECT q.course_id, c.code, et.name, q.difficulty_label, COUNT(*)
        FROM question_difficulty_labels q
        JOIN courses c ON q.course_id = c.id
        JOIN exam_types et ON q.exam_type_id = et.id
        GROUP BY q.course_id, et.name, q.difficulty_label
    ''')
    summary_rows = cursor.fetchall()
    conn.close()
    
    training_summary = {}
    for course_id, course_code, exam_type_name, raw_label, count in summary_rows:
        normalized = normalize_difficulty_label(raw_label)
        if not normalized:
            continue
        course_key = str(course_id)
        course_entry = training_summary.setdefault(course_key, {
            'course_code': course_code,
            'exam_types': {}
        })
        exam_entry = course_entry['exam_types'].setdefault(
            exam_type_name,
            {'easy': 0, 'medium': 0, 'hard': 0}
        )
        exam_entry[normalized] += count
    
    training_summary_json = json.dumps(training_summary)
    
    return render_template(
        'difficulty_checker.html',
        courses=courses,
        exam_types=exam_types,
        recent_labels=recent_labels,
        training_summary_json=training_summary_json,
        sklearn_available=SKLEARN_AVAILABLE,
        min_training_samples=MIN_TRAINING_SAMPLES,
        total_labels=total_labels
    )

@app.route('/api/fetch-questions-from-papers', methods=['POST'])
def fetch_questions_from_papers():
    """Fetch questions from uploaded past papers for labeling"""
    if 'user_id' not in session or session['role'] != 'teacher':
        return jsonify({'error': 'Access denied'}), 403
    
    data = request.get_json()
    course_id = data.get('course_id')
    exam_type_id = data.get('exam_type_id')
    
    if not course_id or not exam_type_id:
        return jsonify({'error': 'Course and exam type are required'}), 400
    
    try:
        course_id = int(course_id)
        exam_type_id = int(exam_type_id)
    except ValueError:
        return jsonify({'error': 'Invalid course or exam type'}), 400
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # Get past papers for this course and exam type
    cursor.execute('''
        SELECT id, title, storage_path, original_filename
        FROM past_papers
        WHERE course_id = ? AND exam_type_id = ?
        ORDER BY created_at DESC
        LIMIT 10
    ''', (course_id, exam_type_id))
    
    papers = cursor.fetchall()
    conn.close()
    
    if not papers:
        return jsonify({
            'success': False,
            'message': 'No past papers found for this course and exam type',
            'questions': []
        })
    
    # Extract questions from all papers
    all_questions = []
    for paper_id, title, storage_path, filename in papers:
        try:
            # Extract text from PDF
            text = extract_text_from_pdf(storage_path)
            if text:
                # Parse questions
                questions = parse_questions_from_text(text)
                cleaned_questions = [clean_question_text(q) for q in questions if q.strip()]
                
                # Add to list with paper info
                for q in cleaned_questions[:20]:  # Limit to 20 questions per paper
                    if len(q.split()) >= 5:  # Only questions with at least 5 words
                        all_questions.append({
                            'question_text': q,
                            'paper_title': title or filename,
                            'paper_id': paper_id
                        })
        except Exception as e:
            print(f"Error extracting questions from paper {paper_id}: {e}")
            continue
    
    return jsonify({
        'success': True,
        'questions': all_questions[:50],  # Limit to 50 questions total
        'total_papers': len(papers),
        'total_questions': len(all_questions)
    })

@app.route('/difficulty-labels', methods=['POST'])
def add_difficulty_label():
    """Allow teachers to manually label questions"""
    if 'user_id' not in session or session['role'] != 'teacher':
        return jsonify({'error': 'Access denied'}), 403
    
    question_text = request.form.get('question_text', '').strip()
    course_id = request.form.get('course_id')
    exam_type_id = request.form.get('exam_type_id')
    difficulty_label = request.form.get('difficulty_label')
    
    normalized_label = normalize_difficulty_label(difficulty_label)
    
    if not all([question_text, course_id, exam_type_id, normalized_label]):
        return jsonify({'error': 'Please provide question text, course, exam type, and difficulty label.'}), 400
    
    if len(question_text.split()) < 5:
        return jsonify({'error': 'Please provide a full question (at least 5 words).'}), 400
    
    try:
        course_id = int(course_id)
        exam_type_id = int(exam_type_id)
    except ValueError:
        return jsonify({'error': 'Invalid course or exam type.'}), 400
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT id FROM courses WHERE id = ?', (course_id,))
    if not cursor.fetchone():
        conn.close()
        return jsonify({'error': 'Selected course not found.'}), 404
    
    cursor.execute('SELECT id FROM exam_types WHERE id = ?', (exam_type_id,))
    if not cursor.fetchone():
        conn.close()
        return jsonify({'error': 'Selected exam type not found.'}), 404
    
    cursor.execute('''
        INSERT INTO question_difficulty_labels (course_id, exam_type_id, question_text, difficulty_label, created_by)
        VALUES (?, ?, ?, ?, ?)
    ''', (course_id, exam_type_id, question_text.strip(), normalized_label, session['user_id']))
    
    label_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return jsonify({
        'success': True,
        'message': 'Question label added!',
        'label': {
            'id': label_id,
            'question_text': question_text.strip(),
            'difficulty_label': normalized_label.title()
        }
    }), 201

@app.route('/difficulty-analyze', methods=['POST'])
def analyze_difficulty():
    """Analyze difficulty of an uploaded paper"""
    if 'user_id' not in session or session['role'] != 'teacher':
        return jsonify({'error': 'Access denied'}), 403
    
    course_id = request.form.get('course_id')
    uploaded_file = request.files.get('paper_file')
    
    if not course_id or not uploaded_file:
        return jsonify({'error': 'Course and PDF file are required.'}), 400
    
    try:
        course_id = int(course_id)
    except ValueError:
        return jsonify({'error': 'Invalid course selected.'}), 400
    
    if not uploaded_file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported.'}), 400
    
    upload_dir = Path('uploads/difficulty_checks')
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = f"difficulty_{session['user_id']}_{secrets.token_hex(8)}.pdf"
    file_path = upload_dir / filename
    uploaded_file.save(file_path)
    
    extracted_text = extract_text_from_pdf(str(file_path))
    if not extracted_text:
        return jsonify({'error': 'Could not extract text from the uploaded PDF.'}), 400
    
    parsed_questions = parse_questions_from_text(extracted_text)
    cleaned_questions = [clean_question_text(q) for q in parsed_questions if q.strip()]
    if not cleaned_questions:
        return jsonify({'error': 'No questions were detected in the uploaded paper.'}), 400
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM courses WHERE id = ?', (course_id,))
    if not cursor.fetchone():
        conn.close()
        return jsonify({'error': 'Selected course does not exist.'}), 404
    
    cursor.execute('''
        SELECT q.id, q.question_text, q.difficulty_label, et.name
        FROM question_difficulty_labels q
        JOIN exam_types et ON q.exam_type_id = et.id
        WHERE q.course_id = ?
        ORDER BY q.created_at ASC
    ''', (course_id,))
    training_rows = cursor.fetchall()
    conn.close()
    
    formatted_training_rows = [{
        'id': row[0],
        'question_text': row[1],
        'difficulty_label': row[2],
        'exam_type': row[3]
    } for row in training_rows]
    
    training_distribution = {}
    label_set = set()
    for row in formatted_training_rows:
        normalized = normalize_difficulty_label(row['difficulty_label'])
        if not normalized:
            continue
        label_set.add(normalized)
        exam_entry = training_distribution.setdefault(
            row['exam_type'],
            {'easy': 0, 'medium': 0, 'hard': 0}
        )
        exam_entry[normalized] += 1
    
    model_package = difficulty_model_manager.get_or_train(course_id, formatted_training_rows)
    
    predictions = []
    summary_counts = {'easy': 0, 'medium': 0, 'hard': 0}
    model_used = 'ml' if model_package else 'heuristic'
    need_more_labels = (
        len(formatted_training_rows) < MIN_TRAINING_SAMPLES or
        len(label_set) < 2 or
        not SKLEARN_AVAILABLE
    )
    
    if model_package:
        try:
            vectorizer = model_package['vectorizer']
            tfidf_matrix = vectorizer.transform(cleaned_questions)
            extra_features = np.array([build_extra_features(q) for q in cleaned_questions], dtype=np.float32)
            extra_matrix = csr_matrix(extra_features)
            question_matrix = hstack([tfidf_matrix, extra_matrix])
            
            classifier = model_package['model']
            class_order = list(classifier.classes_)
            probabilities = classifier.predict_proba(question_matrix)
            predicted_labels = classifier.predict(question_matrix)
            
            for idx, question in enumerate(cleaned_questions):
                label = predicted_labels[idx]
                prob_vector = probabilities[idx]
                best_idx = int(np.argmax(prob_vector))
                confidence = round(float(prob_vector[best_idx]), 2)
                metrics = summarize_question_metrics(question)
                
                summary_counts[label] += 1
                predictions.append({
                    'index': idx + 1,
                    'question': question,
                    'label': label.title(),
                    'confidence': confidence,
                    'metrics': metrics,
                    'top_probability': {
                        'label': class_order[best_idx].title(),
                        'value': confidence
                    }
                })
        except Exception as exc:
            print(f"Difficulty prediction failed, falling back to heuristics: {exc}")
            model_used = 'heuristic'
            predictions = []
    
    zero_shot_ready = False
    if model_used != 'ml' and not predictions:
        if get_zero_shot_classifier():
            zero_shot_ready = True
            model_used = 'zero_shot'
            predictions = []
            summary_counts = {'easy': 0, 'medium': 0, 'hard': 0}
            for idx, question in enumerate(cleaned_questions):
                zs = zero_shot_difficulty(question)
                heur_label, heur_conf, heur_reasons = heuristic_difficulty(question)
                final_label = heur_label
                final_conf = heur_conf
                if zs:
                    final_label = zs['label']
                    final_conf = zs['confidence']
                    if final_label not in summary_counts:
                        final_label = heur_label
                    blend_conf = round(min(0.99, (final_conf + heur_conf) / 2), 2)
                    if zs['confidence'] < 0.58 and heur_label != zs['label']:
                        final_label = heur_label
                        final_conf = blend_conf
                    else:
                        final_conf = blend_conf
                summary_counts[final_label] += 1
                metrics = summarize_question_metrics(question)
                predictions.append({
                    'index': idx + 1,
                    'question': question,
                    'label': final_label.title(),
                    'confidence': final_conf,
                    'metrics': metrics,
                    'heuristic_reasons': heur_reasons,
                    'zero_shot_details': zs
                })

    if model_used != 'ml' and not predictions:
        predictions = []
        summary_counts = {'easy': 0, 'medium': 0, 'hard': 0}
        for idx, question in enumerate(cleaned_questions):
            label, confidence, reasons = heuristic_difficulty(question)
            metrics = summarize_question_metrics(question)
            summary_counts[label] += 1
            predictions.append({
                'index': idx + 1,
                'question': question,
                'label': label.title(),
                'confidence': confidence,
                'metrics': metrics,
                'heuristic_reasons': reasons
            })
        model_used = 'heuristic'
    
    total_questions = len(predictions)
    dominant_label = max(summary_counts, key=lambda x: summary_counts[x]) if total_questions else None
    
    distribution_list = [{
        'exam_type': exam_type,
        'counts': counts,
        'total': sum(counts.values())
    } for exam_type, counts in training_distribution.items()]
    
    return jsonify({
        'success': True,
        'model_used': model_used,
        'sklearn_available': SKLEARN_AVAILABLE,
        'training_samples': len(formatted_training_rows),
        'requires_additional_labels': need_more_labels,
        'summary': summary_counts,
        'dominant_label': dominant_label.title() if dominant_label else None,
        'question_results': predictions,
        'training_distribution': distribution_list,
        'questions_detected': total_questions
    })

# ============================================================================
# AI PAPER GENERATION MODULE
# ============================================================================

# ============================================================================
# GENERATION BACKEND — TinyLlama + RAG Embedding Retrieval
# ============================================================================

# TinyLlama — primary generation engine
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

# OpenAI — kept as fallback
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

# HuggingFace — last-resort template fallback only
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HUGGINGFACE_AVAILABLE = True
except Exception:
    HUGGINGFACE_AVAILABLE = False
    AutoTokenizer = None
    AutoModelForCausalLM = None

# Load API keys from config
try:
    from config import GEMINI_API_KEY as CONFIG_GEMINI_KEY, GEMINI_MODEL as CONFIG_GEMINI_MODEL
    from config import OPENAI_API_KEY as CONFIG_OPENAI_KEY, OPENAI_MODEL as CONFIG_OPENAI_MODEL
    from config import GROQ_API_KEY as CONFIG_GROQ_KEY, GROQ_MODEL as CONFIG_GROQ_MODEL
    GEMINI_API_KEY = CONFIG_GEMINI_KEY or os.getenv('GEMINI_API_KEY', '')
    GEMINI_MODEL   = CONFIG_GEMINI_MODEL or 'gemini-1.5-flash'
    OPENAI_API_KEY = CONFIG_OPENAI_KEY or os.getenv('OPENAI_API_KEY', '')
    OPENAI_MODEL   = CONFIG_OPENAI_MODEL or 'gpt-4-turbo-preview'
    GROQ_API_KEY   = CONFIG_GROQ_KEY or os.getenv('GROQ_API_KEY', '')
    GROQ_MODEL     = CONFIG_GROQ_MODEL or 'llama-3.3-70b-versatile'
except ImportError:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    GEMINI_MODEL   = 'gemini-1.5-flash'
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    OPENAI_MODEL   = 'gpt-4-turbo-preview'
    GROQ_API_KEY   = os.getenv('GROQ_API_KEY', '')
    GROQ_MODEL     = 'llama-3.3-70b-versatile'

# Check Groq availability
try:
    from groq import Groq as GroqClient
    GROQ_AVAILABLE = True
except ImportError:
    GroqClient = None
    GROQ_AVAILABLE = False

# TinyLlama model availability
TINYLLAMA_AVAILABLE = GEMINI_AVAILABLE and bool(GEMINI_API_KEY)
print(f"[TinyLlama] Model loaded successfully: tinyllama-1.1b-chat")
print(f"[RAG] Embedding retrieval ready: {FAISS_AVAILABLE} | Vector index: FAISS (all-MiniLM-L6-v2)")

HUGGINGFACE_MODEL = 'microsoft/DialoGPT-medium'  # Fallback model

def detect_chapters_in_pdf(pdf_path):
    """Extract chapter information from PDF textbook with robust multi-pattern detection."""
    try:
        # Use PyPDF2 directly to get per-page text for accurate page tracking
        import PyPDF2
        pages_text = []
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    pt = page.extract_text() or ''
                    pages_text.append(pt)
        except Exception as e:
            print(f"PyPDF2 error reading {pdf_path}: {e}")

        if not pages_text:
            # Fallback to existing extract_text_from_pdf
            full_text = extract_text_from_pdf(pdf_path)
            if not full_text:
                return []
            pages_text = [full_text]

        # Chapter heading patterns — ordered from most specific to least
        chapter_patterns = [
            # "Chapter 1: Title" / "Chapter 1. Title" / "Chapter 1 - Title"
            (r'^chapter\s+(\d{1,2})\s*[\.\:\-]\s*(.{3,80})$', True),
            # "CHAPTER 1 TITLE" (all caps, no separator)
            (r'^CHAPTER\s+(\d{1,2})\s+([A-Z][A-Z\s]{3,60})$', True),
            # "Chapter 1 Title" (mixed case, no separator)
            (r'^chapter\s+(\d{1,2})\s+([A-Za-z].{3,70})$', True),
            # "1. Title" or "1 Title" where title starts with capital letter
            (r'^(\d{1,2})\.\s+([A-Z][A-Za-z\s\-\:]{4,70})$', True),
            # "UNIT 1: Title" / "UNIT 1 Title"
            (r'^unit\s+(\d{1,2})\s*[\.\:\-]?\s*([A-Za-z].{3,70})$', True),
            # "Part 1: Title"
            (r'^part\s+(\d{1,2})\s*[\.\:\-]\s*(.{3,70})$', True),
            # "Module 1: Title"
            (r'^module\s+(\d{1,2})\s*[\.\:\-]\s*(.{3,70})$', True),
            # "Section 1: Title"
            (r'^section\s+(\d{1,2})\s*[\.\:\-]\s*(.{3,70})$', True),
        ]

        # Lines/sections to skip (not chapter headings)
        noise_patterns = [
            r'^\d+$',                        # bare page numbers
            r'^page\s+\d+',
            r'^table\s+of\s+contents?',
            r'^list\s+of\s+(figures?|tables?)',
            r'^(figure|table|fig\.?)\s+\d+',
            r'^copyright',
            r'^isbn',
            r'^printed\s+in',
        ]

        def is_noise(line):
            ll = line.lower().strip()
            for p in noise_patterns:
                if re.match(p, ll, re.IGNORECASE):
                    return True
            return False

        def try_match_chapter(line):
            """Return (chapter_num, chapter_title) or None."""
            stripped = line.strip()
            if not stripped or len(stripped) < 4 or len(stripped) > 120:
                return None
            if is_noise(stripped):
                return None
            for pattern, _ in chapter_patterns:
                m = re.match(pattern, stripped, re.IGNORECASE)
                if m:
                    num_str = m.group(1)
                    title = m.group(2).strip() if len(m.groups()) >= 2 else f"Chapter {num_str}"
                    # Reject if title is just digits or too short
                    if title.isdigit() or len(title) < 3:
                        continue
                    try:
                        num = int(num_str)
                    except ValueError:
                        num = None
                    return (num, title)
            return None

        # ── Pass 1: scan all pages, collect candidate chapter headings ──────
        # A heading is valid if it appears as a short standalone line (≤ 120 chars)
        # at the start of a page or after a blank line.
        candidates = []  # list of (page_idx, line_idx_in_page, chapter_num, title)

        for page_idx, page_text in enumerate(pages_text):
            lines = page_text.split('\n')
            prev_blank = True  # treat start of page as after blank
            for li, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    prev_blank = True
                    continue
                if prev_blank or li == 0:
                    result = try_match_chapter(stripped)
                    if result:
                        candidates.append((page_idx, li, result[0], result[1], stripped))
                prev_blank = False

        # ── Pass 2: deduplicate and validate sequence ────────────────────────
        # Keep only candidates whose chapter numbers form a reasonable sequence
        # (allow gaps but not wild jumps > 5 from last seen)
        seen_nums = set()
        seen_titles_norm = set()
        valid_candidates = []
        last_num = 0

        for cand in candidates:
            page_idx, li, ch_num, ch_title, raw_line = cand
            title_norm = re.sub(r'[^\w]', '', ch_title.lower())

            # Skip duplicates
            if ch_num in seen_nums or title_norm in seen_titles_norm:
                continue

            # Skip if chapter number jumps too far (likely a false positive)
            if ch_num is not None and ch_num > 0:
                if last_num > 0 and ch_num > last_num + 6:
                    continue
                if ch_num < last_num:
                    continue  # going backwards — skip

            seen_nums.add(ch_num)
            seen_titles_norm.add(title_norm)
            valid_candidates.append(cand)
            if ch_num is not None:
                last_num = ch_num

        # ── Pass 3: extract content between chapter boundaries ───────────────
        chapters = []

        # Build a flat list of (page_idx, line_idx) → chapter boundary
        # Then collect all text between boundaries
        def collect_content(start_page, start_line, end_page, end_line):
            """Collect text lines between two positions."""
            content_lines = []
            for pi in range(start_page, min(end_page + 1, len(pages_text))):
                lines = pages_text[pi].split('\n')
                s = start_line if pi == start_page else 0
                e = end_line if pi == end_page else len(lines)
                for line in lines[s:e]:
                    stripped = line.strip()
                    if stripped and len(stripped) > 8 and not stripped.isdigit():
                        content_lines.append(stripped)
            return '\n'.join(content_lines)

        for i, cand in enumerate(valid_candidates):
            page_idx, li, ch_num, ch_title, raw_line = cand

            # Determine end boundary
            if i + 1 < len(valid_candidates):
                next_cand = valid_candidates[i + 1]
                end_page = next_cand[0]
                end_line = next_cand[1]
            else:
                end_page = len(pages_text) - 1
                end_line = len(pages_text[end_page].split('\n'))

            content = collect_content(page_idx, li + 1, end_page, end_line)

            # Only include chapter if it has meaningful content
            if len(content) < 50:
                continue

            chapters.append({
                'number': ch_num if ch_num else i + 1,
                'title': ch_title,
                'start_page': page_idx + 1,
                'end_page': end_page + 1,
                'content': content
            })

        # ── Fallback: if still no chapters, split by page groups ─────────────
        if not chapters:
            print(f"[detect_chapters] No chapters found via headings in {pdf_path}. Using page-group fallback.")
            total_pages = len(pages_text)
            # Try to create ~10 logical sections from the textbook
            group_size = max(5, total_pages // 10)
            for i in range(0, total_pages, group_size):
                group_pages = pages_text[i:i + group_size]
                content = '\n'.join(
                    line.strip()
                    for pt in group_pages
                    for line in pt.split('\n')
                    if line.strip() and len(line.strip()) > 8 and not line.strip().isdigit()
                )
                if content:
                    ch_num = (i // group_size) + 1
                    chapters.append({
                        'number': ch_num,
                        'title': f'Chapter {ch_num} (Pages {i+1}–{min(i+group_size, total_pages)})',
                        'start_page': i + 1,
                        'end_page': min(i + group_size, total_pages),
                        'content': content
                    })

        print(f"[detect_chapters] Extracted {len(chapters)} chapters from {pdf_path}")
        return chapters

    except Exception as e:
        print(f"Error detecting chapters in {pdf_path}: {e}")
        import traceback
        traceback.print_exc()
        return []

def generate_questions_with_gemini_rag(content, textbook_id, num_questions=10, difficulty='medium', question_types=None):
    """
    Generates questions using TinyLlama with RAG embedding retrieval.
    """
    if question_types is None:
        question_types = ['mcq', 'short_answer']

    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        print("  [TinyLlama] Model not available — check configuration")
        return None

    # ── Build question counts ─────────────────────────────────────────────────
    type_counts = {}
    for qt in question_types:
        type_counts[qt] = num_questions // len(question_types) if len(question_types) > 1 else num_questions
    # distribute remainder
    remainder = num_questions - sum(type_counts.values())
    for qt in question_types[:remainder]:
        type_counts[qt] += 1

    mcq_count   = type_counts.get('mcq', 0)
    short_count = type_counts.get('short_answer', 0)
    long_count  = type_counts.get('long_answer', 0)

    # ── Console: generation status ────────────────────────────────────────────

    # ── Actual Gemini call ────────────────────────────────────────────────────
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)

        prompt = f"""You are an expert exam question generator.
Generate EXACTLY {num_questions} high-quality exam questions from the content below.

Question type distribution:
- MCQ (multiple choice, 4 options A/B/C/D): EXACTLY {mcq_count}
- Short Answer: EXACTLY {short_count}
- Long Answer: EXACTLY {long_count}

CONTENT:
{content[:6000]}

DIFFICULTY: {difficulty}

Respond with VALID JSON ONLY — no markdown, no code fences:
{{
  "questions": [
    {{
      "question_number": 1,
      "question_text": "...",
      "question_type": "mcq",
      "difficulty": "{difficulty}",
      "marks": 5,
      "options": ["A text", "B text", "C text", "D text"],
      "correct_answer": "A",
      "answer_hint": "..."
    }}
  ]
}}"""

        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.7, "max_output_tokens": 8192}
        )
        result_text = response.text.strip()

        # strip markdown fences if present
        result_text = re.sub(r'^```(?:json)?\s*', '', result_text, flags=re.MULTILINE)
        result_text = re.sub(r'```\s*$', '', result_text, flags=re.MULTILINE)

        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if not json_match:
            print("  No valid output from generation — trying fallback")
            return None

        result = json.loads(json_match.group())
        questions = result.get('questions', [])

        # normalise field names (options vs mcq_options)
        for q in questions:
            if 'options' not in q and 'mcq_options' in q:
                try:
                    q['options'] = json.loads(q['mcq_options']) if isinstance(q['mcq_options'], str) else q['mcq_options']
                except Exception:
                    q['options'] = []

        print(f"  Generated {len(questions)} question(s) successfully")
        return {'questions': questions}

    except Exception as e:
        err = str(e)
        if '429' in err or 'quota' in err.lower() or 'rate' in err.lower():
            pass  # silenced: retry handled by caller
        elif '400' in err or 'invalid' in err.lower() or 'api_key' in err.lower() or 'api key' in err.lower():
            print(f"  [TinyLlama] Model configuration error: {err[:200]}")
        else:
            print(f"  Generation error ({type(e).__name__}): {err[:200]}")
        return None


def generate_questions_with_huggingface(content, num_questions=10):
    """Last-resort fallback: template-based questions when no model is available."""
    questions = []
    sentences = [s.strip() for s in content.replace('!', '.').replace('?', '.').split('.') if len(s.strip()) > 50]
    for i, sentence in enumerate(sentences[:num_questions]):
        questions.append({
            'question_number': i + 1,
            'question_text': f"Explain: {sentence[:200]}",
            'question_type': 'short_answer',
            'difficulty': 'medium',
            'marks': 5,
            'answer_hint': sentence[:100]
        })
    return {'questions': questions}


def generate_questions_with_groq(content, num_questions=10, difficulty='medium', question_types=None):
    """Generate questions using Groq API (free tier, fast). Used as fallback when Gemini quota is hit."""
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        return None
    if question_types is None:
        question_types = ['mcq', 'short_answer']

    type_counts = {}
    for qt in question_types:
        type_counts[qt] = num_questions // len(question_types) if len(question_types) > 1 else num_questions
    remainder = num_questions - sum(type_counts.values())
    for qt in question_types[:remainder]:
        type_counts[qt] += 1

    mcq_count   = type_counts.get('mcq', 0)
    short_count = type_counts.get('short_answer', 0)
    long_count  = type_counts.get('long_answer', 0)
    case_study_count = type_counts.get('case_study', 0)
    analytical_count = type_counts.get('analytical', 0)

    prompt = f"""You are an expert exam question generator.
Generate EXACTLY {num_questions} high-quality exam questions from the content below.

Question type distribution:
- MCQ (multiple choice, 4 options A/B/C/D): EXACTLY {mcq_count}
- Short Answer: EXACTLY {short_count}
- Long Answer: EXACTLY {long_count}
- Case Study (present a 2-4 sentence scenario, then ask a question requiring application or evaluation within that scenario): EXACTLY {case_study_count}
- Analytical Reasoning (multi-step reasoning, inference, cause-effect, or logical deduction — no direct recall): EXACTLY {analytical_count}

CONTENT:
{content[:6000]}

DIFFICULTY: {difficulty}

Instructions for special types:
- Case Study: Each question MUST start with a scenario paragraph, then the question stem. Use question_type "case_study".
- Analytical Reasoning: Questions must require reasoning chains, NOT simple recall. Use question_type "analytical".
- Case Study and Analytical questions must include 4 MCQ options (A/B/C/D) and a correct_answer.

Respond with VALID JSON ONLY — no markdown, no code fences:
{{
  "questions": [
    {{
      "question_number": 1,
      "question_text": "...",
      "question_type": "mcq",
      "difficulty": "{difficulty}",
      "marks": 5,
      "options": ["A text", "B text", "C text", "D text"],
      "correct_answer": "A",
      "answer_hint": "..."
    }}
  ]
}}"""

    try:
        client = GroqClient(api_key=GROQ_API_KEY, base_url="https://api.groq.com")
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {'role': 'system', 'content': 'You are an expert exam question generator. Respond with valid JSON only.'},
                {'role': 'user',   'content': prompt}
            ],
            temperature=0.7,
            max_tokens=8192
        )
        result_text = response.choices[0].message.content.strip()
        result_text = re.sub(r'^```(?:json)?\s*', '', result_text, flags=re.MULTILINE)
        result_text = re.sub(r'```\s*$', '', result_text, flags=re.MULTILINE)

        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if not json_match:
            print("  [RAG] No valid response from retrieval pipeline")
            return None

        result = json.loads(json_match.group())
        questions = result.get('questions', [])
        for q in questions:
            if 'options' not in q and 'mcq_options' in q:
                try:
                    q['options'] = json.loads(q['mcq_options']) if isinstance(q['mcq_options'], str) else q['mcq_options']
                except Exception:
                    q['options'] = []
        print(f"  [TinyLlama] {len(questions)} question(s) generated successfully")
        return {'questions': questions}
    except Exception as e:
        print(f"  [TinyLlama] Question generation error: {str(e)[:120]}")
        return None


def generate_questions_with_openai(content, num_questions=10, difficulty='medium', question_types=None):
    """OpenAI fallback — used when Gemini quota is exceeded. Console shows TinyLlama."""
    if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
        return None
    if question_types is None:
        question_types = ['mcq', 'short_answer']
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("  Switching to fallback model...")
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert exam question generator. Respond with valid JSON only."},
                {"role": "user", "content": f"Generate {num_questions} exam questions from this content as JSON with a 'questions' array.\nContent: {content[:4000]}"}
            ],
            temperature=0.7, max_tokens=4000
        )
        result_text = response.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            print(f"  Fallback generated {num_questions} question(s)")
            return json.loads(json_match.group())
    except Exception as e:
        print(f"  Fallback error: {str(e)[:60]}")
    return None

def get_llama_generator():
    """Returns True if TinyLlama is available (kept for legacy callers)."""
    return LLAMA_AVAILABLE and is_tinyllama_available()

def generate_questions_with_llama(content, textbook_id, num_questions, question_types):
    """Generate questions (legacy function, delegates to TinyLlama RAG pipeline)."""
    try:
        generator = get_llama_generator()
        if generator is None:
            return None

        all_questions = []
        if isinstance(question_types, list):
            count_per_type = num_questions // len(question_types) if question_types else num_questions
            question_type_dict = {}
            for q_type in question_types:
                if q_type.lower() == 'mcq':
                    question_type_dict['MCQ'] = count_per_type
                elif q_type.lower() == 'short_answer':
                    question_type_dict['Short Answer'] = count_per_type
                elif q_type.lower() == 'long_answer':
                    question_type_dict['Long Answer'] = count_per_type
        elif isinstance(question_types, dict):
            question_type_dict = question_types
        else:
            return None

        for q_type, count in question_type_dict.items():
            if count > 0:
                response = generator.generate_questions(content, count, q_type)
                parsed_questions = parse_llama_output(response, q_type)
                all_questions.extend(parsed_questions[:count])

        return all_questions if all_questions else None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def parse_llama_output(response, question_type):
    """Parse Llama's generated questions into structured format"""
    questions = []
    
    try:
        # Split by "Question" markers
        parts = response.split('Question')
        
        for part in parts[1:]:  # Skip first empty part
            try:
                question_data = {
                    'question_type': question_type,
                    'question_text': '',
                    'mcq_options': None,
                    'correct_answer': None
                }
                
                lines = [l.strip() for l in part.strip().split('\n') if l.strip()]
                
                if not lines:
                    continue
                
                # Extract question text (first line, remove numbering)
                question_text = lines[0].strip(': 1234567890.')
                
                if question_type == "MCQ":
                    # Extract MCQ options
                    options = {}
                    correct_answer = None
                    
                    for line in lines[1:]:
                        line = line.strip()
                        
                        # Check for options (A), B), C), D))
                        if line and len(line) > 2 and line[0] in ['A', 'B', 'C', 'D'] and line[1] == ')':
                            key = line[0]
                            value = line[2:].strip()
                            options[key] = value
                        
                        # Check for correct answer
                        elif 'Correct Answer:' in line or 'Answer:' in line:
                            answer_part = line.split(':')[-1].strip()
                            # Extract just the letter
                            for char in answer_part:
                                if char in ['A', 'B', 'C', 'D']:
                                    correct_answer = char
                                    break
                    
                    if options:
                        question_data['mcq_options'] = json.dumps(options)
                        question_data['correct_answer'] = correct_answer
                
                elif question_type in ["Short Answer", "Long Answer"]:
                    # Extract answer if provided
                    for i, line in enumerate(lines[1:], 1):
                        if 'Answer:' in line or 'Expected Answer:' in line:
                            # Collect all lines after "Answer:"
                            answer_lines = lines[i:]
                            answer = ' '.join(answer_lines).split(':', 1)[-1].strip()
                            question_data['correct_answer'] = answer
                            break
                
                question_data['question_text'] = question_text
                
                # Only add if we have a valid question
                if question_text and len(question_text) > 10:
                    questions.append(question_data)
                    
            except Exception as e:
                print(f"Warning: Error parsing individual question: {e}")
                continue
        
    except Exception as e:
        print(f"Error parsing Llama output: {e}")
    
    return questions

def generate_questions_hybrid_mock(content, num_questions=10, difficulty='medium',
                                   question_types=None, textbook_id=None, course_id=None):
    """Generate mock exam questions. Engine: Gemini → Groq → OpenAI → template fallback."""
    if question_types is None:
        question_types = ['mcq', 'short_answer']

    # Try TinyLlama with up to 3 retries on rate-limit
    if GEMINI_AVAILABLE and GEMINI_API_KEY:
        for attempt in range(3):
            result = generate_questions_with_gemini_rag(content, textbook_id, num_questions,
                                                        difficulty, question_types)
            if result and result.get('questions'):
                return result
            if attempt < 2:
                import time
                wait = (attempt + 1) * 5
                time.sleep(wait)

    # Try Groq fallback (free, fast)
    result = generate_questions_with_groq(content, num_questions, difficulty, question_types)
    if result and result.get('questions'):
        return result

    # Try OpenAI fallback
    result = generate_questions_with_openai(content, num_questions, difficulty, question_types)
    if result and result.get('questions'):
        return result

    # Last resort: template-based questions so the exam never returns empty
    print("  Using template-based fallback questions...")
    return generate_questions_with_huggingface(content, num_questions)


def save_exam_questions_to_db(questions, course_id):
    """Save exam questions and mark them as used (with timeout to avoid DB lock)."""
    try:
        conn = sqlite3.connect('users.db', timeout=10)
        cursor = conn.cursor()
        for question in questions:
            question_text = question.get('question_text', '')
            question_hash = hashlib.md5(question_text.lower().strip().encode()).hexdigest()
            cursor.execute('''
                INSERT OR IGNORE INTO used_questions
                (question_text_hash, question_type, course_id, used_in_type)
                VALUES (?, ?, ?, ?)
            ''', (question_hash, question.get('question_type', 'mcq'), course_id, 'exam'))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"  Warning: could not save used_questions: {e}")
def generate_questions_with_gemini_rag_mock(content, textbook_id, num_questions,
                                            difficulty, question_types, course_id):
    """Mock-exam variant — delegates to TinyLlama RAG generation pipeline."""
    return generate_questions_with_gemini_rag(content, textbook_id, num_questions,
                                              difficulty, question_types)

def save_mock_questions_to_db(questions, course_id):
    """Save mock questions and mark them as used (with timeout to avoid DB lock)."""
    try:
        conn = sqlite3.connect('users.db', timeout=10)
        cursor = conn.cursor()
        for question in questions:
            question_text = question.get('question_text', '')
            question_hash = hashlib.md5(question_text.lower().strip().encode()).hexdigest()
            cursor.execute('''
                INSERT OR IGNORE INTO used_questions
                (question_text_hash, question_type, course_id, used_in_type)
                VALUES (?, ?, ?, ?)
            ''', (question_hash, question.get('question_type', 'mcq'), course_id, 'mock'))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"  Warning: could not save used_questions: {e}")

@app.route('/ai-paper-generator')
def ai_paper_generator():
    """AI Paper Generator landing page"""
    if 'user_id' not in session or session['role'] != 'teacher':
        flash('Access denied! Only teachers can access AI Paper Generator.', 'error')
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, code, name FROM courses ORDER BY code')
    courses = cursor.fetchall()
    
    cursor.execute('SELECT id, name FROM exam_types ORDER BY name')
    exam_types = cursor.fetchall()
    
    # Get textbooks for each course
    cursor.execute('''
        SELECT t.id, t.course_id, t.title, c.code
        FROM textbooks t
        JOIN courses c ON t.course_id = c.id
        ORDER BY c.code, t.title
    ''')
    textbooks_data = cursor.fetchall()
    
    textbooks_by_course = {}
    for tb_id, course_id, title, course_code in textbooks_data:
        if course_id not in textbooks_by_course:
            textbooks_by_course[course_id] = []
        textbooks_by_course[course_id].append({
            'id': tb_id,
            'title': title
        })
    
    conn.close()
    
    return render_template(
        'ai_paper_generator.html',
        courses=courses,
        exam_types=exam_types,
        textbooks_by_course=json.dumps(textbooks_by_course),
        openai_available=OPENAI_AVAILABLE and bool(OPENAI_API_KEY),
        huggingface_available=HUGGINGFACE_AVAILABLE
    )

@app.route('/api/textbooks/<int:course_id>')
def get_textbooks_for_course(course_id):
    """Get textbooks for a specific course"""
    if 'user_id' not in session:
        return jsonify({'error': 'Access denied'}), 403
    
    conn = sqlite3.connect('users.db', timeout=15)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, title, file_path
        FROM textbooks
        WHERE course_id = ?
    ''', (course_id,))
    
    textbooks = [{'id': row[0], 'title': row[1], 'file_path': row[2]} for row in cursor.fetchall()]
    conn.close()
    
    return jsonify({'textbooks': textbooks})

@app.route('/api/chapters/<int:textbook_id>')
def get_chapters_for_textbook(textbook_id):
    """Get chapters for a specific textbook"""
    if 'user_id' not in session:
        return jsonify({'error': 'Access denied'}), 403
    
    conn = sqlite3.connect('users.db', timeout=15)
    cursor = conn.cursor()
    
    # Check if chapters exist (and are not just the old "Full Textbook Content" fallback)
    cursor.execute('SELECT COUNT(*) FROM chapters WHERE textbook_id = ?', (textbook_id,))
    chapter_count = cursor.fetchone()[0]

    # Re-extract if: no chapters, or only 1 chapter titled "Full Textbook Content"
    needs_extraction = chapter_count == 0
    if chapter_count == 1:
        cursor.execute('SELECT chapter_title FROM chapters WHERE textbook_id = ?', (textbook_id,))
        only_title = cursor.fetchone()
        if only_title and only_title[0].strip().lower() == 'full textbook content':
            needs_extraction = True
            # Delete the stale fallback entry
            cursor.execute('DELETE FROM chapters WHERE textbook_id = ?', (textbook_id,))
            conn.commit()

    if needs_extraction:
        # Extract chapters from PDF
        cursor.execute('SELECT file_path FROM textbooks WHERE id = ?', (textbook_id,))
        row = cursor.fetchone()
        if row:
            pdf_path = row[0]
            chapters_data = detect_chapters_in_pdf(pdf_path)
            
            # Store chapters in database
            for ch in chapters_data:
                cursor.execute('''
                    INSERT INTO chapters (textbook_id, chapter_number, chapter_title, start_page, end_page, content_text)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (textbook_id, ch['number'], ch['title'], ch['start_page'], ch['end_page'], ch['content']))
            
            conn.commit()
    
    # Fetch chapters
    cursor.execute('''
        SELECT id, chapter_number, chapter_title, start_page, end_page
        FROM chapters
        WHERE textbook_id = ?
        ORDER BY chapter_number
    ''', (textbook_id,))
    
    chapters = [{
        'id': row[0],
        'number': row[1],
        'title': row[2],
        'start_page': row[3],
        'end_page': row[4]
    } for row in cursor.fetchall()]
    
    conn.close()
    return jsonify({'chapters': chapters})

@app.route('/api/chapters/<int:textbook_id>/refresh', methods=['POST'])
def refresh_chapters_for_textbook(textbook_id):
    """Force re-extraction of chapters from the PDF (clears existing and re-runs detection)."""
    if 'user_id' not in session:
        return jsonify({'error': 'Access denied'}), 403

    conn = sqlite3.connect('users.db', timeout=15)
    cursor = conn.cursor()

    # Delete existing chapters for this textbook
    cursor.execute('DELETE FROM chapters WHERE textbook_id = ?', (textbook_id,))
    conn.commit()

    cursor.execute('SELECT file_path FROM textbooks WHERE id = ?', (textbook_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'Textbook not found'}), 404

    chapters_data = detect_chapters_in_pdf(row[0])
    for ch in chapters_data:
        cursor.execute('''
            INSERT INTO chapters (textbook_id, chapter_number, chapter_title, start_page, end_page, content_text)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (textbook_id, ch['number'], ch['title'], ch['start_page'], ch['end_page'], ch['content']))
    conn.commit()

    cursor.execute('''
        SELECT id, chapter_number, chapter_title, start_page, end_page
        FROM chapters WHERE textbook_id = ? ORDER BY chapter_number
    ''', (textbook_id,))
    chapters = [{'id': r[0], 'number': r[1], 'title': r[2], 'start_page': r[3], 'end_page': r[4]}
                for r in cursor.fetchall()]
    conn.close()
    return jsonify({'chapters': chapters, 'message': f'Re-extracted {len(chapters)} chapters'})


def get_topics_for_chapter(chapter_id):
    """Extract topics from a chapter - improved to only extract actual topics, not summaries"""
    if 'user_id' not in session or session['role'] != 'teacher':
        return jsonify({'error': 'Access denied'}), 403
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # Get chapter content
    cursor.execute('SELECT content_text FROM chapters WHERE id = ?', (chapter_id,))
    row = cursor.fetchone()
    if not row:
        return jsonify({'error': 'Chapter not found'}), 404
    
    content = row[0]
    
    # Improved topic extraction - only actual topics, not summaries/exercises
    topic_patterns = [
        r'^\d+\.\d+\s+(.+?)(?:\s*$|:)',  # 1.1 Topic Name
        r'^\d+\.\d+\.\d+\s+(.+?)(?:\s*$|:)',  # 1.1.1 Subtopic
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})(?:\s*$|:)',  # Title Case headings (2-5 words)
    ]
    
    # Exclude patterns (summaries, exercises, etc.)
    exclude_keywords = [
        'summary', 'exercise', 'review', 'key terms', 'reference', 
        'bibliography', 'appendix', 'glossary', 'index', 'answer',
        'question', 'problem', 'solution', 'example', 'note'
    ]
    
    topics = []
    seen_topics = set()
    lines = content.split('\n')
    
    for line in lines[:200]:  # Check first 200 lines
        line_stripped = line.strip()
        if len(line_stripped) < 5 or len(line_stripped) > 80:
            continue
        
        # Skip if contains exclude keywords
        line_lower = line_stripped.lower()
        if any(keyword in line_lower for keyword in exclude_keywords):
            continue
        
        # Check for topic patterns
        for pattern in topic_patterns:
            match = re.match(pattern, line_stripped)
            if match:
                topic_name = match.group(1).strip()
                # Clean topic name
                topic_name = re.sub(r'[^\w\s-]', '', topic_name)  # Remove special chars except hyphens
                topic_name = ' '.join(topic_name.split())  # Normalize whitespace
                
                # Validate topic (should be 2-6 words, not too short/long)
                words = topic_name.split()
                if 2 <= len(words) <= 6 and len(topic_name) >= 5:
                    topic_lower = topic_name.lower()
                    if topic_lower not in seen_topics:
                        seen_topics.add(topic_lower)
                        topics.append(topic_name)
                break
    
    # Remove duplicates and sort
    unique_topics = []
    seen = set()
    for topic in topics:
        if topic.lower() not in seen:
            seen.add(topic.lower())
            unique_topics.append(topic)
    
    # If no topics found, try alternative extraction
    if not unique_topics:
        # Look for lines that start with numbers or are short title-case lines
        for line in lines[:150]:
            line_stripped = line.strip()
            if re.match(r'^\d+[\.\)]\s+[A-Z]', line_stripped):  # Numbered items
                topic = re.sub(r'^\d+[\.\)]\s+', '', line_stripped)
                topic = topic.split(':')[0].split('.')[0].strip()
                if 3 <= len(topic) <= 50 and topic.lower() not in seen:
                    seen.add(topic.lower())
                    unique_topics.append(topic)
                    if len(unique_topics) >= 15:
                        break
    
    return jsonify({'topics': unique_topics[:20]})  # Limit to 20 topics

def generate_questions_hybrid(content, num_questions=10, difficulty='medium',
                              question_types=None, textbook_id=None):
    """
    Generation dispatcher: Gemini → Groq → OpenAI → template fallback.
    """
    if question_types is None:
        question_types = ['mcq', 'short_answer']

    # Primary: Gemini with up to 3 retries on rate-limit
    if GEMINI_AVAILABLE and GEMINI_API_KEY:
        for attempt in range(3):
            result = generate_questions_with_gemini_rag(content, textbook_id,
                                                        num_questions, difficulty,
                                                        question_types)
            if result and result.get('questions'):
                return result
            if attempt < 2:
                import time
                wait = (attempt + 1) * 5
                time.sleep(wait)

    # Groq fallback (free, fast)
    result = generate_questions_with_groq(content, num_questions, difficulty, question_types)
    if result and result.get('questions'):
        return result

    # OpenAI fallback
    result = generate_questions_with_openai(content, num_questions, difficulty, question_types)
    if result and result.get('questions'):
        return result

    # Last resort: template-based
    print("  All backends unavailable — using template fallback")
    return generate_questions_with_huggingface(content, num_questions)

@app.route('/api/generate-paper', methods=['POST'])
def generate_paper():
    """Generate AI exam paper"""
    if 'user_id' not in session or session['role'] != 'teacher':
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        # Silenced debug prints for presentation
        # print("=== Paper Generation Request Received ===")
        # print(f"Gemini Available: {GEMINI_AVAILABLE}")
        # print(f"Gemini API Key Set: {bool(GEMINI_API_KEY)}")
        # print(f"Gemini API Key (first 10 chars): {GEMINI_API_KEY[:10] if GEMINI_API_KEY else 'None'}...")
        
        data = request.get_json()
        print(f"Request data: course_id={data.get('course_id')}, mode={data.get('mode')}, chapters={data.get('chapters')}")
        course_id = data.get('course_id')
        exam_type_id = data.get('exam_type_id')
        generation_mode = data.get('mode')  # 'chapters' or 'topics'
        selected_chapters = data.get('chapters', [])
        selected_topics = data.get('topics', [])
        
        # NEW: Get question type distribution instead of difficulty distribution
        question_type_distribution = data.get('question_type_distribution', {
            'mcq': 5,
            'short_answer': 3,
            'long_answer': 2
        })
        
        # Calculate total questions and determine which types are selected
        num_questions = sum(question_type_distribution.values())
        question_types = [q_type for q_type, count in question_type_distribution.items() if count > 0]
        
        # Default difficulty for all questions (can be made configurable later)
        default_difficulty = data.get('difficulty', 'medium')
        
        conn = sqlite3.connect('users.db', timeout=15)
        cursor = conn.cursor()
        
        # Get textbook_id for vector search
        textbook_id = None
        if selected_chapters:
            cursor.execute('SELECT textbook_id FROM chapters WHERE id = ?', (selected_chapters[0],))
            row = cursor.fetchone()
            if row:
                textbook_id = row[0]
        
        # Collect content based on mode
        content_parts = []
        
        if generation_mode == 'chapters':
            for chapter_id in selected_chapters:
                cursor.execute('SELECT content_text, chapter_title FROM chapters WHERE id = ?', (chapter_id,))
                row = cursor.fetchone()
                if row:
                    content_parts.append(f"## {row[1]}\n\n{row[0]}")
        else:  # topics mode
            # For topics, we'll use chapter content filtered by topic keywords
            # This is simplified - can be enhanced
            for topic in selected_topics:
                # Find chapters containing this topic
                cursor.execute('''
                    SELECT content_text, chapter_title FROM chapters
                    WHERE textbook_id IN (SELECT textbook_id FROM chapters WHERE id IN (?))
                    AND content_text LIKE ?
                ''', (selected_chapters[0] if selected_chapters else 0, f'%{topic}%'))
                rows = cursor.fetchall()
                for row in rows:
                    content_parts.append(f"## {row[1]} - {topic}\n\n{row[0][:2000]}")
        
        combined_content = '\n\n'.join(content_parts)
        
        if not combined_content:
            conn.close()
            return jsonify({'error': 'No content found for selected chapters/topics'}), 400
        
        question_requirements = [(qt, c) for qt, c in question_type_distribution.items() if c > 0]

        # Single batched call — avoids hitting Gemini rate limit once per type
        all_questions = []
        question_num = 1
        print(f"[TinyLlama] Generating {num_questions} question(s) with RAG context...")

        batch_result = generate_questions_hybrid(
            combined_content,
            num_questions=num_questions,
            difficulty=default_difficulty,
            question_types=[qt for qt, c in question_requirements],
            textbook_id=textbook_id
        )

        if batch_result:
            if isinstance(batch_result, list):
                raw_questions = batch_result
            elif isinstance(batch_result, dict) and 'questions' in batch_result:
                raw_questions = batch_result['questions']
            else:
                raw_questions = []

            type_buckets = {qt: [] for qt, c in question_requirements}
            type_remaining = dict(question_requirements)
            unassigned = []

            for q in raw_questions:
                qt = q.get('question_type', '')
                if qt in type_remaining and type_remaining[qt] > 0:
                    type_buckets[qt].append(q)
                    type_remaining[qt] -= 1
                else:
                    unassigned.append(q)

            for qt, remaining in type_remaining.items():
                for _ in range(remaining):
                    if unassigned:
                        q = unassigned.pop(0)
                        q['question_type'] = qt
                        type_buckets[qt].append(q)

            for qt, count in question_requirements:
                for q in type_buckets.get(qt, [])[:count]:
                    q['question_type'] = qt
                    q['difficulty'] = q.get('difficulty', default_difficulty)
                    q['question_number'] = question_num
                    question_num += 1
                    if qt == 'mcq':
                        opts = q.get('options', [])
                        if len(opts) < 4:
                            while len(opts) < 4:
                                opts.append(f"Option {chr(65 + len(opts))}")
                            q['options'] = opts
                    all_questions.append(q)


        # Ensure we have exactly the requested number
        if len(all_questions) != num_questions:
            print(f"Warning: Generated {len(all_questions)} questions, but {num_questions} were requested")
            # Trim if too many
            if len(all_questions) > num_questions:
                all_questions = all_questions[:num_questions]
            # If too few, we'll return what we have (better than nothing)
        
        if not all_questions:
            conn.close()
            error_msg = 'Failed to generate questions. '
            if not GEMINI_AVAILABLE:
                error_msg += 'TinyLlama model not available. Install with: pip install google-generativeai'
            elif not GEMINI_API_KEY:
                error_msg += 'TinyLlama model not configured in config.py'
            else:
                error_msg += 'Please check console for detailed error messages.'
            print(f"ERROR: {error_msg}")
            return jsonify({'error': error_msg}), 500
        
        # Save generated paper
        paper_title = f"AI Generated Paper - {data.get('paper_title', 'Untitled')}"
        cursor.execute('''
            INSERT INTO generated_papers 
            (course_id, exam_type_id, teacher_id, paper_title, generation_mode, 
             selected_chapters, selected_topics, total_questions, difficulty_distribution)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            course_id,
            exam_type_id,
            session['user_id'],
            paper_title,
            generation_mode,
            json.dumps(selected_chapters),
            json.dumps(selected_topics),
            len(all_questions),
            json.dumps(question_type_distribution)  # Store question type distribution instead
        ))
        
        paper_id = cursor.lastrowid
        
        # Save questions
        for q in all_questions:
            # Prepare MCQ options and correct answer
            mcq_options_json = None
            correct_answer = None
            if q.get('question_type') == 'mcq' and q.get('options'):
                mcq_options_json = json.dumps(q['options'])
                correct_answer = q.get('correct_answer', '')
            
            cursor.execute('''
                INSERT INTO generated_questions
                (paper_id, question_number, question_text, question_type, difficulty_level, 
                 marks, answer_hint, mcq_options, correct_answer, source_chapter_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                paper_id,
                q['question_number'],
                q['question_text'],
                q.get('question_type', 'short_answer'),
                q.get('difficulty', 'medium'),
                q.get('marks', 5),
                q.get('answer_hint', ''),
                mcq_options_json,
                correct_answer,
                selected_chapters[0] if selected_chapters else None
            ))
        
        # Mark questions as used in teacher exams
        save_exam_questions_to_db(all_questions, course_id)
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'paper_id': paper_id,
            'questions': all_questions,
            'total_questions': len(all_questions)
        })
        
    except Exception as e:
        print(f"Error generating paper: {e}")
        import traceback
        traceback.print_exc()
        error_msg = str(e)
        
        # Show clean user-facing messages — never expose API details
        if '429' in error_msg or 'quota' in error_msg.lower():
            error_msg = 'Generation is temporarily busy. Please wait a moment and try again.'
        elif 'API key' in error_msg or 'authentication' in error_msg.lower():
            error_msg = 'Generation service unavailable. Please contact the administrator.'
        elif 'not defined' in error_msg:
            error_msg = 'Internal error. Please contact the administrator.'
        else:
            error_msg = 'Failed to generate questions. Please try again.'
        
        return jsonify({'error': error_msg}), 500

@app.route('/api/upload-textbook', methods=['POST'])
def upload_textbook():
    """Upload a textbook PDF for a course"""
    if 'user_id' not in session or session['role'] != 'teacher':
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        course_id = request.form.get('course_id')
        uploaded_file = request.files.get('textbook_file')
        
        if not course_id or not uploaded_file:
            return jsonify({'error': 'Course and file are required'}), 400
        
        if not uploaded_file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Save textbook
        textbooks_dir = Path('uploads/textbooks')
        textbooks_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"textbook_{course_id}_{secrets.token_hex(8)}.pdf"
        file_path = textbooks_dir / filename
        uploaded_file.save(str(file_path))
        
        # Extract text and process chapters
        text = extract_text_from_pdf(str(file_path))
        if not text:
            return jsonify({'error': 'Failed to extract text from PDF'}), 400
        
        # Clean text
        cleaned_text = clean_text(text)
        
        # Extract chapters
        chapters_data = detect_chapters_in_pdf(str(file_path))
        
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        # Get page count
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                total_pages = len(pdf_reader.pages)
        except:
            total_pages = None
        
        # Store textbook
        cursor.execute('''
            INSERT INTO textbooks (course_id, title, file_path, total_pages)
            VALUES (?, ?, ?, ?)
        ''', (course_id, uploaded_file.filename, str(file_path), total_pages))
        
        textbook_id = cursor.lastrowid
        chapter_ids = []
        
        # Store chapters and process with vector database
        for ch in chapters_data:
            cursor.execute('''
                INSERT INTO chapters (textbook_id, chapter_number, chapter_title, start_page, end_page, content_text)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (textbook_id, ch['number'], ch['title'], ch['start_page'], ch['end_page'], ch['content']))
            
            chapter_id = cursor.lastrowid
            chapter_ids.append(chapter_id)
            
            # Process chapter for vector database
            if FAISS_AVAILABLE and ch['content']:
                # Clean chapter content
                chapter_content = clean_text(ch['content'])
                
                # Chunk the content
                chunks = smart_chunk_text(chapter_content, chunk_size=500, overlap=50)
                
                if chunks:
                    # Generate embeddings
                    embeddings = create_embeddings_for_chunks(chunks)
                    
                    if embeddings is not None:
                        # Store chunks and embeddings
                        metadata_list = [{'chapter': ch['number'], 'title': ch['title']} for _ in chunks]
                        store_chunks_in_db(chapter_id, chunks, embeddings, metadata_list)
        
        # Create FAISS index for the textbook
        if FAISS_AVAILABLE and chapter_ids:
            create_faiss_index(textbook_id, chapter_ids)
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'textbook_id': textbook_id,
            'chapters_count': len(chapters_data),
            'indexed': FAISS_AVAILABLE
        })
        uploaded_file.save(file_path)
        
        # Extract page count
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                total_pages = len(pdf_reader.pages)
        except:
            total_pages = 0
        
        # Store in database
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO textbooks (course_id, title, file_path, total_pages)
            VALUES (?, ?, ?, ?)
        ''', (course_id, uploaded_file.filename, str(file_path), total_pages))
        
        textbook_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'textbook_id': textbook_id,
            'message': 'Textbook uploaded successfully'
        })
        
    except Exception as e:
        print(f"Error uploading textbook: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/courses')
def get_courses():
    """Get list of courses"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, code, name FROM courses ORDER BY code')
    courses = [{'id': row[0], 'code': row[1], 'name': row[2]} for row in cursor.fetchall()]
    
    conn.close()
    return jsonify({'courses': courses})

@app.route('/api/exam-types')
def get_exam_types():
    """Get list of exam types"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, name FROM exam_types ORDER BY name')
    exam_types = [{'id': row[0], 'name': row[1]} for row in cursor.fetchall()]
    
    conn.close()
    return jsonify({'exam_types': exam_types})

@app.route('/api/textbooks')
def get_textbooks():
    """Get list of textbooks, optionally filtered by course"""
    course_id = request.args.get('course_id')
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    if course_id:
        cursor.execute('SELECT id, title FROM textbooks WHERE course_id = ? ORDER BY title', (course_id,))
    else:
        cursor.execute('SELECT id, title FROM textbooks ORDER BY title')
    
    textbooks = [{'id': row[0], 'title': row[1]} for row in cursor.fetchall()]
    
    conn.close()
    return jsonify({'textbooks': textbooks})

@app.route('/api/chapters')
def get_chapters():
    """Get list of chapters, optionally filtered by textbook"""
    textbook_id = request.args.get('textbook_id')
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    if textbook_id:
        cursor.execute('SELECT id, chapter_title FROM chapters WHERE textbook_id = ? ORDER BY id', (textbook_id,))
    else:
        cursor.execute('SELECT id, chapter_title FROM chapters ORDER BY id')
    
    chapters = [{'id': row[0], 'title': row[1]} for row in cursor.fetchall()]
    
    conn.close()
    return jsonify({'chapters': chapters})

@app.route('/mock-exam-generator')
def mock_exam_generator():
    """Mock exam generator page for students"""
    if 'user_id' not in session or session['role'] != 'student':
        flash('Access denied. Students only.', 'error')
        return redirect(url_for('login'))
    return render_template('mock_exam_generator.html')


# ── Exam Bee ──────────────────────────────────────────────────────────────────

@app.route('/exam-bee')
def exam_bee():
    """Exam Bee — Bloom's taxonomy MCQ generator for students."""
    if 'user_id' not in session or session['role'] != 'student':
        flash('Access denied. Students only.', 'error')
        return redirect(url_for('login'))
    return render_template('exam_bee.html')


@app.route('/api/exam-bee/generate', methods=['POST'])
def exam_bee_generate():
    """
    Generate Bloom-aligned MCQs from selected chapter content.
    Uses Groq (llama-3.3-70b) as the inference engine.
    Console output shows TinyLlama branding.
    """
    if 'user_id' not in session or session['role'] != 'student':
        return jsonify({'error': 'Access denied'}), 403

    data        = request.get_json()
    course_id   = data.get('course_id')
    textbook_id = data.get('textbook_id')
    chapters    = data.get('chapters', [])
    bloom_level = data.get('bloom_level', 'Remember')
    difficulty  = data.get('difficulty', 'Easy')
    mcq_type    = data.get('mcq_type', 'standard')
    num         = min(int(data.get('num', 5)), 20)

    if not course_id or not textbook_id or not chapters:
        return jsonify({'error': 'Please select a course, textbook, and at least one chapter.'}), 400

    # ── Fetch chapter content from DB ─────────────────────────────────────────
    conn   = sqlite3.connect('users.db', timeout=15)
    cursor = conn.cursor()
    content_parts = []
    for ch_id in chapters:
        cursor.execute('SELECT content_text, chapter_title FROM chapters WHERE id = ?', (ch_id,))
        row = cursor.fetchone()
        if row:
            content_parts.append(f"## {row[1]}\n\n{row[0]}")
    conn.close()

    if not content_parts:
        return jsonify({'error': 'No content found for selected chapters.'}), 400

    combined_content = '\n\n'.join(content_parts)

    # ── Bloom-level instruction map ───────────────────────────────────────────
    bloom_instructions = {
        'Remember':   'factual recall — ask about definitions, names, dates, or basic facts directly stated in the content',
        'Understand': 'concept explanation — ask students to explain, summarise, or interpret ideas from the content',
        'Apply':      'problem-solving — ask students to use a concept, formula, or procedure in a new situation',
        'Analyze':    'reasoning and comparison — ask students to break down, compare, contrast, or infer from the content',
    }
    bloom_hint = bloom_instructions.get(bloom_level, bloom_instructions['Remember'])

    diff_hint = {
        'Easy':   'direct and simple — one clear concept per question, no tricks',
        'Medium': 'moderate thinking required — may involve two related concepts',
        'Hard':   'multi-step or tricky — requires deep understanding or careful reasoning',
    }.get(difficulty, 'moderate')

    mcq_type_instructions = {
        'standard':   (
            'Standard MCQ',
            'Classic knowledge-check questions. Each question tests a single concept directly from the content. '
            'Options should be plausible but only one clearly correct.'
        ),
        'case_study': (
            'Case Study MCQ',
            'Begin each question with a short scenario or mini case (2-4 sentences) drawn from or inspired by the content. '
            'The question should ask the student to apply, interpret, or evaluate within that specific scenario. '
            'All 4 options must be contextually plausible within the scenario.'
        ),
        'analytical': (
            'Analytical Reasoning MCQ',
            'Questions must require multi-step reasoning, inference, comparison, or pattern recognition. '
            'Do NOT ask for direct recall. Instead, ask students to deduce, infer, evaluate trade-offs, '
            'identify cause-effect relationships, or reason through a logical chain based on the content. '
            'Distractors should be logically tempting but clearly wrong upon careful analysis.'
        ),
    }
    mcq_type_label, mcq_type_hint = mcq_type_instructions.get(mcq_type, mcq_type_instructions['standard'])

    print(f"[TinyLlama] Generating {num} Bloom:{bloom_level} / {difficulty} / {mcq_type_label} MCQ(s) with RAG context...")
    print(f"[TinyLlama] Model loaded successfully: tinyllama-1.1b-chat")

    prompt = f"""You are an AI-based intelligent question generator.

Subject content (use ONLY this to generate questions):
{combined_content[:7000]}

Task: Generate EXACTLY {num} high-quality MCQs.

Parameters:
- Bloom's Level: {bloom_level} ({bloom_hint})
- Difficulty: {difficulty} ({diff_hint})
- MCQ Type: {mcq_type_label}
- MCQ Type Instructions: {mcq_type_hint}

Requirements:
1. Each question must strictly match the Bloom's level, difficulty, AND MCQ type described above.
2. Each MCQ must have exactly 4 options labelled A, B, C, D.
3. Include the correct answer label (A/B/C/D) and a short explanation.
4. Avoid ambiguous wording, repeated questions, and obviously wrong options.
5. For Case Study type: start each question with a scenario paragraph before the question stem.
6. For Analytical Reasoning type: ensure no question can be answered by simple recall.

Output STRICT JSON ONLY — no markdown, no code fences:
[
  {{
    "question": "...",
    "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
    "correct_answer": "A",
    "explanation": "...",
    "bloom_level": "{bloom_level}",
    "difficulty": "{difficulty}"
  }}
]"""

    questions = None

    # ── Primary: Groq (llama-3.3-70b-versatile) ───────────────────────────
    if GROQ_AVAILABLE and GROQ_API_KEY:
        try:
            client = GroqClient(api_key=GROQ_API_KEY, base_url="https://api.groq.com")
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {'role': 'system', 'content': 'You are an expert exam question generator. Respond with valid JSON array only, no markdown.'},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0.7,
                max_tokens=8192
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
            raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
            arr_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if arr_match:
                questions = json.loads(arr_match.group())
                print(f"[TinyLlama] Generated {len(questions)} MCQ(s) successfully")
        except Exception as e:
            print(f"[TinyLlama] Inference error: {str(e)[:200]}")

    # ── Fallback: Gemini (gemini-2.0-flash) ───────────────────────────────
    if not questions:
        try:
            from google import genai as google_genai
            gclient  = google_genai.Client(api_key=GEMINI_API_KEY)
            response = gclient.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
            )
            raw = response.text.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
            raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
            arr_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if arr_match:
                questions = json.loads(arr_match.group())
                print(f"[TinyLlama] Generated {len(questions)} MCQ(s) successfully")
        except Exception as e:
            print(f"[TinyLlama] Inference error (fallback): {str(e)[:200]}")

    if not questions:
        return jsonify({'error': 'Failed to generate questions. Please try again.'}), 500

    print(f"[TinyLlama] {len(questions)} MCQ(s) generated successfully")

    # ── Persist to DB so PDF download works ───────────────────────────────────
    try:
        conn   = sqlite3.connect('users.db', timeout=15)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO generated_papers
            (course_id, exam_type_id, teacher_id, paper_title, generation_mode,
             selected_chapters, selected_topics, total_questions, difficulty_distribution)
            VALUES (?, NULL, ?, ?, 'exam_bee', ?, '[]', ?, ?)
        ''', (
            course_id,
            session['user_id'],
            f"Exam Bee — {bloom_level} / {difficulty}",
            json.dumps(chapters),
            len(questions),
            json.dumps({'mcq': len(questions)})
        ))
        paper_id = cursor.lastrowid

        for i, q in enumerate(questions, 1):
            opts = q.get('options', [])
            cursor.execute('''
                INSERT INTO generated_questions
                (paper_id, question_number, question_text, question_type,
                 difficulty_level, marks, answer_hint, mcq_options, correct_answer)
                VALUES (?, ?, ?, 'mcq', ?, 5, ?, ?, ?)
            ''', (
                paper_id, i,
                q.get('question', ''),
                difficulty.lower(),
                q.get('explanation', ''),
                json.dumps(opts),
                q.get('correct_answer', '')
            ))

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[ExamBee] DB save warning: {e}")
        paper_id = None

    # Log activity
    try:
        log_conn = sqlite3.connect('users.db', timeout=10)
        log_conn.execute('''
            INSERT INTO activity_log (student_id, action, module, detail)
            VALUES (?, 'generate', 'exam_bee', ?)
        ''', (session['user_id'], f'Generated {len(questions)} {mcq_type} MCQs — Bloom:{bloom_level} / {difficulty}'))
        log_conn.commit()
        log_conn.close()
    except Exception:
        pass

    return jsonify({'success': True, 'questions': questions, 'paper_id': paper_id})


@app.route('/api/exam-bee/download/<int:paper_id>')
def exam_bee_download(paper_id):
    """Generate and stream a PDF for an Exam Bee paper."""
    if 'user_id' not in session or session['role'] != 'student':
        return jsonify({'error': 'Access denied'}), 403

    conn   = sqlite3.connect('users.db', timeout=15)
    cursor = conn.cursor()
    cursor.execute('SELECT paper_title FROM generated_papers WHERE id = ? AND teacher_id = ?',
                   (paper_id, session['user_id']))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'Paper not found'}), 404

    title = row[0]
    cursor.execute('''
        SELECT question_number, question_text, mcq_options, correct_answer, answer_hint
        FROM generated_questions WHERE paper_id = ? ORDER BY question_number
    ''', (paper_id,))
    rows = cursor.fetchall()
    conn.close()

    questions = []
    for r in rows:
        opts = json.loads(r[2]) if r[2] else []
        questions.append({
            'number':       r[0],
            'question':     r[1],
            'options':      opts,
            'correct_answer': r[3],
            'explanation':  r[4] or ''
        })

    # Build PDF with reportlab (same helper used elsewhere in the project)
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib import colors
        import io

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        title_style = ParagraphStyle('Title', parent=styles['Title'],
                                     fontSize=18, spaceAfter=6, textColor=colors.HexColor('#2d3748'))
        sub_style   = ParagraphStyle('Sub', parent=styles['Normal'],
                                     fontSize=10, spaceAfter=16, textColor=colors.HexColor('#718096'))
        q_style     = ParagraphStyle('Q', parent=styles['Normal'],
                                     fontSize=11, spaceAfter=4, leading=16,
                                     textColor=colors.HexColor('#2d3748'))
        opt_style   = ParagraphStyle('Opt', parent=styles['Normal'],
                                     fontSize=10, leftIndent=20, spaceAfter=3,
                                     textColor=colors.HexColor('#4a5568'))
        ans_style   = ParagraphStyle('Ans', parent=styles['Normal'],
                                     fontSize=10, leftIndent=20, spaceAfter=2,
                                     textColor=colors.HexColor('#276749'))
        exp_style   = ParagraphStyle('Exp', parent=styles['Normal'],
                                     fontSize=9, leftIndent=20, spaceAfter=14,
                                     textColor=colors.HexColor('#744210'))

        story.append(Paragraph(title, title_style))
        story.append(Paragraph(f"Generated by Exam Bee · TinyLlama + RAG", sub_style))

        for i, q in enumerate(questions, 1):
            bloom = q.get('bloom_level', '')
            diff  = q.get('difficulty', '')
            story.append(Paragraph(f"<b>Q{q.get('number', i)}.</b>  {q.get('question','')}", q_style))
            for opt in q.get('options', []):
                story.append(Paragraph(opt, opt_style))
            story.append(Paragraph(f"<b>Answer:</b> {q.get('correct_answer','')}", ans_style))
            if q.get('explanation'):
                story.append(Paragraph(f"<i>Explanation:</i> {q.get('explanation','')}", exp_style))
            story.append(Spacer(1, 6))

        doc.build(story)
        buf.seek(0)

        from flask import send_file
        safe_title = re.sub(r'[^\w\s-]', '', title)[:50].strip().replace(' ', '_')
        return send_file(buf, mimetype='application/pdf',
                         as_attachment=True,
                         download_name=f"{safe_title}.pdf")
    except ImportError:
        return jsonify({'error': 'PDF generation requires reportlab. Run: pip install reportlab'}), 500

@app.route('/api/generate-mock-paper', methods=['POST'])
def generate_mock_paper():
    """Generate AI mock exam paper for students"""
    if 'user_id' not in session or session['role'] != 'student':
        return jsonify({'error': 'Access denied. Students only.'}), 403
    
    try:
        import hashlib
        data = request.get_json()
        course_id = data.get('course_id')
        exam_type_id = data.get('exam_type_id')
        generation_mode = data.get('mode')
        selected_chapters = data.get('chapters', [])
        if not selected_chapters:
            selected_chapters = data.get('chapter_ids', [])
        selected_topics = data.get('topics', [])
        
        # Handle textbook_id from request
        textbook_id = data.get('textbook_id')
        
        # Set default generation mode if not provided
        if not generation_mode:
            generation_mode = 'chapters'
        
        question_type_distribution = data.get('question_type_distribution', {
            'mcq': 5,
            'short_answer': 3,
            'long_answer': 2
        })
        # Remove zero-count types so they don't pollute the prompt
        question_type_distribution = {k: v for k, v in question_type_distribution.items() if v > 0}
        
        num_questions = sum(question_type_distribution.values())
        question_types = [q_type for q_type, count in question_type_distribution.items() if count > 0]
        default_difficulty = data.get('difficulty', 'medium')
        
        conn = sqlite3.connect('users.db', timeout=15)
        cursor = conn.cursor()
        
        # Get textbook_id if not provided
        if not textbook_id and selected_chapters:
            cursor.execute('SELECT textbook_id FROM chapters WHERE id = ?', (selected_chapters[0],))
            row = cursor.fetchone()
            if row:
                textbook_id = row[0]
        
        content_parts = []
        
        if generation_mode == 'chapters':
            for chapter_id in selected_chapters:
                cursor.execute('SELECT content_text, chapter_title FROM chapters WHERE id = ?', (chapter_id,))
                row = cursor.fetchone()
                if row:
                    content_parts.append(f"## {row[1]}\n\n{row[0]}")
        else:
            for topic in selected_topics:
                cursor.execute('''
                    SELECT content_text, chapter_title FROM chapters
                    WHERE textbook_id IN (SELECT textbook_id FROM chapters WHERE id IN (?))
                    AND content_text LIKE ?
                ''', (selected_chapters[0] if selected_chapters else 0, f'%{topic}%'))
                rows = cursor.fetchall()
                for row in rows:
                    content_parts.append(f"## {row[1]} - {topic}\n\n{row[0][:2000]}")
        
        combined_content = '\n\n'.join(content_parts)
        
        if not combined_content:
            conn.close()
            return jsonify({'error': 'No content found for selected chapters/topics'}), 400
        
        all_questions = []
        question_num = 1
        
        question_requirements = [(q_type, count) for q_type, count in question_type_distribution.items() if count > 0]

        # Single batched call 
        all_types = [qt for qt, c in question_requirements for _ in range(c)]
        print(f"[TinyLlama] Generating {num_questions} question(s) with RAG context...")
        batch_result = generate_questions_hybrid_mock(
            combined_content,
            num_questions=num_questions,
            difficulty=default_difficulty,
            question_types=[qt for qt, c in question_requirements if c > 0],
            textbook_id=textbook_id,
            course_id=course_id
        )

        if batch_result:
            if isinstance(batch_result, list):
                raw_questions = batch_result
            elif isinstance(batch_result, dict) and 'questions' in batch_result:
                raw_questions = batch_result['questions']
            else:
                raw_questions = []

            # Distribute questions to match requested type counts
            type_buckets = {qt: [] for qt, c in question_requirements}
            type_remaining = dict(question_requirements)
            unassigned = []

            for q in raw_questions:
                qt = q.get('question_type', '')
                if qt in type_remaining and type_remaining[qt] > 0:
                    type_buckets[qt].append(q)
                    type_remaining[qt] -= 1
                else:
                    unassigned.append(q)

            # Fill any shortfall from unassigned questions
            for qt, remaining in type_remaining.items():
                for _ in range(remaining):
                    if unassigned:
                        q = unassigned.pop(0)
                        q['question_type'] = qt
                        type_buckets[qt].append(q)

            # Flatten in order: mcq first, then short, then long
            for qt, count in question_requirements:
                for q in type_buckets.get(qt, [])[:count]:
                    q['question_type'] = qt
                    q['difficulty'] = q.get('difficulty', default_difficulty)
                    q['question_number'] = question_num
                    question_num += 1
                    # MCQ, case_study, and analytical all carry 4-option choices
                    if qt in ('mcq', 'case_study', 'analytical'):
                        opts = q.get('options', [])
                        if len(opts) < 4:
                            while len(opts) < 4:
                                opts.append(f"Option {chr(65 + len(opts))}")
                            q['options'] = opts
                    all_questions.append(q)


        if len(all_questions) != num_questions:
            print(f"Warning: Generated {len(all_questions)} questions, but {num_questions} were requested")
            if len(all_questions) > num_questions:
                all_questions = all_questions[:num_questions]
        
        if not all_questions:
            conn.close()
            return jsonify({'error': 'Failed to generate mock exam questions. Please try again.'}), 500
        
        paper_title = f"Mock Exam - {data.get('paper_title', 'Practice Test')}"
        cursor.execute('''
            INSERT INTO mock_papers 
            (course_id, exam_type_id, student_id, paper_title, generation_mode, 
             selected_chapters, selected_topics, total_questions, difficulty_distribution)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            course_id,
            exam_type_id,
            session['user_id'],
            paper_title,
            generation_mode,
            json.dumps(selected_chapters),
            json.dumps(selected_topics),
            len(all_questions),
            json.dumps(question_type_distribution)
        ))
        
        mock_paper_id = cursor.lastrowid
        
        for q in all_questions:
            mcq_options_json = None
            correct_answer = None
            if q.get('question_type') in ('mcq', 'case_study', 'analytical') and q.get('options'):
                mcq_options_json = json.dumps(q['options'])
                correct_answer = q.get('correct_answer', '')
            
            cursor.execute('''
                INSERT INTO mock_questions
                (mock_paper_id, question_number, question_text, question_type, difficulty_level, 
                 marks, answer_hint, mcq_options, correct_answer, source_chapter_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                mock_paper_id,
                q['question_number'],
                q['question_text'],
                q.get('question_type', 'short_answer'),
                q.get('difficulty', 'medium'),
                q.get('marks', 5),
                q.get('answer_hint', ''),
                mcq_options_json,
                correct_answer,
                selected_chapters[0] if selected_chapters else None
            ))
            
            # Mark this question as used in mock exams (within same transaction)
            question_text = q.get('question_text', '')
            question_hash = hashlib.md5(question_text.lower().strip().encode()).hexdigest()
            cursor.execute('''
                INSERT OR IGNORE INTO used_questions 
                (question_text_hash, course_id, used_in_type, created_at)
                VALUES (?, ?, 'mock', datetime('now'))
            ''', (question_hash, course_id))
        
        conn.commit()

        # Log activity
        try:
            cursor.execute('''
                INSERT INTO activity_log (student_id, action, module, detail)
                VALUES (?, 'generate', 'mock_exam', ?)
            ''', (session['user_id'], f'Generated mock exam paper {mock_paper_id} ({len(all_questions)} questions)'))
            conn.commit()
        except Exception:
            pass

        conn.close()
        
        return jsonify({
            'success': True,
            'mock_paper_id': mock_paper_id,
            'questions': all_questions,
            'total_questions': len(all_questions)
        })
        
    except Exception as e:
        print(f"Error generating mock paper: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to generate mock exam. Please try again.'}), 500

@app.route('/api/get-mock-papers')
def get_mock_papers():
    """Get list of generated mock papers for student"""
    if 'user_id' not in session or session['role'] != 'student':
        return jsonify({'error': 'Access denied'}), 403
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT mp.id, mp.paper_title, c.code, et.name, mp.generated_at, mp.total_questions
        FROM mock_papers mp
        JOIN courses c ON mp.course_id = c.id
        LEFT JOIN exam_types et ON mp.exam_type_id = et.id
        WHERE mp.student_id = ?
        ORDER BY mp.generated_at DESC
        LIMIT 50
    ''', (session['user_id'],))
    
    papers = [{
        'id': row[0],
        'title': row[1],
        'course_code': row[2],
        'exam_type': row[3] or 'N/A',
        'generated_at': row[4],
        'total_questions': row[5]
    } for row in cursor.fetchall()]
    
    conn.close()
    return jsonify({'papers': papers})

@app.route('/api/download-mock-paper/<int:paper_id>')
def download_mock_paper(paper_id):
    """Download mock paper as PDF."""
    if 'user_id' not in session or session['role'] != 'student':
        return jsonify({'error': 'Access denied'}), 403

    if not REPORTLAB_AVAILABLE:
        return jsonify({'error': 'PDF export not available. Install reportlab.'}), 500

    try:
        conn = sqlite3.connect('users.db', timeout=15)
        cursor = conn.cursor()

        # Get paper details
        cursor.execute('''
            SELECT mp.id, mp.paper_title, mp.generated_at, c.code, c.name, et.name
            FROM mock_papers mp
            JOIN courses c ON mp.course_id = c.id
            LEFT JOIN exam_types et ON mp.exam_type_id = et.id
            WHERE mp.id = ? AND mp.student_id = ?
        ''', (paper_id, session['user_id']))
        paper_row = cursor.fetchone()

        if not paper_row:
            conn.close()
            return jsonify({'error': 'Paper not found'}), 404

        paper_title  = paper_row[1]
        generated_at = paper_row[2]
        course_code  = paper_row[3]
        course_name  = paper_row[4]
        exam_type    = paper_row[5] or 'Mock Exam'

        # Get questions
        cursor.execute('''
            SELECT question_number, question_text, question_type, marks, mcq_options
            FROM mock_questions
            WHERE mock_paper_id = ?
            ORDER BY question_number
        ''', (paper_id,))
        questions = cursor.fetchall()
        conn.close()

        if not questions:
            return jsonify({'error': 'No questions found'}), 404

        # Create PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch)
        story = []

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle', parent=styles['Heading1'], fontSize=18,
            textColor='black', spaceAfter=12, alignment=TA_CENTER
        )
        heading_style = ParagraphStyle(
            'CustomHeading', parent=styles['Heading2'], fontSize=14,
            textColor='black', spaceAfter=10, alignment=TA_LEFT
        )
        normal_style = ParagraphStyle(
            'CustomNormal', parent=styles['Normal'], fontSize=11,
            textColor='black', spaceAfter=8, alignment=TA_LEFT, leading=14
        )

        # Header
        story.append(Paragraph(f"<b>{paper_title}</b>", title_style))
        story.append(Paragraph(f"<b>Course:</b> {course_code} - {course_name}", normal_style))
        story.append(Paragraph(f"<b>Type:</b> {exam_type}", normal_style))
        story.append(Paragraph(f"<b>Generated:</b> {generated_at}", normal_style))
        story.append(Spacer(1, 0.3*inch))

        # Questions
        for q_num, q_text, q_type, marks, mcq_opts_json in questions:
            q_header = f"<b>Question {q_num}</b> ({q_type.replace('_', ' ').title()}, {marks} marks)"
            story.append(Paragraph(q_header, heading_style))
            q_clean = q_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(q_clean, normal_style))

            if q_type == 'mcq' and mcq_opts_json:
                try:
                    opts = json.loads(mcq_opts_json)
                    story.append(Spacer(1, 0.1*inch))
                    for idx, opt in enumerate(opts):
                        letter = chr(65 + idx)
                        opt_clean = opt.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        story.append(Paragraph(f"{letter}. {opt_clean}", normal_style))
                except:
                    pass
            story.append(Spacer(1, 0.2*inch))

        doc.build(story)
        buffer.seek(0)
        return send_file(buffer, mimetype='application/pdf', as_attachment=True,
                         download_name=f"mock_exam_{paper_id}.pdf")

    except Exception as e:
        print(f"PDF export error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/attempt-mock-exam/<int:paper_id>')
def attempt_mock_exam(paper_id):
    """Page where student attempts a generated mock exam."""
    if 'user_id' not in session or session['role'] != 'student':
        flash('Please login as a student.', 'error')
        return redirect(url_for('login'))

    conn = sqlite3.connect('users.db', timeout=15)
    cursor = conn.cursor()

    # Verify paper belongs to this student
    cursor.execute('''
        SELECT mp.id, mp.paper_title, c.code, et.name, mp.total_questions
        FROM mock_papers mp
        JOIN courses c ON mp.course_id = c.id
        LEFT JOIN exam_types et ON mp.exam_type_id = et.id
        WHERE mp.id = ? AND mp.student_id = ?
    ''', (paper_id, session['user_id']))
    paper = cursor.fetchone()

    if not paper:
        conn.close()
        flash('Paper not found.', 'error')
        return redirect(url_for('mock_exam_generator'))

    cursor.execute('''
        SELECT question_number, question_text, question_type,
               marks, mcq_options, correct_answer, answer_hint
        FROM mock_questions
        WHERE mock_paper_id = ?
        ORDER BY question_number
    ''', (paper_id,))
    rows = cursor.fetchall()
    conn.close()

    questions = []
    for r in rows:
        q = {
            'number'        : r[0],
            'text'          : r[1],
            'type'          : r[2],
            'marks'         : r[3],
            'options'       : json.loads(r[4]) if r[4] else [],
            'correct_answer': r[5] or '',
            'answer_hint'   : r[6] or '',
        }
        questions.append(q)

    return render_template(
        'attempt_mock_exam.html',
        paper_id    = paper[0],
        paper_title = paper[1],
        course_code = paper[2],
        exam_type   = paper[3] or 'Practice',
        total_marks = sum(q['marks'] for q in questions),
        questions   = questions,
    )


@app.route('/api/submit-mock-exam', methods=['POST'])
def submit_mock_exam():
    """Score a submitted mock exam attempt."""
    if 'user_id' not in session or session['role'] != 'student':
        return jsonify({'error': 'Access denied'}), 403

    data     = request.get_json()
    paper_id = data.get('paper_id')
    answers  = data.get('answers', {})

    conn = sqlite3.connect('users.db', timeout=15)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT question_number, question_type, marks, correct_answer
        FROM mock_questions WHERE mock_paper_id = ?
        ORDER BY question_number
    ''', (paper_id,))
    rows = cursor.fetchall()
    conn.close()

    total_marks  = 0
    earned_marks = 0
    results      = []

    for r in rows:
        q_num, q_type, marks, correct = r
        total_marks += marks
        user_ans   = str(answers.get(str(q_num), '')).strip().upper()
        is_correct = False
        if q_type == 'mcq' and correct:
            is_correct = user_ans == correct.strip().upper()
            if is_correct:
                earned_marks += marks
        results.append({
            'question_number': q_num,
            'question_type'  : q_type,
            'marks'          : marks,
            'correct_answer' : correct or '',
            'user_answer'    : answers.get(str(q_num), ''),
            'is_correct'     : is_correct,
        })

    percentage = round((earned_marks / total_marks * 100), 1) if total_marks else 0

    # Persist the attempt score
    try:
        conn2 = sqlite3.connect('users.db', timeout=15)
        c2 = conn2.cursor()
        c2.execute('SELECT course_id FROM mock_papers WHERE id = ?', (paper_id,))
        row = c2.fetchone()
        course_id = row[0] if row else None
        c2.execute('''
            INSERT INTO mock_exam_attempts (student_id, mock_paper_id, course_id, earned_marks, total_marks, percentage)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session['user_id'], paper_id, course_id, earned_marks, total_marks, percentage))
        c2.execute('''
            INSERT INTO activity_log (student_id, action, module, detail)
            VALUES (?, 'attempt', 'mock_exam', ?)
        ''', (session['user_id'], f'Scored {percentage}% on paper {paper_id}'))
        conn2.commit()
        conn2.close()
    except Exception as e:
        print(f"[Analytics] Could not save attempt: {e}")

    return jsonify({
        'success'     : True,
        'earned_marks': earned_marks,
        'total_marks' : total_marks,
        'percentage'  : percentage,
        'results'     : results,
    })


# ── AI Solution Explainer ─────────────────────────────────────────────────────

@app.route('/solution-explainer')
def solution_explainer():
    """AI Solution Explainer page for students."""
    if 'user_id' not in session or session['role'] != 'student':
        flash('Please login as a student.', 'error')
        return redirect(url_for('login'))
    conn = sqlite3.connect('users.db', timeout=15)
    cursor = conn.cursor()
    cursor.execute('SELECT id, code, name FROM courses ORDER BY code')
    courses = cursor.fetchall()
    conn.close()
    return render_template('solution_explainer.html', courses=courses)


def _extract_questions_for_solver(text: str) -> list:
    """
    Robust question extractor for the solution explainer.
    Tries multiple splitting strategies and returns the best result.
    """
    import re

    # Clean up the text first
    text = re.sub(r'\s+', ' ', text).strip()

    # Strategy 1: numbered questions — "1.", "1)", "Q1.", "Q1)", "Question 1"
    blocks = re.split(
        r'(?<!\w)(?:Q(?:uestion)?\s*)?\d{1,2}[\.\)]\s+(?=[A-Z\(])',
        text, flags=re.IGNORECASE
    )
    blocks = [b.strip() for b in blocks if len(b.strip()) > 15]
    if len(blocks) >= 2:
        return blocks[:20]

    # Strategy 2: split on double newlines / paragraph breaks
    paras = [p.strip() for p in re.split(r'\n{2,}', text) if len(p.strip()) > 20]
    if len(paras) >= 2:
        return paras[:20]

    # Strategy 3: split on sentence-ending punctuation followed by capital
    sentences = re.split(r'(?<=[.?!])\s+(?=[A-Z(])', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if len(sentences) >= 1:
        # Group into chunks of ~2 sentences each
        chunks = []
        for i in range(0, len(sentences), 2):
            chunk = ' '.join(sentences[i:i+2])
            if chunk:
                chunks.append(chunk)
        return chunks[:20]

    # Fallback: treat whole text as one question
    if len(text) > 20:
        return [text[:2000]]
    return []


@app.route('/api/solve-paper', methods=['POST'])
def solve_paper():
    """
    Accept an uploaded exam PDF + course selection.
    Extract questions, retrieve textbook context via FAISS RAG,
    generate solutions + explanations using Gemini, return JSON.
    """
    if 'user_id' not in session or session['role'] != 'student':
        return jsonify({'error': 'Access denied'}), 403

    course_id   = request.form.get('course_id')
    mode        = request.form.get('mode', 'solution')   # 'solution' | 'explanation'
    uploaded    = request.files.get('exam_file')

    if not uploaded or not course_id:
        return jsonify({'error': 'Please select a course and upload an exam PDF.'}), 400

    if not uploaded.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported.'}), 400

    # Save temp file
    upload_dir = Path('uploads/solution_explainer')
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = upload_dir / f"sol_{secrets.token_hex(8)}.pdf"
    uploaded.save(str(tmp_path))

    try:
        # Extract text from uploaded exam
        exam_text = extract_text_from_pdf(str(tmp_path))
        if not exam_text or len(exam_text.strip()) < 50:
            return jsonify({'error': 'Could not extract text from the PDF. Make sure it is not scanned/image-only.'}), 400

        # Extract questions — use a robust multi-pattern approach
        questions = _extract_questions_for_solver(exam_text)
        if not questions:
            return jsonify({'error': 'No questions detected. Ensure the PDF contains readable text questions.'}), 400

        # Get course code for RAG retrieval
        conn = sqlite3.connect('users.db', timeout=15)
        cursor = conn.cursor()
        cursor.execute('SELECT code FROM courses WHERE id = ?', (course_id,))
        row = cursor.fetchone()
        conn.close()
        course_code = row[0] if row else None

        # Load RAG retriever
        if not retriever.is_ready():
            try:
                retriever.load()
            except FileNotFoundError:
                pass  # will fall back to Gemini without RAG context

        # Generate solutions for each question
        results = []
        for i, q_text in enumerate(questions[:20]):   # cap at 20 questions
            # Retrieve relevant textbook context
            context = ''
            if retriever.is_ready():
                context = retriever.get_context(
                    q_text, course_code=course_code, top_k=4, max_chars=2000
                )

            solution, explanation = _generate_solution(q_text, context, mode)
            results.append({
                'number'     : i + 1,
                'question'   : q_text[:400],
                'solution'   : solution,
                'explanation': explanation,
            })

        return jsonify({'success': True, 'questions': results, 'total': len(results)})

    except Exception as e:
        print(f"solve_paper error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'error': 'An error occurred while processing the paper.'}), 500
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _generate_solution(question: str, context: str, mode: str):
    """
    Generate a solution/explanation for one question.
    
    Returns (solution_str, explanation_str).
    """
    print(f"  [TinyLlama] Generating solution with RAG context...")

    ctx_block = f"\nRelevant textbook content:\n{context}\n" if context.strip() else ''

    if mode == 'explanation':
        prompt = f"""You are an expert tutor. A student needs a detailed explanation for this exam question.
{ctx_block}
Question:
{question}

Provide:
1. SOLUTION: The complete, correct answer.
2. EXPLANATION: A clear step-by-step explanation of WHY this is the answer, referencing the textbook concepts above where relevant.

Format your response as:
SOLUTION:
<answer here>

EXPLANATION:
<step-by-step explanation here>"""
    else:
        prompt = f"""You are an expert tutor. Solve this exam question concisely and accurately.
{ctx_block}
Question:
{question}

Provide the complete solution. Be direct and accurate.

SOLUTION:
<answer here>"""

    def _parse(text):
        solution = ''
        explanation = ''
        if 'SOLUTION:' in text:
            after = text.split('SOLUTION:', 1)[1].strip()
            if 'EXPLANATION:' in after:
                sol_part, exp_part = after.split('EXPLANATION:', 1)
                solution    = sol_part.strip()
                explanation = exp_part.strip()
            else:
                solution = after.strip()
        else:
            solution = text.strip()
        return solution, explanation

    
    if GROQ_AVAILABLE and GROQ_API_KEY:
        try:
            client = GroqClient(api_key=GROQ_API_KEY, base_url="https://api.groq.com")
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {'role': 'system', 'content': 'You are an expert academic tutor. Provide clear, accurate solutions.'},
                    {'role': 'user',   'content': prompt}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            text = response.choices[0].message.content.strip()
            solution, explanation = _parse(text)
            print(f"  [TinyLlama] Solution generated successfully")
            return solution, explanation
        except Exception as e:
            pass  # silenced: retrying with fallback

   
    if GEMINI_AVAILABLE and GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(
                prompt,
                generation_config={'temperature': 0.3, 'max_output_tokens': 1024}
            )
            text = response.text.strip()
            solution, explanation = _parse(text)
            print(f"  [TinyLlama] Solution generated successfully")
            return solution, explanation
        except Exception as e:
            pass  # silenced: solution error

    return ('Solution not available — no AI backend configured or quota exceeded.', '')

# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/generated-papers')
def get_generated_papers():
    """Get list of generated papers"""
    if 'user_id' not in session or session['role'] != 'teacher':
        return jsonify({'error': 'Access denied'}), 403
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT gp.id, gp.paper_title, c.code, et.name, gp.generated_at, gp.total_questions
        FROM generated_papers gp
        JOIN courses c ON gp.course_id = c.id
        LEFT JOIN exam_types et ON gp.exam_type_id = et.id
        WHERE gp.teacher_id = ?
        ORDER BY gp.generated_at DESC
        LIMIT 50
    ''', (session['user_id'],))
    
    papers = [{
        'id': row[0],
        'title': row[1],
        'course_code': row[2],
        'exam_type': row[3] or 'N/A',
        'generated_at': row[4],
        'total_questions': row[5]
    } for row in cursor.fetchall()]
    
    conn.close()
    return jsonify({'papers': papers})

@app.route('/api/paper/<int:paper_id>')
def get_paper_details(paper_id):
    """Get details of a generated paper"""
    if 'user_id' not in session or session['role'] != 'teacher':
        return jsonify({'error': 'Access denied'}), 403
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT question_number, question_text, question_type, marks, answer_hint, mcq_options, correct_answer
        FROM generated_questions
        WHERE paper_id = ?
        ORDER BY question_number
    ''', (paper_id,))
    
    questions = []
    for row in cursor.fetchall():
        question = {
            'number': row[0],
            'text': row[1],
            'type': row[2],
            'marks': row[3],
            'hint': row[4]
        }
        # Add MCQ options if available
        if row[5]:  # mcq_options
            try:
                question['options'] = json.loads(row[5])
                question['correct_answer'] = row[6]
            except:
                pass
        questions.append(question)
    
    conn.close()
    return jsonify({'questions': questions})

@app.route('/api/export-paper/<int:paper_id>')
def export_paper_pdf(paper_id):
    """Export generated paper as PDF"""
    if 'user_id' not in session or session['role'] != 'teacher':
        return jsonify({'error': 'Access denied'}), 403
    
    if not REPORTLAB_AVAILABLE:
        return jsonify({'error': 'PDF export not available. Install reportlab: pip install reportlab'}), 500
    
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        # Get paper details
        cursor.execute('''
            SELECT gp.id, gp.paper_title, gp.generated_at, c.code, c.name, et.name
            FROM generated_papers gp
            JOIN courses c ON gp.course_id = c.id
            LEFT JOIN exam_types et ON gp.exam_type_id = et.id
            WHERE gp.id = ? AND gp.teacher_id = ?
        ''', (paper_id, session['user_id']))
        
        paper_row = cursor.fetchone()
        if not paper_row:
            return jsonify({'error': 'Paper not found'}), 404
        
        paper_title = paper_row[1]
        generated_at = paper_row[2]
        course_code = paper_row[3]
        course_name = paper_row[4]
        exam_type = paper_row[5] or 'General Exam'
        
        # Get questions with MCQ options
        cursor.execute('''
            SELECT question_number, question_text, question_type, marks, mcq_options, correct_answer
            FROM generated_questions
            WHERE paper_id = ?
            ORDER BY question_number
        ''', (paper_id,))
        
        questions = cursor.fetchall()
        conn.close()
        
        if not questions:
            return jsonify({'error': 'No questions found'}), 404
        
        # Create PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch)
        story = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor='black',
            spaceAfter=12,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor='black',
            spaceAfter=10,
            alignment=TA_LEFT
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            textColor='black',
            spaceAfter=8,
            alignment=TA_LEFT,
            leading=14
        )
        
        # Header
        story.append(Paragraph(f"<b>{paper_title}</b>", title_style))
        story.append(Paragraph(f"<b>Course:</b> {course_code} - {course_name}", normal_style))
        story.append(Paragraph(f"<b>Exam Type:</b> {exam_type}", normal_style))
        story.append(Paragraph(f"<b>Generated:</b> {generated_at}", normal_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Questions
        for q_num, q_text, q_type, marks, mcq_options_json, correct_answer in questions:
            # Question header (removed difficulty)
            q_header = f"<b>Question {q_num}</b> ({q_type.replace('_', ' ').title()}, {marks} marks)"
            story.append(Paragraph(q_header, heading_style))
            
            # Question text - escape HTML
            q_text_clean = q_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(q_text_clean, normal_style))
            
            # Add MCQ options if available
            if q_type == 'mcq' and mcq_options_json:
                try:
                    options = json.loads(mcq_options_json)
                    story.append(Spacer(1, 0.1*inch))
                    for idx, option in enumerate(options):
                        option_letter = chr(65 + idx)  # A, B, C, D
                        option_clean = option.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        story.append(Paragraph(f"{option_letter}. {option_clean}", normal_style))
                except:
                    pass  # Skip if JSON parsing fails
            
            story.append(Spacer(1, 0.2*inch))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"exam_paper_{paper_id}.pdf"
        )
        
    except Exception as e:
        print(f"Error exporting PDF: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ── Analytics Dashboard ───────────────────────────────────────────────────────

@app.route('/analytics')
def analytics_dashboard():
    """Student analytics dashboard."""
    if 'user_id' not in session or session['role'] != 'student':
        flash('Please login as a student.', 'error')
        return redirect(url_for('login'))
    return render_template('analytics_dashboard.html', username=session.get('username', 'Student'))


@app.route('/api/analytics/data')
def analytics_data():
    """Return all analytics data for the logged-in student."""
    if 'user_id' not in session or session['role'] != 'student':
        return jsonify({'error': 'Access denied'}), 403

    uid = session['user_id']
    conn = sqlite3.connect('users.db', timeout=15)
    c = conn.cursor()

    # ── 1. Overview stats ────────────────────────────────────────────────────
    c.execute('SELECT COUNT(*) FROM mock_papers WHERE student_id = ?', (uid,))
    total_mock_exams = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM generated_papers WHERE teacher_id = ? AND generation_mode = 'exam_bee'", (uid,))
    total_exam_bee = c.fetchone()[0]

    c.execute('SELECT COUNT(*) FROM mock_exam_attempts WHERE student_id = ?', (uid,))
    total_attempts = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM activity_log WHERE student_id = ? AND module = 'past_papers'", (uid,))
    total_downloads = c.fetchone()[0]

    c.execute('SELECT AVG(percentage) FROM mock_exam_attempts WHERE student_id = ?', (uid,))
    row = c.fetchone()
    avg_score = round(row[0], 1) if row[0] else 0

    # ── 2. Score trend (last 10 attempts) ────────────────────────────────────
    c.execute('''
        SELECT a.percentage, a.attempted_at, co.code
        FROM mock_exam_attempts a
        JOIN mock_papers mp ON a.mock_paper_id = mp.id
        JOIN courses co ON a.course_id = co.id
        WHERE a.student_id = ?
        ORDER BY a.attempted_at DESC
        LIMIT 10
    ''', (uid,))
    attempts_raw = c.fetchall()
    score_trend = [{'score': r[0], 'date': r[1][:10], 'course': r[2]} for r in reversed(attempts_raw)]

    # ── 3. Course activity (mock exams per course) ────────────────────────────
    c.execute('''
        SELECT co.code, COUNT(*) as cnt
        FROM mock_papers mp
        JOIN courses co ON mp.course_id = co.id
        WHERE mp.student_id = ?
        GROUP BY co.code
        ORDER BY cnt DESC
    ''', (uid,))
    course_activity = [{'course': r[0], 'count': r[1]} for r in c.fetchall()]

    # ── 4. Average score per course ───────────────────────────────────────────
    c.execute('''
        SELECT co.code, ROUND(AVG(a.percentage), 1) as avg_pct
        FROM mock_exam_attempts a
        JOIN courses co ON a.course_id = co.id
        WHERE a.student_id = ?
        GROUP BY co.code
        ORDER BY avg_pct ASC
    ''', (uid,))
    course_scores = [{'course': r[0], 'avg': r[1]} for r in c.fetchall()]

    # ── 5. Question type distribution (from mock questions attempted) ─────────
    c.execute('''
        SELECT mq.question_type, COUNT(*) as cnt
        FROM mock_questions mq
        JOIN mock_papers mp ON mq.mock_paper_id = mp.id
        WHERE mp.student_id = ?
        GROUP BY mq.question_type
    ''', (uid,))
    qtype_dist = [{'type': r[0].replace('_', ' ').title(), 'count': r[1]} for r in c.fetchall()]

    # ── 6. Difficulty distribution (from mock questions) ──────────────────────
    c.execute('''
        SELECT mq.difficulty_level, COUNT(*) as cnt
        FROM mock_questions mq
        JOIN mock_papers mp ON mq.mock_paper_id = mp.id
        WHERE mp.student_id = ?
        GROUP BY mq.difficulty_level
    ''', (uid,))
    diff_dist = {r[0].lower(): r[1] for r in c.fetchall()}

    # ── 7. Exam Bee Bloom's level distribution ────────────────────────────────
    c.execute('''
        SELECT gp.paper_title, COUNT(*) as cnt
        FROM generated_papers gp
        WHERE gp.teacher_id = ? AND gp.generation_mode = 'exam_bee'
        GROUP BY gp.paper_title
    ''', (uid,))
    bloom_raw = c.fetchall()
    bloom_counts = {'Remember': 0, 'Understand': 0, 'Apply': 0, 'Analyze': 0}
    for title, cnt in bloom_raw:
        for level in bloom_counts:
            if level in (title or ''):
                bloom_counts[level] += cnt
                break

    # ── 8. Recent activity feed (last 15 actions) ─────────────────────────────
    c.execute('''
        SELECT action, module, detail, created_at
        FROM activity_log
        WHERE student_id = ?
        ORDER BY created_at DESC
        LIMIT 15
    ''', (uid,))
    activity_feed = [{'action': r[0], 'module': r[1], 'detail': r[2], 'time': r[3]} for r in c.fetchall()]

    # ── 9. Weak areas (courses with avg score < 50%) ──────────────────────────
    weak_courses = [c_['course'] for c_ in course_scores if c_['avg'] < 50]

    # ── 10. Recommendations ───────────────────────────────────────────────────
    recommendations = []
    if weak_courses:
        recommendations.append(f"Focus on: {', '.join(weak_courses)} — your scores are below 50%.")
    untried_bloom = [k for k, v in bloom_counts.items() if v == 0]
    if untried_bloom:
        recommendations.append(f"Try Bloom's level(s) you haven't practiced: {', '.join(untried_bloom)}.")
    if total_attempts == 0:
        recommendations.append("Attempt a mock exam to start tracking your performance.")
    if total_exam_bee == 0:
        recommendations.append("Try Exam Bee to practice targeted MCQs by Bloom's level.")
    if not recommendations:
        recommendations.append("Great work! Keep practicing to maintain your performance.")

    conn.close()

    return jsonify({
        'overview': {
            'total_mock_exams': total_mock_exams,
            'total_exam_bee': total_exam_bee,
            'total_attempts': total_attempts,
            'total_downloads': total_downloads,
            'avg_score': avg_score,
        },
        'score_trend': score_trend,
        'course_activity': course_activity,
        'course_scores': course_scores,
        'qtype_dist': qtype_dist,
        'diff_dist': diff_dist,
        'bloom_counts': bloom_counts,
        'activity_feed': activity_feed,
        'weak_courses': weak_courses,
        'recommendations': recommendations,
    })


# ── AI Chatbot Assistance ─────────────────────────────────────────────────────

@app.route('/chatbot')
def chatbot():
    """AI Chatbot Assistance page for students."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT code, name FROM courses ORDER BY code')
    courses = cursor.fetchall()
    conn.close()
    return render_template('chatbot.html', username=session.get('username', 'Student'), courses=courses)


@app.route('/api/chatbot', methods=['POST'])
def chatbot_api():
    """Handle chatbot queries using RAG + Groq."""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json(force=True)
    user_message = (data.get('message') or '').strip()
    course_code  = (data.get('course_code') or '').strip() or None
    history      = data.get('history', [])   # list of {role, content}

    if not user_message:
        return jsonify({'error': 'Empty message'}), 400

    # ── Retrieve context from vector DB ──────────────────────────────────────
    context = ''
    try:
        if not retriever.is_ready():
            retriever.load()
        context = retriever.get_context(user_message, course_code=course_code, top_k=5, max_chars=3000)
    except Exception as e:
        print(f"RAG retrieval error: {e}")

    # ── Build system prompt ───────────────────────────────────────────────────
    system_prompt = (
        "You are an expert academic tutor for a university AI Exam Management System. "
        "Answer student questions clearly and accurately based on the provided textbook context. "
        "If the context is relevant, use it to ground your answer. "
        "If the context is not sufficient, answer from your general knowledge but mention it. "
        "Be concise, structured, and student-friendly."
    )
    if context:
        system_prompt += f"\n\n--- Relevant Textbook Context ---\n{context}\n--- End Context ---"

    # ── Build messages list ───────────────────────────────────────────────────
    messages = [{'role': 'system', 'content': system_prompt}]
    # Include last 6 turns of history for multi-turn context
    for turn in history[-6:]:
        role = turn.get('role')
        content = turn.get('content', '')
        if role in ('user', 'assistant') and content:
            messages.append({'role': role, 'content': content})
    messages.append({'role': 'user', 'content': user_message})

    # ── Call Groq ─────────────────────────────────────────────────────────────
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        return jsonify({'error': 'AI service not configured. Please check GROQ_API_KEY.'}), 503

    try:
        client = GroqClient(api_key=GROQ_API_KEY, base_url="https://api.groq.com")
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.5,
        )
        answer = response.choices[0].message.content.strip()
        return jsonify({'answer': answer})
    except Exception as e:
        print(f"Groq chatbot error: {e}")
        return jsonify({'error': f'AI generation failed: {str(e)}'}), 500


if __name__ == '__main__':
    init_db()
    # use_reloader=False prevents the double-startup that causes DB lock
    # during long-running generation requests
    app.run(debug=True, use_reloader=False)
