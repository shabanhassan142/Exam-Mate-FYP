import os
import mimetypes
import sqlite3
import shutil
from pathlib import Path

# Configure your source directory containing past papers hierarchy
SOURCE_ROOT = Path('C:\\Users\\Pc planet\\Music\\past_papers') # e.g., past_papers/CS101/sessional1/file.pdf

# Destination root inside the app folder where files will be stored
DEST_ROOT = Path('uploads/past_papers')

# Map folder names to standardized exam types
EXAM_TYPE_MAPPING = {
    'sessional1': 'sessional1',
    'sessional2': 'sessional2', 
    'sessional 2': 'sessional2',
    'final': 'final',
    'final paper': 'final',
    'final exam': 'final'
}

ALLOWED_EXAM_TYPES = set(EXAM_TYPE_MAPPING.keys())


def ensure_course(conn, code: str) -> int:
    cur = conn.cursor()
    cur.execute('SELECT id FROM courses WHERE code = ?', (code,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute('INSERT INTO courses (code, name) VALUES (?, ?)', (code, code))
    conn.commit()
    return cur.lastrowid


def get_exam_type_id(conn, name: str) -> int:
    cur = conn.cursor()
    cur.execute('SELECT id FROM exam_types WHERE name = ?', (name,))
    row = cur.fetchone()
    if row:
        return row[0]
    # Fallback: create if missing
    cur.execute('INSERT INTO exam_types (name) VALUES (?)', (name,))
    conn.commit()
    return cur.lastrowid


def insert_paper(conn, course_id: int, exam_type_id: int, title: str, src_path: Path, dst_path: Path):
    cur = conn.cursor()
    
    # Check if this paper already exists to prevent duplicates
    cur.execute('''
        SELECT id FROM past_papers 
        WHERE course_id = ? AND exam_type_id = ? AND title = ?
    ''', (course_id, exam_type_id, title))
    
    if cur.fetchone():
        print(f"⚠️  Skipping duplicate: {title} (already exists)")
        return
    
    mime_type, _ = mimetypes.guess_type(str(dst_path))
    size_bytes = dst_path.stat().st_size if dst_path.exists() else 0

    cur.execute('''
        INSERT INTO past_papers (course_id, exam_type_id, title, original_filename, storage_path, mime_type, size_bytes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        course_id,
        exam_type_id,
        title,
        src_path.name,
        str(dst_path.as_posix()),
        mime_type or 'application/octet-stream',
        size_bytes
    ))
    conn.commit()
    print(f"✅ Added new paper: {title}")


def ingest():
    conn = sqlite3.connect('users.db')
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    if not SOURCE_ROOT.exists():
        print(f"Source directory not found: {SOURCE_ROOT}")
        return

    # Iterate: past_papers/<course>/<exam_type>/<file>
    for course_dir in SOURCE_ROOT.iterdir():
        if not course_dir.is_dir():
            continue
        course_code = course_dir.name
        course_id = ensure_course(conn, course_code)

        for exam_type_dir in course_dir.iterdir():
            if not exam_type_dir.is_dir():
                continue
            exam_type = exam_type_dir.name.lower()
            print(f"Processing exam type: '{exam_type}' from folder: '{exam_type_dir.name}'")
            if exam_type not in ALLOWED_EXAM_TYPES:
                print(f"Skipping unknown exam type: '{exam_type}' (not in {ALLOWED_EXAM_TYPES})")
                continue
            
            # Map to standardized exam type
            standardized_type = EXAM_TYPE_MAPPING[exam_type]
            print(f"Mapped '{exam_type}' -> '{standardized_type}'")
            exam_type_id = get_exam_type_id(conn, standardized_type)

            for file_path in exam_type_dir.iterdir():
                if not file_path.is_file():
                    continue

                # Destination path: uploads/past_papers/<course>/<exam_type>/<filename>
                dst_dir = DEST_ROOT / course_code / exam_type
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_file = dst_dir / file_path.name

                # Check if file already exists in destination
                if dst_file.exists():
                    print(f"📁 File already exists: {dst_file}")
                else:
                    # Copy file only if it doesn't exist
                    shutil.copy2(file_path, dst_file)
                    print(f"📁 Copied file: {file_path} -> {dst_file}")

                # Insert metadata (will skip if duplicate)
                title = file_path.stem
                insert_paper(conn, course_id, exam_type_id, title, file_path, dst_file)

    conn.close()
    print("Ingestion complete.")


if __name__ == '__main__':
    ingest()


