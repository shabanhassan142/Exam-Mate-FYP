"""
Script to scan textbooks folder and populate database with textbook information.
Expected folder structure:
    textbooks/
        CS101/
            textbook1.pdf
            textbook2.pdf
        MATH102/
            textbook.pdf
        ...
"""
import os
import sqlite3
import shutil
from pathlib import Path
import PyPDF2

# Configure your source directory containing textbooks
SOURCE_ROOT = Path('textbooks')  # e.g., textbooks/CS101/textbook.pdf

# Destination root inside the app folder where files will be stored
DEST_ROOT = Path('uploads/textbooks')

def ensure_course(conn, code: str) -> int:
    """Ensure course exists, return course_id"""
    cur = conn.cursor()
    cur.execute('SELECT id FROM courses WHERE code = ?', (code,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute('INSERT INTO courses (code, name) VALUES (?, ?)', (code, code))
    conn.commit()
    return cur.lastrowid

def get_pdf_page_count(file_path):
    """Get number of pages in PDF"""
    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            return len(pdf_reader.pages)
    except Exception:
        return 0

def ingest_textbooks():
    """Scan textbooks folder and populate database"""
    if not SOURCE_ROOT.exists():
        print(f"ERROR: Source directory not found: {SOURCE_ROOT}")
        print(f"Please create the folder structure: {SOURCE_ROOT}/COURSE_CODE/textbook.pdf")
        return
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # Ensure tables exist
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
    
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    total_skipped = 0
    
    # Scan each course folder
    for course_folder in SOURCE_ROOT.iterdir():
        if not course_folder.is_dir():
            continue
        
        course_code = course_folder.name.upper()
        print(f"\nProcessing course: {course_code}")
        
        # Ensure course exists in database
        course_id = ensure_course(conn, course_code)
        
        # Process PDF files in course folder
        pdf_files = list(course_folder.glob('*.pdf'))
        if not pdf_files:
            print(f"  No PDF files found in {course_code}/")
            continue
        
        for pdf_file in pdf_files:
            filename = pdf_file.name
            
            # Check if already exists (by filename)
            cursor.execute('''
                SELECT id FROM textbooks 
                WHERE course_id = ? AND title = ?
            ''', (course_id, filename))
            
            if cursor.fetchone():
                print(f"  Skipping (already exists): {filename}")
                total_skipped += 1
                continue
            
            # Copy file to uploads/textbooks
            dest_filename = f"{course_code}_{filename}"
            dest_path = DEST_ROOT / dest_filename
            
            try:
                shutil.copy2(pdf_file, dest_path)
                print(f"  Copying: {filename} -> {dest_path}")
            except Exception as e:
                print(f"  ERROR copying {filename}: {e}")
                continue
            
            # Get page count
            page_count = get_pdf_page_count(dest_path)
            
            # Insert into database
            try:
                cursor.execute('''
                    INSERT INTO textbooks (course_id, title, file_path, total_pages)
                    VALUES (?, ?, ?, ?)
                ''', (course_id, filename, str(dest_path), page_count))
                
                textbook_id = cursor.lastrowid
                print(f"  ✓ Added textbook (ID: {textbook_id}, Pages: {page_count})")
                total_processed += 1
            except Exception as e:
                print(f"  ERROR inserting {filename}: {e}")
                # Remove copied file on error
                if dest_path.exists():
                    dest_path.unlink()
    
    conn.commit()
    conn.close()
    
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Processed: {total_processed}")
    print(f"  Skipped: {total_skipped}")
    print(f"{'='*50}")

if __name__ == '__main__':
    print("="*50)
    print("Textbook Ingestion Script")
    print("="*50)
    print(f"Source: {SOURCE_ROOT.absolute()}")
    print(f"Destination: {DEST_ROOT.absolute()}")
    print()
    
    ingest_textbooks()
    
    print("\nDone! You can now use the AI Paper Generator in the teacher portal.")


