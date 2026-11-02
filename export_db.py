import sqlite3
from werkzeug.security import generate_password_hash

def export_database():
    """Export SQLite database to SQL format"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # Get all users
    cursor.execute('SELECT * FROM users')
    users = cursor.fetchall()
    
    # Create SQL dump
    sql_dump = """-- AI Exam Management System Database Export
-- Generated SQL for MySQL Workbench
-- Contains all current user data from users.db

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
        user_id, username, email, password, role, is_verified, verification_token, created_at = user
        
        # Escape single quotes in strings
        username_escaped = username.replace("'", "''")
        email_escaped = email.replace("'", "''")
        password_escaped = password.replace("'", "''")
        token_escaped = verification_token.replace("'", "''") if verification_token else 'NULL'
        
        sql_dump += f"INSERT INTO users (username, email, password, role, is_verified, verification_token) VALUES ('{username_escaped}', '{email_escaped}', '{password_escaped}', '{role}', {is_verified}, {f\"'{token_escaped}'\" if verification_token else 'NULL'});\n"
    
    conn.close()
    
    # Save to file
    with open('database_export_complete.sql', 'w', encoding='utf-8') as f:
        f.write(sql_dump)
    
    print(f"Database exported successfully!")
    print(f"Found {len(users)} users in the database")
    
    # Also print the SQL content
    print("\nSQL Content:")
    print(sql_dump)
    
    return sql_dump

if __name__ == "__main__":
    export_database()
