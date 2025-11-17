"""
Utility script to add texts to the database programmatically.
Can be used for datagen integration or bulk text import.
"""

import sqlite3
import json
import sys
from pathlib import Path

DB_PATH = "texts.db"

def add_text(text, language="ar", is_rtl=True):
    """Add a new text to the database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO texts (text, language, is_rtl) VALUES (?, ?, ?)', 
              (text, language, is_rtl))
    conn.commit()
    text_id = c.lastrowid
    conn.close()
    return text_id

def add_texts_from_file(file_path, language="ar", is_rtl=True):
    """Add texts from a JSON file or text file"""
    file_path = Path(file_path)
    
    if file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        text = item.get('text', '')
                        lang = item.get('language', language)
                        rtl = item.get('is_rtl', is_rtl)
                        if text:
                            add_text(text, lang, rtl)
                    elif isinstance(item, str):
                        add_text(item, language, is_rtl)
            elif isinstance(data, dict):
                text = data.get('text', '')
                if text:
                    lang = data.get('language', language)
                    rtl = data.get('is_rtl', is_rtl)
                    add_text(text, lang, rtl)
    else:
        # Plain text file, one text per line
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    add_text(line, language, is_rtl)

def add_texts_from_list(texts, language="ar", is_rtl=True):
    """Add texts from a Python list"""
    for text in texts:
        if isinstance(text, dict):
            text_content = text.get('text', '')
            lang = text.get('language', language)
            rtl = text.get('is_rtl', is_rtl)
            if text_content:
                add_text(text_content, lang, rtl)
        elif isinstance(text, str):
            add_text(text, language, is_rtl)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        language = sys.argv[2] if len(sys.argv) > 2 else "ar"
        is_rtl = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else True
        add_texts_from_file(file_path, language, is_rtl)
        print(f"Texts added from {file_path}")
    else:
        # Example usage
        sample_texts = [
            "مرحبا بك في تطبيق النسخ الصوتي",
            "هذا تطبيق لتدريب النطق والنسخ",
            "يمكنك تسجيل صوتك ورفعه للنسخ",
        ]
        add_texts_from_list(sample_texts)
        print("Sample texts added to database")

