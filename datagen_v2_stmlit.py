"""
RTL Language Transcription App (Streamlit) — with CLI fallback & tests
- If Streamlit is unavailable, this file runs in CLI mode and executes self-tests.
- When Streamlit is installed, it launches the full web app.

Install (typical):
  pip install streamlit audio-recorder-streamlit requests

Optional (Linux):
  sudo apt-get install -y portaudio19-dev && pip install pyaudio  # only if you need local mic capture helpers
"""

import os
import random
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple, List
import hashlib
import pyotp
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import qrcode
from io import BytesIO
import re


# --- Persistent DB via S3 ---
DB_PATH = "/tmp/texts.db"
PROGRESS_DB_PATH = "/tmp/user_progress_v2.db"

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "us-west-1")

# Use the same bucket for DB persistence
S3_DB_BUCKET = AWS_BUCKET_NAME

# Store DBs under a "db/" prefix in your bucket
S3_DB_PREFIX = "db"


print("CWD =", os.getcwd())
print("Looking for progress DB at:", os.path.abspath("user_progress.db"))
print("Directory content:", os.listdir(os.getcwd()))


def download_db_from_s3(s3_key, local_path):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )
    try:
        s3.download_file(S3_DB_BUCKET, s3_key, local_path)
        print(f"Downloaded {s3_key} from S3")
    except Exception as e:
        print(f"No existing {s3_key} in S3 — starting empty DB.")


# Download both DBs before initializing
download_db_from_s3(f"{S3_DB_PREFIX}/texts.db", DB_PATH)
download_db_from_s3(f"{S3_DB_PREFIX}/user_progress_v2.db", PROGRESS_DB_PATH)


def upload_db_to_s3(local_path, s3_key):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )
    s3.upload_file(local_path, S3_DB_BUCKET, s3_key)



def upload_bytes_to_s3(bytes_data: bytes, s3_key: str) -> str:
    """Uploads raw bytes directly to S3 without saving locally."""
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION,
        )

        s3.put_object(
            Bucket=AWS_BUCKET_NAME,
            Key=s3_key,
            Body=bytes_data,
            ContentType="audio/wav"
        )

        return f"s3://{AWS_BUCKET_NAME}/{s3_key}"

    except (BotoCoreError, ClientError) as e:
        print("S3 upload failed:", e)
        return None


import threading

def upload_async(audio_bytes, audio_key, text_key, text):
    """
    Performs audio + transcript upload in a separate background thread
    so the UI does NOT freeze.
    """
    def _upload():
        upload_bytes_to_s3(audio_bytes, audio_key)
        upload_bytes_to_s3(text.encode("utf-8"), text_key)

    threading.Thread(target=_upload, daemon=True).start()



# --- Optional imports (Streamlit UI & mic recorder) ---
HAS_STREAMLIT = True
try:
    import streamlit as st  # type: ignore
except Exception:
    HAS_STREAMLIT = False

try:
    from audio_recorder_streamlit import audio_recorder  # type: ignore
except Exception:
    # If missing, we'll gracefully disable recording widget in UI mode
    audio_recorder = None  # type: ignore

import requests
import sqlite3

def get_db_connection():
    conn = sqlite3.connect(PROGRESS_DB_PATH)
    return conn


# Create table if it does not exist
conn = get_db_connection()
conn.execute("""
    CREATE TABLE IF NOT EXISTS progress (
        username TEXT PRIMARY KEY,
        step INTEGER,
        audio BLOB,
        completed INTEGER DEFAULT 0
    )
""")

# Per-project progress (username + project)
conn.execute("""
    CREATE TABLE IF NOT EXISTS project_progress (
        username TEXT,
        project INTEGER,
        step INTEGER,
        completed INTEGER DEFAULT 0,
        PRIMARY KEY (username, project)
    )
""")

# Migration for older DBs that don't have 'completed' yet
try:
    conn.execute("ALTER TABLE progress ADD COLUMN completed INTEGER DEFAULT 0")
except sqlite3.OperationalError:
    # Column already exists
    pass

conn.commit()
conn.close()



# ---------------------------
# Database setup & helpers
# ---------------------------
def init_db(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # Add is_admin column if it doesn't exist (for existing databases)

    try:
        c.execute("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # ✅ Add chosen_language column if missing (migration)    
    try:
        c.execute("ALTER TABLE users ADD COLUMN chosen_language TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Add source_file column if missing (to track uploaded file origin)
    try:
        c.execute("ALTER TABLE texts ADD COLUMN source_file TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists
    # ✅ Add MFA secret column if missing
    
    try:
        c.execute("ALTER TABLE users ADD COLUMN mfa_secret TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # ✅ Add last_mfa timestamp column for MFA cooldown
    try:
        c.execute("ALTER TABLE users ADD COLUMN last_mfa TIMESTAMP")
    except sqlite3.OperationalError:
        pass  # already exists

    # Add project column if missing (default Project 1)
    try:
        c.execute("ALTER TABLE texts ADD COLUMN project INTEGER DEFAULT 1")
    except sqlite3.OperationalError:
        pass  # column already exists

        # ✅ Add user_id column to recordings (who actually recorded)
    try:
        c.execute("ALTER TABLE recordings ADD COLUMN user_id INTEGER")
    except sqlite3.OperationalError:
        pass  # column already exists


    try:
        c.execute("ALTER TABLE recordings ADD COLUMN duration_seconds REAL")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists
    


    c.execute(
        """
        CREATE TABLE IF NOT EXISTS texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompts TEXT NOT NULL,
            language TEXT NOT NULL,
            is_rtl BOOLEAN DEFAULT 1,
            user_id INTEGER,
            source_file TEXT,
            project INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
    )



    # Migration: Check existing columns and migrate if needed
    try:
        c.execute("PRAGMA table_info(texts)")
        columns = {col[1]: col for col in c.fetchall()}
        
        # Add user_id column if it doesn't exist
        if 'user_id' not in columns:
            try:
                c.execute("ALTER TABLE texts ADD COLUMN user_id INTEGER")
                conn.commit()
            except sqlite3.OperationalError:
                pass
        
        # Rename 'text' column to 'prompts' if 'text' exists and 'prompts' doesn't
        if 'text' in columns and 'prompts' not in columns:
            try:
                # SQLite 3.25.0+ supports RENAME COLUMN
                c.execute("ALTER TABLE texts RENAME COLUMN text TO prompts")
                conn.commit()
            except sqlite3.OperationalError:
                # Fallback for older SQLite: create new table, copy data, drop old, rename
                try:
                    c.execute("""
                        CREATE TABLE texts_new (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            prompts TEXT NOT NULL,
                            language TEXT NOT NULL,
                            is_rtl BOOLEAN DEFAULT 1,
                            user_id INTEGER,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    c.execute("""
                        INSERT INTO texts_new (id, prompts, language, is_rtl, user_id, created_at)
                        SELECT id, text, language, is_rtl, user_id, created_at FROM texts
                    """)
                    c.execute("DROP TABLE texts")
                    c.execute("ALTER TABLE texts_new RENAME TO texts")
                    conn.commit()
                except sqlite3.OperationalError:
                    pass  # Migration failed, will use prompts column directly
    except sqlite3.OperationalError:
        pass  # Table doesn't exist yet, will be created with prompts column
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text_id INTEGER,
            audio_file_path TEXT,
            hoppepper_job_id TEXT,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (text_id) REFERENCES texts (id)
        )
        """
    )
    conn.commit()
    conn.close()

def backfill_recording_users():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT id, audio_file_path FROM recordings WHERE user_id IS NULL")
    rows = c.fetchall()

    for rid, path in rows:
        if path and path.startswith("s3://"):
            clean = path.replace(f"s3://{AWS_BUCKET_NAME}/", "")
            parts = clean.split("/")
            if len(parts) >= 3:
                username = parts[1]
                c.execute("SELECT id FROM users WHERE username=?", (username,))
                row = c.fetchone()
                if row:
                    c.execute(
                        "UPDATE recordings SET user_id=? WHERE id=?",
                        (row[0], rid)
                    )

    conn.commit()
    conn.close()
    upload_db_to_s3(DB_PATH, f"{S3_DB_PREFIX}/texts.db")


def get_all_users(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, username, is_admin, chosen_language FROM users ORDER BY username ASC")
    users = c.fetchall()
    conn.close()
    return users


def set_user_admin(user_id: int, make_admin: bool, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("UPDATE users SET is_admin = ? WHERE id = ?", (1 if make_admin else 0, user_id))
    conn.commit()
    conn.close()


def reset_user_mfa(user_id: int, db_path=DB_PATH):
    """Admin reset: clears MFA so user must re-enroll on next login."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("UPDATE users SET mfa_secret = NULL WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()



def get_all_texts(user_id: Optional[int] = None, db_path: str = DB_PATH) -> list:
    """
    Returns rows as:
      (id, prompts, language, is_rtl, user_id, project)
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    if user_id:
        c.execute("SELECT id, prompts, language, is_rtl, user_id, project FROM texts WHERE user_id = ?", (user_id,))
    else:
        c.execute("SELECT id, prompts, language, is_rtl, user_id, project FROM texts")
    texts = c.fetchall()
    conn.close()
    return texts



def get_text_by_id(text_id: int, db_path: str = DB_PATH):
    """
    Returns:
      (id, prompts, language, is_rtl, user_id, project)
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, prompts, language, is_rtl, user_id, project FROM texts WHERE id = ?", (text_id,))
    text = c.fetchone()
    conn.close()
    return text


def add_text(text: str, language: str = "ar", is_rtl: bool = True,
             user_id: Optional[int] = None, source_file: Optional[str] = None,
             project: int = 1, db_path: str = DB_PATH) -> int:
    """
    Insert a script into texts with a project number (default = 1).
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
        INSERT INTO texts (prompts, language, is_rtl, user_id, source_file, project)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (text, language, is_rtl, user_id, source_file, project))

    conn.commit()
    text_id = c.lastrowid
    conn.close()

    # Persist DB to S3
    upload_db_to_s3(DB_PATH, f"{S3_DB_PREFIX}/texts.db")

    return text_id




def create_user(username: str, password: str, db_path: str = DB_PATH) -> Optional[int]:
    """Create a new user. Returns user_id if successful, None if username already exists."""
    import hashlib
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    try:
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password_hash))
        conn.commit()
        user_id = c.lastrowid
        conn.close()

        # 🚀 NEW — persist user DB to S3
        upload_db_to_s3(DB_PATH, f"{S3_DB_PREFIX}/texts.db")

        return user_id

    except sqlite3.IntegrityError:
        conn.close()
        return None




def get_user_language(user_id, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT chosen_language FROM users WHERE id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row and row[0] else None

def save_user_language(user_id, language, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("UPDATE users SET chosen_language=? WHERE id=?", (language, user_id))
    conn.commit()
    conn.close()

    # 🚀 NEW: persist updated DB to S3
    upload_db_to_s3(DB_PATH, f"{S3_DB_PREFIX}/texts.db")

def ensure_user_dirs(user_id):
    base = f"recordings/user_{user_id}"
    audio_dir = f"{base}/audio"
    txt_dir = f"{base}/transcripts"
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    return audio_dir, txt_dir


def authenticate_user(username: str, password: str, db_path: str = DB_PATH) -> Tuple[Optional[int], bool]:
    """Authenticate a user. Returns (user_id, is_admin) tuple if successful, (None, False) otherwise."""
    import hashlib
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    # Check for admin user first
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

    
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        # Check if admin user exists, create if not
        c.execute("SELECT id, is_admin FROM users WHERE username = ?", (ADMIN_USERNAME,))
        result = c.fetchone()
        if result:
            user_id, is_admin = result
            # Ensure admin flag is set
            if not is_admin:
                c.execute("UPDATE users SET is_admin = 1 WHERE id = ?", (user_id,))
                conn.commit()
        else:
            # Create admin user
            admin_hash = hashlib.sha256(ADMIN_PASSWORD.encode()).hexdigest()
            c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", 
                     (ADMIN_USERNAME, admin_hash, 1))
            conn.commit()
            user_id = c.lastrowid
        conn.close()
        return (user_id, True)
    
    # Regular user authentication
    c.execute("SELECT id, is_admin FROM users WHERE username = ? AND password = ?", (username, password_hash))
    result = c.fetchone()
    conn.close()
    if result:
        return (result[0], bool(result[1]))
    return (None, False)

def get_user_mfa_secret(user_id, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT mfa_secret FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row and row[0] else None


def set_user_mfa_secret(user_id, secret, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("UPDATE users SET mfa_secret = ? WHERE id = ?", (secret, user_id))
    conn.commit()
    conn.close()

# -------------------------
# MFA Cooldown Helpers
# -------------------------
def update_mfa_timestamp(user_id, db_path=DB_PATH):
    """Store timestamp of last successful MFA verification."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("UPDATE users SET last_mfa=CURRENT_TIMESTAMP WHERE id=?", (user_id,))
    conn.commit()
    conn.close()


def needs_mfa(user_id, hours=24, db_path=DB_PATH):
    """Return True if MFA is required again."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT last_mfa FROM users WHERE id=?", (user_id,))
    row = c.fetchone()
    conn.close()

    if not row or not row[0]:
        return True  # never MFA’d before

    try:
        last = datetime.fromisoformat(row[0])
    except Exception:
        return True

    diff = datetime.now() - last
    return diff.total_seconds() > hours * 3600



def load_progress(user):
    conn = get_db_connection()
    try:
        # New schema: step, audio, completed
        row = conn.execute(
            "SELECT step, audio, COALESCE(completed, 0) FROM progress WHERE username=?",
            (user,)
        ).fetchone()
    except sqlite3.OperationalError:
        # Fallback for very old DB without 'completed'
        row = conn.execute(
            "SELECT step, audio FROM progress WHERE username=?",
            (user,)
        ).fetchone()
        row = row + (0,) if row else None  # add completed=0
    conn.close()

    if row:
        saved_step, audio_blob, completed = row
        st.session_state["current_text_index"] = saved_step
        st.session_state["audio_bytes"] = audio_blob
        st.session_state["user_completed"] = bool(completed)

    else:
        st.session_state["current_text_index"] = 0
        st.session_state["audio_bytes"] = None
        st.session_state["user_completed"] = False



def save_progress(user):
    conn = get_db_connection()
    step = st.session_state.get("current_text_index", 0)
    audio = st.session_state.get("audio_bytes")
    completed_flag = 1 if st.session_state.get("user_completed", False) else 0

    conn.execute(
        "REPLACE INTO progress (username, step, audio, completed) VALUES (?, ?, ?, ?)",
        (user, step, audio, completed_flag)
    )
    conn.commit()
    conn.close()

    # Also sync per-project progress if we know the current project
    project = st.session_state.get("current_project")
    if project is not None:
        save_project_progress(user, project, step, bool(completed_flag))

    # Upload DB to S3 (save_progress was doing this already; we keep it)
    upload_db_to_s3(PROGRESS_DB_PATH, f"{S3_DB_PREFIX}/user_progress_v2.db")



def get_project_progress(username: str, project: int) -> Tuple[int, bool]:
    """
    Returns (step, completed) for a given username + project.
    If no row exists yet, returns (0, False).
    """
    conn = get_db_connection()
    row = conn.execute(
        "SELECT step, completed FROM project_progress WHERE username=? AND project=?",
        (username, project)
    ).fetchone()
    conn.close()

    if row:
        step, completed = row
        return step, bool(completed)
    return 0, False


def save_project_progress(username: str, project: int, step: int, completed: bool) -> None:
    """
    Upserts per-project progress, and syncs to S3.
    """
    conn = get_db_connection()
    conn.execute(
        """
        REPLACE INTO project_progress (username, project, step, completed)
        VALUES (?, ?, ?, ?)
        """,
        (username, project, step, 1 if completed else 0)
    )
    conn.commit()
    conn.close()

    # Persist progress DB to S3
    upload_db_to_s3(PROGRESS_DB_PATH, f"{S3_DB_PREFIX}/user_progress_v2.db")


def project_is_completed(username: str, project: int) -> bool:
    """
    True if this project is fully completed for this user.
    """
    conn = get_db_connection()
    row = conn.execute(
        "SELECT completed FROM project_progress WHERE username=? AND project=?",
        (username, project)
    ).fetchone()
    conn.close()
    return bool(row and row[0] == 1)


    

# --------------------------------------
# NEW: Audio SHA256 and Duplicate Helper
# --------------------------------------

def get_audio_sha256(audio_bytes: bytes) -> str:
    """Returns the SHA256 hex digest of a bytes blob (audio)."""
    return hashlib.sha256(audio_bytes).hexdigest()

def recording_exists_for_text(text_id: int, audio_bytes: bytes, db_path: str = DB_PATH) -> bool:
    """Checks if the given audio (via hash) already exists in this text's recordings."""
    audio_hash = get_audio_sha256(audio_bytes)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT audio_file_path FROM recordings WHERE text_id = ?", (text_id,))
    for (audio_path,) in c.fetchall():
        if os.path.exists(audio_path):
            try:
                with open(audio_path, "rb") as f:
                    if get_audio_sha256(f.read()) == audio_hash:
                        conn.close()
                        return True
            except Exception:
                continue
    conn.close()
    return False

# --- End of Section 1 ---



# ----------------------
# Recording DB functions
# ----------------------
def generate_presigned_url(s3_uri: str, expires=300):
    # s3_uri is like: s3://bucket/key
    bucket = AWS_BUCKET_NAME

    key = s3_uri.replace(f"s3://{bucket}/", "")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )

    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires
    )


def get_s3_file_size(s3_uri: str) -> int:
    """Return file size (bytes) for an S3 object."""
    bucket = AWS_BUCKET_NAME
    key = s3_uri.replace(f"s3://{bucket}/", "")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )

    resp = s3.head_object(Bucket=bucket, Key=key)
    return resp["ContentLength"]


def estimate_wav_duration_seconds(size_bytes: int) -> float:
    """
    Estimate WAV duration using PCM 16-bit mono 16 kHz assumption.
    If you use a different sample rate, tell me and I'll adjust it.
    """
    bytes_per_second = 16000 * 2  # sample_rate * bytes_per_sample
    return size_bytes / bytes_per_second


def save_recording(
    text_id: int,
    audio_file_path: str,
    status: str = "saved",
    db_path: str = DB_PATH,
) -> int:

    duration_seconds = None
    try:
        if audio_file_path.startswith("s3://"):
            size = get_s3_file_size(audio_file_path)
            duration_seconds = estimate_wav_duration_seconds(size)
    except Exception:
        pass

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    user_id = st.session_state.get("user_id")

    try:
        c.execute(
            """
            INSERT INTO recordings
                (text_id, audio_file_path, hoppepper_job_id, status, duration_seconds, user_id)
            VALUES (?, ?, NULL, ?, ?, ?)
            """,
            (text_id, audio_file_path, status, duration_seconds, user_id)
        )
    except sqlite3.OperationalError:
        # Backward compatibility
        c.execute(
            """
            INSERT INTO recordings (text_id, audio_file_path, hoppepper_job_id, status)
            VALUES (?, ?, NULL, ?)
            """,
            (text_id, audio_file_path, status),
        )

    conn.commit()
    rid = c.lastrowid
    conn.close()

    upload_db_to_s3(DB_PATH, f"{S3_DB_PREFIX}/texts.db")
    return rid


def update_recording_status(recording_id: int, status: str, db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("UPDATE recordings SET status = ? WHERE id = ?", (status, recording_id))
    conn.commit()
    conn.close()

def get_recordings_by_text(text_id: int, db_path: str = DB_PATH) -> list:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        "SELECT id, audio_file_path, hoppepper_job_id, status, created_at FROM recordings WHERE text_id = ? ORDER BY created_at DESC",
        (text_id,),
    )
    rows = c.fetchall()
    conn.close()
    return rows

def get_all_recordings_by_user(user_id: Optional[int] = None, db_path: str = DB_PATH) -> list:
    """Return recordings belonging to a specific user, based on the audio file path."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    if user_id is None:
        # Admin sees ALL recordings
        c.execute("""
            SELECT
                r.id,
                r.audio_file_path,
                r.hoppepper_job_id,
                r.status,
                r.created_at,
                t.prompts,
                t.id AS text_id,
                u.username,
                r.duration_seconds
            FROM recordings r
            LEFT JOIN users u ON r.user_id = u.id   -- ✅ FIX
            LEFT JOIN texts t ON r.text_id = t.id
            ORDER BY r.created_at DESC
        """)

    else:
        # Regular users — now filter by username folder, not user_id
        username = get_username_from_user_id(user_id)

        pattern = f"%/{username}/audio/%"

        c.execute("""
            SELECT
                r.id,
                r.audio_file_path,
                r.hoppepper_job_id,
                r.status,
                r.created_at,
                t.prompts,
                t.id AS text_id,
                u.username,
                r.duration_seconds
            FROM recordings r
            JOIN texts t ON r.text_id = t.id
            LEFT JOIN users u ON t.user_id = u.id
            WHERE r.audio_file_path LIKE ?
            ORDER BY r.created_at DESC
        """, (pattern,))




    rows = c.fetchall()
    conn.close()
    return rows

def get_username_from_user_id(user_id, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


# ---------------------------
# Streamlit UI 
# ---------------------------

def run_streamlit_app() -> None:
    st.set_page_config(page_title="datagen_v2", layout="wide")

    st.session_state.setdefault("final_extra_recording_allowed", True)
    st.session_state.setdefault("final_extra_recording_done", False)


    # Ensure DB is ready
    init_db()

    # Session defaults
    for key, val in {
        "text_ids": [],
        "current_text_index": 0,
        "audio_bytes": None,
        "user_id": None,
        "username": None,
        "authenticated": False,
        "is_admin": False,
        "mfa_stage": None,              # "enroll" or "verify"
        "pending_mfa_secret": None,
        "pending_mfa_user_id": None,
        "pending_mfa_username": None,
    }.items():
        st.session_state.setdefault(key, val)


    # Authentication UI
    # Authentication UI
    if not st.session_state.get("authenticated", False):

        # --- Login Title ---
        st.markdown(
            "<h1 style='text-align: center;'>Datagen_v2</h1>",
            unsafe_allow_html=True
        )
        st.markdown("---")

        # --- Tabs ---
        tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

        # --------------------------
        # SIGN IN TAB
        # --------------------------
        with tab1:
            col1, col2, col3 = st.columns([4, 2, 4])
            with col2:
                with st.form("signin_form"):
                    username = st.text_input("Username", key="signin_username")
                    password = st.text_input("Password", type="password", key="signin_password")
                    submitted = st.form_submit_button("Sign In")

                if submitted:
                    user_id, is_admin = authenticate_user(username, password)
                    if user_id and is_admin:
                        # -------- ADMIN: Check if MFA is needed --------
                        if not needs_mfa(user_id):
                            # MFA NOT required — auto-login
                            st.session_state["user_id"] = user_id
                            st.session_state["username"] = username
                            st.session_state["authenticated"] = True
                            st.session_state["is_admin"] = True
                            st.success("Signed in — MFA not required this time.")
                            st.rerun()

                        # -------- ADMIN: require TOTP MFA --------
                        # Save temp admin identity
                        st.session_state["pending_mfa_user_id"] = user_id
                        st.session_state["pending_mfa_username"] = username

                        # Check if admin already has an MFA secret
                        existing_secret = get_user_mfa_secret(user_id)

                        if existing_secret:
                            # Go straight to verify stage
                            st.session_state["pending_mfa_secret"] = existing_secret
                            st.session_state["mfa_stage"] = "verify"
                        else:
                            # First-time setup: generate secret and go to enroll stage
                            secret = pyotp.random_base32()
                            st.session_state["pending_mfa_secret"] = secret
                            st.session_state["mfa_stage"] = "enroll"

                        st.rerun()


                    elif user_id:
                        # Check whether user is admin
                        if is_admin:
                            # Admin must have MFA
                            st.session_state["pending_mfa_user_id"] = user_id
                            st.session_state["pending_mfa_username"] = username

                            existing_secret = get_user_mfa_secret(user_id)
                            if existing_secret:
                                st.session_state["pending_mfa_secret"] = existing_secret
                                st.session_state["mfa_stage"] = "verify"
                            else:
                                secret = pyotp.random_base32()
                                st.session_state["pending_mfa_secret"] = secret
                                st.session_state["mfa_stage"] = "enroll"

                            st.rerun()

                        # Normal user (non-admin)
                        st.session_state["user_id"] = user_id
                        st.session_state["username"] = username
                        st.session_state["authenticated"] = True
                        st.session_state["is_admin"] = False


                        # Load language + progress as you already do
                        user_lang = get_user_language(user_id)
                        st.session_state["chosen_language"] = user_lang
                        load_progress(username)
                        st.session_state["progress_loaded"] = True

                        # Load text list
                        if user_lang:
                            texts = get_all_texts(db_path=DB_PATH)
                            lang_texts = [t for t in texts if t[2] == user_lang]
                            st.session_state["text_ids"] = [t[0] for t in lang_texts]

                            idx = st.session_state.get("current_text_index", 0)
                            if idx < 0 or idx >= len(st.session_state["text_ids"]):
                                st.session_state["current_text_index"] = 0

                        st.success("Signed in successfully!")
                        st.rerun()

                    else:
                        st.error("Invalid username or password")
                    
        # --------------------------
        # SIGN UP TAB
        # --------------------------
        with tab2:
            col1, col2, col3 = st.columns([4, 2, 4])
            with col2:
                with st.form("signup_form"):
                    new_username = st.text_input("Username", key="signup_username")
                    new_password = st.text_input("Password", type="password", key="signup_password")
                    confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
                    submitted_signup = st.form_submit_button("Sign Up")

                if submitted_signup:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif not new_username or not new_password:
                        st.error("Username and password are required")
                    else:
                        user_id = create_user(new_username, new_password)
                        if user_id:
                            st.session_state["user_id"] = user_id
                            st.session_state["username"] = new_username
                            st.session_state["authenticated"] = True
                            st.session_state["is_admin"] = False

                            # 💥 CRITICAL: Mark new user as having no chosen language yet
                            st.session_state["chosen_language"] = None

                            st.success("Account created successfully!")
                            st.rerun()

                        else:
                            st.error("Username already exists")


    # --------------------------
    # ADMIN TOTP MFA SCREENS
    # --------------------------
    if st.session_state.get("mfa_stage") == "enroll":
        st.markdown("### Admin: Set up Two-Factor Authentication (TOTP)")

        secret = st.session_state["pending_mfa_secret"]
        admin_name = st.session_state.get("pending_mfa_username", "admin")

        # TOTP URI for Google Authenticator
        totp = pyotp.TOTP(secret)
        uri = totp.provisioning_uri(
            name=admin_name,
            issuer_name="deen-mamadou-yacoubou"
        )

        # QR Code MFA Setup

        totp = pyotp.TOTP(secret)
        uri = totp.provisioning_uri(
            name=admin_name,
            issuer_name="datagen_v2"
        )

        # Generate QR code
        qr = qrcode.make(uri)
        buf = BytesIO()
        qr.save(buf, format="PNG")

        st.markdown("Scan this QR code with Google Authenticator:")
        st.image(buf.getvalue(), width=250)

        st.markdown("After scanning the QR code, enter the 6-digit code below.")


        code_input = st.text_input("Enter the 6-digit code from your Authenticator app", max_chars=6)

        col_a, col_b = st.columns(2)
        with col_a:
            confirm = st.button("Verify & Enable MFA")
        with col_b:
            cancel = st.button("Cancel")

        if cancel:
            # Reset MFA state, go back to normal login
            st.session_state["mfa_stage"] = None
            st.session_state["pending_mfa_secret"] = None
            st.session_state["pending_mfa_user_id"] = None
            st.session_state["pending_mfa_username"] = None
            st.rerun()


        if confirm:
            if totp.verify(code_input.strip()):
                # Save permanent secret
                user_id = st.session_state["pending_mfa_user_id"]
                set_user_mfa_secret(user_id, secret)

                update_mfa_timestamp(user_id)

                # Log admin in
                st.session_state["user_id"] = user_id
                st.session_state["username"] = st.session_state["pending_mfa_username"]
                st.session_state["authenticated"] = True
                st.session_state["is_admin"] = True

                # Clear MFA temp state
                st.session_state["mfa_stage"] = None
                st.session_state["pending_mfa_secret"] = None
                st.session_state["pending_mfa_user_id"] = None
                st.session_state["pending_mfa_username"] = None

                st.success("MFA enabled and admin logged in.")
                st.rerun()

            else:
                st.error("Invalid code. Please try again.")

        return  # stop rendering other login UI

    if st.session_state.get("mfa_stage") == "verify":
        st.markdown("### Admin: Enter your TOTP code")

        secret = st.session_state["pending_mfa_secret"]
        totp = pyotp.TOTP(secret)

        code_input = st.text_input("6-digit code", max_chars=6)

        col_a, col_b = st.columns(2)
        with col_a:
            verify = st.button("Verify")
        with col_b:
            cancel = st.button("Cancel")

        if cancel:
            # Reset MFA state, go back to login
            st.session_state["mfa_stage"] = None
            st.session_state["pending_mfa_secret"] = None
            st.session_state["pending_mfa_user_id"] = None
            st.session_state["pending_mfa_username"] = None
            st.rerun()


        if verify:
            if totp.verify(code_input.strip()):
                # Log admin in
                uid = st.session_state["pending_mfa_user_id"]
                st.session_state["user_id"] = uid

                # Correct MFA timestamp update
                update_mfa_timestamp(uid)
                upload_db_to_s3(DB_PATH, f"{S3_DB_PREFIX}/texts.db")


                st.session_state["username"] = st.session_state["pending_mfa_username"]
                st.session_state["authenticated"] = True
                st.session_state["is_admin"] = True


                # Clear MFA temp state
                st.session_state["mfa_stage"] = None
                st.session_state["pending_mfa_secret"] = None
                st.session_state["pending_mfa_user_id"] = None
                st.session_state["pending_mfa_username"] = None

                st.success("Admin authenticated.")
                st.rerun()

            else:
                st.error("Invalid code. Please try again.")

        return  # stop rendering other login UI

    


    # Sidebar configuration
    # Sidebar (ONLY visible after login)
    if st.session_state.get("authenticated", False):

# -----------------------------------
# DARK THEME (applies only after login)
# -----------------------------------
        st.markdown("""
            <style>
                body {
                    background-color: #1e1e1e !important;
                }
                .main .block-container {
                    background-color: #1e1e1e !important;
                    color: white !important;
                }
                .main {
                    background-color: #1e1e1e !important;
                }
                header, footer {
                    background: #1e1e1e !important;
                }
                h1, h2, h3, h4, h5, h6,
                p, div, span, label {
                    color: white !important;
                }
                /* Buttons */
                .stButton > button {
                    background-color: #444 !important;
                    color: white !important;
                    border-radius: 6px !important;
                    font-weight: 600 !important;
                }
                .stButton > button:hover {
                    background-color: #666 !important;
                }
            </style>
        """, unsafe_allow_html=True)


        with st.sidebar:

            # --------------------------
            # User Info + Sign Out
            # --------------------------
            admin_badge = " (Admin)" if st.session_state.get("is_admin", False) else ""
            st.write(f"**Signed in as:** {st.session_state['username']}{admin_badge}")

            if st.button("Sign Out", key="sidebar_signout_btn"):

                # Save progress BEFORE logout
                if st.session_state.get("username"):
                    save_progress(st.session_state["username"])

                # Reset state
                st.session_state["authenticated"] = False
                st.session_state["user_id"] = None
                st.session_state["username"] = None
                st.session_state["is_admin"] = False
                st.session_state["text_ids"] = []
                st.session_state["current_text_index"] = 0
                st.session_state["chosen_language"] = None

                st.rerun()

            st.markdown("---")

            # ---------------------------------------------------
            # ADMIN SIDEBAR BLOCK — MUST BE INSIDE THE SIDEBAR
            # ---------------------------------------------------
            if st.session_state.get("is_admin", False):

                st.header("Configuration")

                uploaded_file = st.file_uploader(
                    "Upload Text File",
                    type=["txt", "csv"],
                    help="Upload a text file with one text per line"
                )

                if uploaded_file is not None:
                    try:
                        import csv
                        decoded = uploaded_file.read().decode("utf-8").splitlines()
                        reader = csv.reader(decoded)

                        lines = []
                        for row in reader:
                            if row and row[0].strip():   # ensure non-empty first column
                                lines.append(row[0].strip())

                        language = st.selectbox("Language", ["ar", "ar-AE",  "ar-SA", "ar-QA", "ar-KW", "ar-SY", 
                        "ar-LB", "ar-PS", "ar-JO", "ar-EG", "ar-SD", "ar-TD", "ar-MA", "ar-DZ", "ar-TN", "he", "fa", "ur"], index=0,
                         key="file_lang")
                        is_rtl = st.checkbox("RTL Language", value=True, key="file_rtl")

                        # NEW: Track the original filename
                        source_file = uploaded_file.name

                        project_num = st.number_input(
                            "Project Number",
                            min_value=1,
                            max_value=999,
                            value=1,
                            step=1,
                            help="Which project these scripts belong to (1, 2, 3, ...)."
                        )
                     

                        if st.button("Import from File", key="admin_import_btn"):
                            count = 0
                            for line in lines:
                                add_text(
                                    line,
                                    language,
                                    is_rtl,
                                    st.session_state["user_id"],
                                    source_file=source_file,
                                    project=project_num
                                )
                                count += 1

                            st.success(f"Imported {count} texts from '{source_file}' into Project {project_num}")
                            st.rerun()


                    except Exception as e:
                        st.error(f"Error reading file: {e}")

                st.markdown("---")
                st.header("Manage Uploaded Files")

                # Get list of distinct files
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("SELECT DISTINCT source_file FROM texts WHERE source_file IS NOT NULL")
                files = [row[0] for row in c.fetchall()]
                conn.close()

                if not files:
                    st.info("No uploaded files found.")
                else:
                    chosen_file = st.selectbox("Select a file to manage:", files)

                    # Show number of texts inside this file
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("SELECT COUNT(*) FROM texts WHERE source_file=?", (chosen_file,))
                    file_count = c.fetchone()[0]
                    conn.close()

                    st.write(f"**This file contains {file_count} text items.**")

                    # Delete file button
                    if st.button("Delete This File and All Its Texts", key="delete_file_btn"):
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        c.execute("DELETE FROM texts WHERE source_file=?", (chosen_file,))
                        conn.commit()
                        conn.close()

                        # ⬅️ Required for persistence
                        upload_db_to_s3(DB_PATH, f"{S3_DB_PREFIX}/texts.db")

                        st.success(f"Deleted all entries from file '{chosen_file}'.")
                        st.rerun()


            # ---------------------------------------------------
            # REGULAR USER LANGUAGE SELECTION (Sidebar Only)
            # ---------------------------------------------------
            if st.session_state.get("authenticated", False) and not st.session_state.get("is_admin", False):

                st.header("Language")

                if not st.session_state.get("chosen_language"):
                    st.info("Assigned language")
                else:
                    st.info(f"Language: **{st.session_state['chosen_language']}**")
                    
        # -----------------------------------
        # DARK THEME (applies only after login)
        # -----------------------------------
        st.markdown("""
            <style>
                .main .block-container {
                    background-color: #1e1e1e !important;
                    color: white !important;
                }
                .main {
                    background-color: #1e1e1e !important;
                }
                h1, h2, h3, h4, h5, h6,
                p, div, span {
                    color: white !important;
                }
                .stMarkdown, .stText {
                    color: white !important;
                }
                /* Buttons */
                .stButton > button {
                    border-radius: 6px !important;
                    font-weight: 600 !important;
                }
            </style>
        """, unsafe_allow_html=True)


    # ---- FIRST-TIME LANGUAGE + PROJECT SELECTION (MAIN AREA) ----
    if st.session_state.get("authenticated", False) and not st.session_state.get("is_admin", False):

        # Always fetch latest language from DB to allow live admin updates  
        latest_lang = get_user_language(st.session_state["user_id"])
        st.session_state["chosen_language"] = latest_lang

        if latest_lang is None:
            st.markdown("### Welcome to datagen_v2")
            st.write("Your account has been created successfully.")
            st.info("Please contact your admin to be assigned a language before you can begin recording.")
            return

        # Fetch all texts for this language
        texts = get_all_texts(db_path=DB_PATH)
        # t structure: (id, prompts, language, is_rtl, user_id, project)
        lang_texts = [t for t in texts if t[2] == latest_lang]

        if not lang_texts:
            st.warning("No scripts have been assigned yet for your language.")
            return

        # Determine available projects for this language
        available_projects = sorted({t[5] for t in lang_texts})  # index 5 = project

        # Choose project (store in session)
        default_project = st.session_state.get("current_project") or min(available_projects)
        selected_project = st.selectbox(
            "Select Project",
            options=available_projects,
            index=available_projects.index(default_project) if default_project in available_projects else 0
        )
        st.session_state["current_project"] = selected_project

        # Enforce linear completion: must finish project N before N+1
        # If user picks project > 1 but hasn't completed previous project, block
        if selected_project > 1 and not project_is_completed(st.session_state["username"], selected_project - 1):
            st.warning(f"You must complete Project {selected_project - 1} before accessing Project {selected_project}.")
            return

        # Filter texts for this project
        project_texts = [t for t in lang_texts if t[5] == selected_project]
        if not project_texts:
            st.warning(f"No scripts found for Project {selected_project}.")
            return

        # Make sure text_ids reflect the current project
        st.session_state["text_ids"] = [t[0] for t in project_texts]

        # Load per-project progress (or start at 0 if none)
        saved_step, project_completed = get_project_progress(st.session_state["username"], selected_project)
        st.session_state["current_text_index"] = min(saved_step, max(len(st.session_state["text_ids"]) - 1, 0))
        st.session_state["user_completed"] = project_completed




    # Main prompt UI as before, but the prompts are now filtered by chosen language
    if st.session_state.get("text_ids"):

        # --- FIX 3: Validate current_text_index before using it ---
        idx = st.session_state.get("current_text_index")

        if (
            idx is None
            or not isinstance(idx, int)
            or idx < 0
            or idx >= len(st.session_state["text_ids"])
        ):
            st.session_state["current_text_index"] = 0
            idx = 0

        current_text_id = st.session_state["text_ids"][idx]
        text_data = get_text_by_id(current_text_id)

        if text_data:
            text_id, text, language, is_rtl, user_id, project = text_data
            st.markdown(
                "<p style='color: white; font-size: 16px; margin-bottom: 10px;'>Please record yourself reading the following scripts out loud:</p>",
                unsafe_allow_html=True
            )
            st.subheader(
                f"Recording {st.session_state['current_text_index'] + 1} of {len(st.session_state['text_ids'])}"
            )
            st.markdown(
                f"<div style='direction:{'rtl' if is_rtl else 'ltr'}; text-align:{'right' if is_rtl else 'left'}; font-size:34px; padding:20px; background:#2d2d2d; color:#ffffff; border-radius:10px; margin:20px 0;'>{text}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("---")

            is_at_end = st.session_state["current_text_index"] == len(st.session_state["text_ids"]) - 1

            if is_at_end and st.session_state.get("user_completed", False):
                col1, col2, col3 = st.columns([2, 1, 2])
                with col2:
                    st.markdown(
                        """
                        <div style="
                            text-align: center;
                            font-size: 40px;
                            font-weight: bold;
                            color: #00ff55;
                            background: #003300;
                            padding: 30px;
                            border-radius: 12px;
                        ">
                            You've now completed the project! Thank you!
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
 

            # Determine whether the mic should be active
            is_final_screen = (
                st.session_state["current_text_index"] == len(st.session_state["text_ids"]) - 1
            )
            mic_disabled = (is_final_screen and st.session_state.get("user_completed", False))

            col1, col2, col3 = st.columns([3, 2, 1.5])
            with col2:
                if mic_disabled:
                    # Render a DISABLED-LOOKING microphone, but DO NOT unmount the component
                    st.warning("You've now completed the project! Thank you!")

                    # Render a 'fake' mic button (non-functional) to preserve layout
                    st.markdown("""
                        <div style="opacity:0.35; pointer-events:none;">
                            <button style="background:#555; color:#999; padding:15px; border-radius:50%; font-size:24px;">
                                🎤
                            </button>
                        </div>
                    """, unsafe_allow_html=True)

                    audio_bytes = None

                else:
                    # Always render the real recorder component so JS stays mounted
                    audio_bytes = audio_recorder(
                        text="",
                        recording_color="#e74c3c",
                        neutral_color="#6c757d",
                        icon_name="microphone",
                        icon_size="6x",
                    )


            # --- FIXED SUBMISSION LOGIC ---
            new_audio_available = audio_bytes not in (None, b"", [], {}, ())

            # Never treat audio as duplicate unless SHA-256 matches
            last_hash = st.session_state.get("last_submitted_audio_hash")

            if audio_bytes:
                import hashlib
                current_hash = hashlib.sha256(audio_bytes).hexdigest()
            else:
                current_hash = None

            is_duplicate_audio = (current_hash is not None and current_hash == last_hash)

            # Always compute submit_disabled
            submit_disabled = (not new_audio_available) or is_duplicate_audio

            # Preview audio if present
            if audio_bytes not in (None, b""):
                st.audio(audio_bytes, format="audio/wav")

            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                submit = st.button(
                    "Submit",
                    use_container_width=True,
                    disabled=submit_disabled
                )

            if submit and new_audio_available and not is_duplicate_audio:
                # --- SAVE AUDIO TO S3 ---
                user_id = st.session_state["user_id"]
                username = st.session_state["username"]
                user_lang = st.session_state["chosen_language"]
                text_number = st.session_state["current_text_index"] + 1

                base_name = f"{language}_{user_id}_{text_number}"
                audio_key = f"{user_lang}/{username}/audio/{base_name}.wav"
                text_key  = f"{user_lang}/{username}/transcripts/{base_name}.txt"

                # Start async upload so UI does NOT freeze
                upload_async(audio_bytes, audio_key, text_key, text)

                # Save the *expected* S3 URI (UI doesn't wait for upload to finish)
                audio_s3_uri = f"s3://{AWS_BUCKET_NAME}/{audio_key}"
                save_recording(text_id, audio_s3_uri, "saved")

                # Remember SHA256 instead of raw bytes
                st.session_state["last_submitted_audio_hash"] = hashlib.sha256(audio_bytes).hexdigest()

                # Advance to next recording
                if st.session_state["current_text_index"] == len(st.session_state["text_ids"]) - 1:
                    st.session_state["user_completed"] = True
                    save_progress(st.session_state["username"])
                else:
                    st.session_state["current_text_index"] += 1
                    save_progress(st.session_state["username"])


                st.success("Recording submitted!")
                st.rerun()

    # User's own recordings section (for regular users)
    if not st.session_state.get("is_admin", False) and st.session_state.get("user_id"):
        st.markdown("---")
        submitted_count = st.session_state["current_text_index"]
        st.subheader(f"My Recordings ({submitted_count})")
        user_recordings = get_all_recordings_by_user(user_id=st.session_state["user_id"])
                # --- Assign user-local recording numbers instead of global DB IDs ---
        # DB returns newest first, so reverse for chronological numbering
        ordered = list(reversed(user_recordings))

        indexed_records = []
        for i, rec in enumerate(ordered, start=1):
            indexed_records.append((i, rec))  # (local_number, db_row)

        if user_recordings:
            for local_num, rec in indexed_records:
                rec_id, audio_path, job_id, status, created_at, text_content, text_id, username, duration_seconds = rec

                with st.expander(f"Recording {local_num} - {created_at} "):
                    st.write(f"**Text ID:** {text_id}")
                    st.write(f"**Text:** {text_content[:100]}{'...' if len(text_content) > 100 else ''}")
                    if audio_path:
                        if audio_path.startswith("s3://"):
                            # S3 object – generate a presigned URL
                            try:
                                url = generate_presigned_url(audio_path)
                                st.audio(url, format="audio/wav")
                            except Exception as e:
                                st.error(f"Could not load audio from S3: {e}")
                        else:
                            # Local file – check existence
                            if os.path.exists(audio_path):
                                st.audio(audio_path, format="audio/wav")
                            else:
                                st.warning(f"Audio file not found: {audio_path}")

        else:
            st.info("You haven't made any recordings yet.")


# Admin section: Show all recordings (GROUPED BY USER)
    if st.session_state.get("is_admin", False):
        # -------------------------
        # ADMIN USER MANAGEMENT
        # -------------------------
        st.subheader("User Management")

        users = get_all_users()
        search = st.text_input("Search users by name")

        filtered = [
            u for u in users
            if search.lower() in u[1].lower()
        ]

        for user_id, uname, is_admin_flag, lang in filtered:
            with st.expander(f"{uname}"):

                SUPER_ADMIN = os.getenv("ADMIN_USERNAME")
                is_super_admin = (uname == SUPER_ADMIN)

                st.write(f"User ID: {user_id}")
                st.write(f"Current language: {lang or 'None assigned'}")
                st.write(f"Admin: {'Yes' if is_admin_flag else 'No'}")

                # -------------------------------
                # Assign Language (Allowed Always)
                # -------------------------------
                new_lang = st.selectbox(
                    "Assign language",
                    ["None", "ar", "ar-AE", "ar-SA", "ar-QA", "ar-KW", "ar-SY",
                    "ar-LB", "ar-PS", "ar-JO", "ar-EG", "ar-SD", "ar-TD",
                    "ar-MA", "ar-DZ", "ar-TN", "he", "fa", "ur"],
                    index=0 if lang is None else 1,
                    key=f"lang_select_user_{user_id}"
                )

                if st.button(f"Save Language for {uname}", key=f"save_lang_{user_id}"):
                    if new_lang == "None":
                        save_user_language(user_id, None)
                    else:
                        save_user_language(user_id, new_lang)
                    st.success("Language updated.")
                    st.rerun()

                st.markdown("---")

                # ------------------------------------
                # SUPER ADMIN PROTECTION (UI Controls)
                # ------------------------------------
                if is_super_admin:
                    st.info("This user is the SUPER ADMIN and cannot be modified.")
                    # ⬆ No admin toggle, no MFA reset
                    continue

                # ---------------------------
                # Promote / Demote Admin
                # ---------------------------
                if is_admin_flag:
                    if st.button(
                        f"Remove Admin Access from {uname}",
                        key=f"demote_admin_{user_id}"
                    ):
                        set_user_admin(user_id, False)
                        st.success("User demoted from admin.")
                        st.rerun()
                else:
                    if st.button(
                        f"Make {uname} Admin",
                        key=f"promote_admin_{user_id}"
                    ):
                        set_user_admin(user_id, True)
                        reset_user_mfa(user_id)  # force MFA setup
                        st.success("User promoted to admin — MFA required on next login.")
                        st.rerun()

                # ---------------------------
                # Reset MFA (Not for super admin)
                # ---------------------------
                if st.button(f"Reset MFA for {uname}", key=f"reset_mfa_{user_id}"):
                    reset_user_mfa(user_id)
                    st.success("MFA reset — user will need to re-enroll.")
                    st.rerun()


        st.markdown("---")
        st.subheader("All Recordings")

        # Fully explicit JOIN so admin always sees all recordings
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute("""
            SELECT
                r.id,
                r.audio_file_path,
                r.hoppepper_job_id,
                r.status,
                r.created_at,
                t.prompts,
                t.id AS text_id,
                u.username,
                r.duration_seconds
            FROM recordings r
            LEFT JOIN users u ON r.user_id = u.id
            LEFT JOIN texts t ON r.text_id = t.id
            ORDER BY r.created_at DESC

        """)


        all_recordings = c.fetchall()
        conn.close()

        if not all_recordings:
            st.info("No recordings found in the system.")
        else:
            # --- Clean, Robust Grouping for Admin View ---
            grouped = {}

            for rec in all_recordings:
                rec_id, audio_path, job_id, status, created_at, text_content, text_id, username_from_db, duration_seconds = rec

                # ✅ ALWAYS derive speaker from S3 path
                username_key = "Unknown"

                if audio_path and audio_path.startswith("s3://"):
                    clean = audio_path.replace(f"s3://{AWS_BUCKET_NAME}/", "")
                    parts = clean.split("/")
                    # Expected: language/username/audio/file.wav
                    if len(parts) >= 3 and parts[1]:
                        username_key = parts[1]

                grouped.setdefault(username_key, []).append(rec)


            # -----------------------------------------
            # TOTAL HOURS FOR ALL USERS (Admin Summary)
            # -----------------------------------------
            global_total_seconds = 0.0

            for rec in all_recordings:
                (
                    rec_id,
                    audio_path,
                    job_id,
                    status,
                    created_at,
                    text_content,
                    text_id,
                    username_from_db,
                    duration_seconds,
                ) = rec

                try:
                    global_total_seconds += float(duration_seconds or 0)
                except (TypeError, ValueError):
                    pass


            global_total_hours = global_total_seconds / 3600.0

            st.success(
                f"📊 **Total Hours Recorded (All Users): {global_total_hours:.2f} hours**"
            )


            # Render group dropdowns
            for username, rec_list in grouped.items():

                total_seconds = 0.0

                for rec in rec_list:
                    rec_id, audio_path, job_id, status, created_at, text_content, text_id, username, duration_seconds = rec
                    if duration_seconds is not None:
                        total_seconds += float(duration_seconds or 0)


                total_hours = total_seconds / 3600.0

                # --- EXPANDER WITH HOURS ---
                with st.expander(f"User: {username} — {len(rec_list)} recordings — {total_hours:.2f} hours recorded"):

                    # Inside each dropdown, list that user's recordings
                    for rec in rec_list:
                        rec_id, audio_path, job_id, status, created_at, text_content, text_id, username_from_db, duration_seconds = rec

                        with st.expander(f"Recording {rec_id} — {created_at}"):

                            st.write(f"**Text ID:** {text_id}")
                            st.write(f"**Text:** {text_content[:100]}{'...' if len(text_content) > 100 else ''}")

                            if audio_path:
                                if audio_path.startswith("s3://"):
                                    try:
                                        url = generate_presigned_url(audio_path)
                                        st.audio(url, format="audio/wav")
                                    except Exception as e:
                                        st.error(f"Could not load audio: {e}")
                                else:
                                    st.audio(audio_path, format="audio/wav")


            # --- End of Section 3 ---



# ---------------------------------
# CLI fallback & simple test suite
# ---------------------------------
class _FakeResp:
    def __init__(self, status_code: int = 200, payload: Optional[dict] = None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload

def run_tests(tmp_db: str = "test_texts.db") -> None:
    """Minimal tests to validate DB & API shims without Streamlit."""
    if os.path.exists(tmp_db):
        os.remove(tmp_db)

    print("[TEST] init_db …", end=" ")
    init_db(tmp_db)
    assert os.path.exists(tmp_db), "DB file not created"
    print("OK")

    print("[TEST] add_text/get_all_texts …", end=" ")
    tid1 = add_text("مرحبا", "ar", True, user_id=None, db_path=tmp_db)
    tid2 = add_text("שלום", "he", True, user_id=None, db_path=tmp_db)
    arr = get_all_texts(user_id=None, db_path=tmp_db)
    assert len(arr) == 2 and arr[0][0] == tid1, "Texts not inserted/read correctly"
    print("OK")

    print("[TEST] save_recording/get_recordings_by_text …", end=" ")
    rid = save_recording(tid1, "recordings/fake.wav", "saved", tmp_db)
    rows = get_recordings_by_text(tid1, tmp_db)
    assert rows and rows[0][0] == rid, "Recording not saved/read"
    print("OK")

    print("[TEST] update_recording_status …", end=" ")
    update_recording_status(rid, "updated", tmp_db)
    rows = get_recordings_by_text(tid1, tmp_db)
    assert rows[0][3] == "updated", "Status not updated"
    print("OK")

    print("[TEST] audio duplicate prevention …", end=" ")
    dummy_bytes = b"FAKEAUDIO"
    audio_fp = "recordings/dummy.wav"
    os.makedirs("recordings", exist_ok=True)
    with open(audio_fp, "wb") as f:
        f.write(dummy_bytes)
    save_recording(tid1, audio_fp, "saved", tmp_db)
    assert recording_exists_for_text(tid1, dummy_bytes, tmp_db), "Duplicate detection failed"
    print("OK")

    print("All tests passed")

if __name__ == "__main__":
    init_db()
    if HAS_STREAMLIT:
        run_streamlit_app()
    else:
        print("Streamlit is not installed in this environment. Running CLI self-tests instead.\n")
        print("To launch the web app, install dependencies:")
        print("    pip install streamlit audio-recorder-streamlit requests\n")
        run_tests()
        print("\nIf you want me to modify this for a pure CLI workflow, tell me your expected behavior (navigation, recording, upload).")

