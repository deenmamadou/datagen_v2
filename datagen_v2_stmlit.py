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
    conn = sqlite3.connect("user_progress.db")
    return conn

# Create table if it does not exist
conn = get_db_connection()
conn.execute("""
    CREATE TABLE IF NOT EXISTS progress (
        username TEXT PRIMARY KEY,
        step INTEGER,
        audio BLOB
    )
""")
conn.commit()
conn.close()


# ---------------------------
# Database setup & helpers
# ---------------------------
DB_PATH = "texts.db"


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


    c.execute(
        """
        CREATE TABLE IF NOT EXISTS texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompts TEXT NOT NULL,
            language TEXT NOT NULL,
            is_rtl BOOLEAN DEFAULT 1,
            user_id INTEGER,
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


def get_all_texts(user_id: Optional[int] = None, db_path: str = DB_PATH) -> list:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    if user_id:
        c.execute("SELECT id, prompts, language, is_rtl, user_id FROM texts WHERE user_id = ?", (user_id,))
    else:
        c.execute("SELECT id, prompts, language, is_rtl, user_id FROM texts")
    texts = c.fetchall()
    conn.close()
    return texts


def get_text_by_id(text_id: int, db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, prompts, language, is_rtl, user_id FROM texts WHERE id = ?", (text_id,))
    text = c.fetchone()
    conn.close()
    return text


def add_text(text: str, language: str = "ar", is_rtl: bool = True,
             user_id: Optional[int] = None, source_file: Optional[str] = None,
             db_path: str = DB_PATH) -> int:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
        INSERT INTO texts (prompts, language, is_rtl, user_id, source_file)
        VALUES (?, ?, ?, ?, ?)
    """, (text, language, is_rtl, user_id, source_file))

    conn.commit()
    text_id = c.lastrowid
    conn.close()
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
    ADMIN_USERNAME = "deen.mamadou-yacoubou@deepgram.com"
    ADMIN_PASSWORD = "Deepgram"
    
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

def load_progress(user):
    conn = get_db_connection()
    row = conn.execute("SELECT step, audio FROM progress WHERE username=?", (user,)).fetchone()
    conn.close()

    if row:
        saved_step = row[0]
        st.session_state["current_text_index"] = saved_step
        st.session_state["audio_bytes"] = row[1]
    else:
        st.session_state["current_text_index"] = 0
        st.session_state["audio_bytes"] = None


def save_progress(user):
    conn = get_db_connection()
    step = st.session_state.get("current_text_index", 0)
    audio = st.session_state.get("audio_bytes")
    conn.execute(
        "REPLACE INTO progress (username, step, audio) VALUES (?, ?, ?)",
        (user, step, audio)
    )
    conn.commit()
    conn.close()

    

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

def save_recording(
    text_id: int,
    audio_file_path: str,
    status: str = "saved",
    db_path: str = DB_PATH,
) -> int:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO recordings (text_id, audio_file_path, hoppepper_job_id, status)
        VALUES (?, ?, NULL, ?)
        """,
        (text_id, audio_file_path, status),
    )
    conn.commit()
    recording_id = c.lastrowid
    conn.close()
    return recording_id

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
            SELECT r.id, r.audio_file_path, r.hoppepper_job_id, r.status, r.created_at,
                   t.prompts, t.id as text_id, u.username
            FROM recordings r
            JOIN texts t ON r.text_id = t.id
            LEFT JOIN users u ON t.user_id = u.id
            ORDER BY r.created_at DESC
        """)
    else:
        # REGULAR USERS — filter by their per-user audio folder
        pattern = f"%user_{user_id}/audio/%"

        c.execute("""
            SELECT r.id, r.audio_file_path, r.hoppepper_job_id, r.status, r.created_at,
                   t.prompts, t.id as text_id, u.username
            FROM recordings r
            JOIN texts t ON r.text_id = t.id
            LEFT JOIN users u ON t.user_id = u.id
            WHERE r.audio_file_path LIKE ?
            ORDER BY r.created_at DESC
        """, (pattern,))

    rows = c.fetchall()
    conn.close()
    return rows

# ---------------------------
# Streamlit UI 
# ---------------------------

def run_streamlit_app() -> None:
    st.set_page_config(page_title="datagen_v2", layout="wide")

    # Ensure DB is ready
    init_db()

    # Session defaults
    for key, val in {"text_ids": [], "current_text_index": 0, "audio_bytes": None, "user_id": None, "username": None, "authenticated": False, "is_admin": False}.items():
        st.session_state.setdefault(key, val)

    # Authentication UI
    # Authentication UI
    if not st.session_state.get("authenticated", False):

        # --- Login Title ---
        st.markdown(
            "<h1 style='text-align: center;'>datagen_v2</h1>",
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

                    if user_id:
                        # Basic session info
                        st.session_state["user_id"] = user_id
                        st.session_state["username"] = username
                        st.session_state["authenticated"] = True
                        st.session_state["is_admin"] = is_admin

                        # Load language + progress
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

                        st.success("Signed in successfully!" + (" (Admin)" if is_admin else ""))
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

        return


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
                        content = uploaded_file.read().decode("utf-8")
                        lines = [line.strip() for line in content.split("\n") if line.strip()]

                        language = st.selectbox("Language", ["ar", "he", "fa", "ur"], index=0, key="file_lang")
                        is_rtl = st.checkbox("RTL Language", value=True, key="file_rtl")

                        # NEW: Track the original filename
                        source_file = uploaded_file.name

                        if st.button("Import from File", key="admin_import_btn"):
                            count = 0
                            for line in lines:
                                add_text(
                                    line,
                                    language,
                                    is_rtl,
                                    st.session_state["user_id"],
                                    source_file=source_file   # NEW
                                )
                                count += 1

                            st.success(f"Imported {count} texts from '{source_file}'")
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
                        st.success(f"Deleted all entries from file '{chosen_file}'.")
                        st.rerun()


            # ---------------------------------------------------
            # REGULAR USER LANGUAGE SELECTION (Sidebar Only)
            # ---------------------------------------------------
            if st.session_state.get("authenticated", False) and not st.session_state.get("is_admin", False):

                st.header("Language")

                if not st.session_state.get("chosen_language"):
                    st.info("Please choose your language in the main panel.")
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


        # ---- FIRST-TIME LANGUAGE SELECTION (MAIN AREA) ----
        if st.session_state.get("authenticated", False) and not st.session_state.get("is_admin", False):
            if st.session_state.get("chosen_language") is None:
                st.markdown("## Choose your language")
                texts = get_all_texts(db_path=DB_PATH)
                language_choices = sorted({t[2] for t in texts})

                if not language_choices:
                    st.warning("No languages available. Please contact your admin.")
                    # Nothing else to show yet
                    return

                chosen_lang = st.selectbox(
                    "Language",
                    language_choices,
                    key="main_lang_choice"
                )

                if st.button("Confirm Language", key="main_lang_confirm"):
                    # Save permanently for this user
                    save_user_language(st.session_state["user_id"], chosen_lang)
                    st.session_state["chosen_language"] = chosen_lang

                    # Load text IDs for that language
                    lang_texts = [t for t in texts if t[2] == chosen_lang]
                    st.session_state["text_ids"] = [t[0] for t in lang_texts]

                    # Start from saved progress if any (for safety)
                    idx = st.session_state.get("current_text_index", 0)
                    if idx < 0 or idx >= len(st.session_state["text_ids"]):
                        idx = 0
                    st.session_state["current_text_index"] = idx

                    st.rerun()

                # IMPORTANT: stop here so main UI doesn't render until language is set
                return

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
            text_id, text, language, is_rtl, user_id = text_data
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

            if is_at_end:
                col1, col2, col3 = st.columns([2, 1, 2])
                with col2:
                    st.markdown(
                        "<div style='text-align: center; font-size: 24px; font-weight: bold; color: white; padding: 20px;'>The End</div>",
                        unsafe_allow_html=True
                    )

            if audio_recorder is not None:
                col1, col2, col3 = st.columns([3, 2, 1.5])
                with col2:
                    # No extra HTML around the widget, just the widget call:
                    from custom_recorder.recorder import audio_recorder_5s

                    audio_bytes = audio_recorder_5s(key="rec1")
                    if audio_bytes:
                        audio_bytes = bytes(audio_bytes)  # convert JS list → bytes

                    st.markdown("""
                    <script>
                        function extendSilenceTimeout() {
                            const recorder = window.streamlitAudioRecorder;
                            if (!recorder) {
                                setTimeout(extendSilenceTimeout, 300);
                                return;
                            }

                            // Increase silence auto-stop timeout (milliseconds)
                            recorder.VAD_SILENCE_TIMEOUT = 44000;   // ← adjust this
                            console.log("Updated VAD silence timeout:", recorder.VAD_SILENCE_TIMEOUT);
                        }
                        extendSilenceTimeout();
                    </script>
                    """, unsafe_allow_html=True)


                    is_disabled = "true" if is_at_end else "false"
                    st.markdown(f"""
                        <script>
                            (function() {{
                                const isDisabled = {is_disabled};
                                function findRecorderButton() {{
                                    let btn = document.querySelector('button[data-testid*="audio"]');
                                    if (btn) return btn;
                                    const buttons = Array.from(document.querySelectorAll('button'));
                                    for (let b of buttons) {{
                                        const svg = b.querySelector('svg');
                                        if (svg && b.offsetParent !== null) {{
                                            return b;
                                        }}
                                    }}
                                    return null;
                                }}
                                function setupBand() {{
                                    const band = document.getElementById('record-band');
                                    if (!band) return;
                                    if (isDisabled) {{
                                        band.onclick = function(e) {{
                                            e.preventDefault(); e.stopPropagation(); return false;
                                        }};
                                        return;
                                    }}
                                    band.onclick = function(e) {{
                                        e.preventDefault(); e.stopPropagation();
                                        const btn = findRecorderButton();
                                        if (btn) btn.click();
                                        band.classList.add('recording');
                                        setTimeout(function() {{
                                            band.classList.remove('recording');
                                        }}, 200);
                                    }};
                                }}
                                setupBand();
                                const observer = new MutationObserver(setupBand);
                                observer.observe(document.body, {{childList: true, subtree: true, attributes: true}});
                            }})();
                        </script>
                    """, unsafe_allow_html=True)
            else:
                audio_bytes = None

            # --- SUBMISSION LOGIC, GREEN BUTTON, NEW/ALREADY SUBMITTED CHECK ---
            new_recording = False
            prev_bytes = st.session_state.get("prev_audio_bytes", None)
            # Only allow submission for freshly captured new audio (not None, not identical to last submitted)
            if audio_bytes:
                if prev_bytes is None or audio_bytes != prev_bytes:
                    st.session_state["audio_submitted"] = False
                st.session_state["audio_bytes"] = audio_bytes
                st.audio(audio_bytes, format="audio/wav")
                new_recording = not st.session_state.get("audio_submitted", False)
            else:
                new_recording = False

            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                submit_disabled = True
                duplicate = False
                if new_recording and audio_bytes:
                    duplicate = recording_exists_for_text(text_id, audio_bytes)
                    submit_disabled = duplicate or st.session_state.get("audio_submitted", False)
                else:
                    submit_disabled = True

                if duplicate:
                    st.error("You have already submitted this exact recording for this text. Please record a new audio.")

                submit_result = st.button("Submit", key="submit-btn", use_container_width=True, disabled=submit_disabled)

                if submit_result and new_recording and not duplicate:

                    # -------------------------
                    # Build per-user directories
                    # -------------------------
                    user_id = st.session_state["user_id"]
                    langcode = language  # from text_data tuple
                    text_number = st.session_state["current_text_index"] + 1

                    base_dir = f"recordings/user_{user_id}"
                    audio_dir = f"{base_dir}/audio"
                    txt_dir = f"{base_dir}/transcripts"
                    os.makedirs(audio_dir, exist_ok=True)
                    os.makedirs(txt_dir, exist_ok=True)

                    # -------------------------
                    # Unified filename
                    # -------------------------
                    base_name = f"{langcode}_{user_id}_{text_number}"

                    # --- Save audio ---
                    audio_filename = f"{audio_dir}/{base_name}.wav"
                    with open(audio_filename, "wb") as f:
                        f.write(audio_bytes)

                    # --- Save displayed text as .txt ---
                    txt_filename = f"{txt_dir}/{base_name}.txt"
                    with open(txt_filename, "w", encoding="utf-8") as f:
                        f.write(text)

                    # Save audio path to DB (no changes)
                    save_recording(text_id, audio_filename, "saved")

                    # -------------------------
                    # Existing state logic stays the same
                    # -------------------------
                    st.session_state["audio_submitted"] = True
                    st.session_state["prev_audio_bytes"] = audio_bytes
                    st.session_state["audio_bytes"] = None

                    if not is_at_end:
                        st.session_state["current_text_index"] += 1

                    if st.session_state["current_text_index"] >= len(st.session_state["text_ids"]):
                        st.session_state["current_text_index"] = len(st.session_state["text_ids"]) - 1

                    st.success("Recording submitted! Moving to the next script.")
                    st.rerun()


    else:
        if st.session_state.get("authenticated", False):
            st.info("Choose your language from the sidebar to get started.")

    # User's own recordings section (for regular users)
    if not st.session_state.get("is_admin", False) and st.session_state.get("user_id"):
        st.markdown("---")
        st.subheader("My Recordings")
        user_recordings = get_all_recordings_by_user(user_id=st.session_state["user_id"])
        if user_recordings:
            for rec in user_recordings:
                rec_id, audio_path, job_id, status, created_at, text_content, text_id, username = rec
                with st.expander(f"Recording {rec_id} - {created_at} | Status: {status}"):
                    st.write(f"**Text ID:** {text_id}")
                    st.write(f"**Text:** {text_content[:100]}{'...' if len(text_content) > 100 else ''}")
                    if os.path.exists(audio_path):
                        st.audio(audio_path, format="audio/wav")
                    st.write(f"**Status:** {status}")
                    st.write(f"**Created:** {created_at}")
        else:
            st.info("You haven't made any recordings yet.")

    # Admin section: Show all recordings
    if st.session_state.get("is_admin", False):
        st.markdown("---")
        st.subheader("Admin: All Recordings")
        all_recordings = get_all_recordings_by_user(user_id=None)
        if all_recordings:
            for rec in all_recordings:
                rec_id, audio_path, job_id, status, created_at, text_content, text_id, username = rec
                with st.expander(f"Recording {rec_id} - User: {username or 'Unknown'} - {created_at} | Status: {status}"):
                    st.write(f"**Text ID:** {text_id}")
                    st.write(f"**Text:** {text_content[:100]}{'...' if len(text_content) > 100 else ''}")
                    st.write(f"**User:** {username or 'Unknown'}")
                    if os.path.exists(audio_path):
                        st.audio(audio_path, format="audio/wav")
                    st.write(f"**Status:** {status}")
                    st.write(f"**Created:** {created_at}")
        else:
            st.info("No recordings found in the system.")


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
