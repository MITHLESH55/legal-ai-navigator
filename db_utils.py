# db_utils.py
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List
import logging
import threading
import uuid # For default session ID generation if needed

logger = logging.getLogger(__name__)
DB_NAME = 'chat_g_history.db'
SESSION_TIMEOUT_MINUTES = 15 # Session timeout duration

# Use a lock for thread-safe database operations, especially with Flask's threading
db_lock = threading.Lock()

def get_db_connection():
    """Returns a new database connection."""
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    """Initializes the SQLite database and tables if they don't exist."""
    with db_lock:
        try:
            # check_same_thread=False is generally okay for Flask apps if used with locks,
            # but consider connection pooling for very high concurrency.
            conn = sqlite3.connect(DB_NAME, check_same_thread=False)
            cursor = conn.cursor()
            
            # === USERS TABLE ===
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_verified INTEGER DEFAULT 1
                )
            ''')
            
            # === PASSWORD RESET TOKENS TABLE ===
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS password_reset_tokens (
                    token_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    otp_code TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    is_used INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            ''')
            
            # Store session info
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    uploaded_files TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            ''')
            
            # Add user_id and uploaded_files columns if they don't exist (migration for existing DBs)
            cursor.execute("PRAGMA table_info(chat_sessions)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'user_id' not in columns:
                cursor.execute('ALTER TABLE chat_sessions ADD COLUMN user_id TEXT')
                logger.info("Added user_id column to chat_sessions table.")
            if 'uploaded_files' not in columns:
                cursor.execute('ALTER TABLE chat_sessions ADD COLUMN uploaded_files TEXT')
                logger.info("Added uploaded_files column to chat_sessions table.")
                
            # Store messages linked to sessions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'model')), -- 'user' or 'model'
                    content TEXT NOT NULL,
                    file_references TEXT, -- JSON list of Gemini file names/URIs used IN that turn
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
                )
            ''')
            # Index for faster message retrieval
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_session_timestamp ON chat_messages (session_id, timestamp);
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_user_sessions ON chat_sessions (user_id);
            ''')
            conn.commit()
            conn.close()
            logger.info(f"Database '{DB_NAME}' initialized successfully.")
        except sqlite3.Error as e:
            logger.error(f"FATAL: Error initializing database '{DB_NAME}': {e}", exc_info=True)
            # Depending on the error, might want to raise it to stop the app
            raise

def _prune_old_sessions():
     """Removes sessions older than the timeout threshold. Called internally."""
     with db_lock:
        try:
            conn = sqlite3.connect(DB_NAME, check_same_thread=False)
            cursor = conn.cursor()
            timeout_threshold = datetime.now() - timedelta(minutes=SESSION_TIMEOUT_MINUTES)
            # Delete expired sessions; linked messages will be deleted by CASCADE
            cursor.execute("DELETE FROM chat_sessions WHERE last_accessed < ?", (timeout_threshold,))
            conn.commit()
            deleted_count = cursor.rowcount
            conn.close()
            if deleted_count > 0:
                logger.info(f"Pruned {deleted_count} expired chat sessions older than {SESSION_TIMEOUT_MINUTES} minutes.")
        except sqlite3.Error as e:
             logger.error(f"Error pruning old sessions: {e}", exc_info=True)


def add_or_update_session(session_id: str, uploaded_files_data: list = None, user_id: int = None):
     """Creates a new session or updates the last_accessed time if it exists.
     
     Args:
         session_id: The session identifier
         uploaded_files_data: List of dicts with file info: [{"uri": "files/xxx", "mime_type": "application/pdf", "display_name": "doc.pdf"}]
         user_id: The user ID to link this session to (optional)
     """
     _prune_old_sessions() # Clean up expired sessions first
     with db_lock:
        try:
            conn = sqlite3.connect(DB_NAME, check_same_thread=False)
            cursor = conn.cursor()
            now = datetime.now()
            
            # Check if session exists
            cursor.execute("SELECT uploaded_files FROM chat_sessions WHERE session_id = ?", (session_id,))
            existing = cursor.fetchone()
            
            if existing and existing[0]:
                # Session exists with files - merge new files with existing
                existing_files = json.loads(existing[0])
                if uploaded_files_data:
                    # Add new files, avoiding duplicates by URI
                    existing_uris = {f['uri'] for f in existing_files}
                    for new_file in uploaded_files_data:
                        if new_file['uri'] not in existing_uris:
                            existing_files.append(new_file)
                files_json = json.dumps(existing_files)
            else:
                # New session or no existing files
                files_json = json.dumps(uploaded_files_data) if uploaded_files_data else None
            
            # Use INSERT OR REPLACE or ON CONFLICT to simplify logic
            cursor.execute('''
                INSERT INTO chat_sessions (session_id, created_at, last_accessed, uploaded_files, user_id)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET 
                    last_accessed = excluded.last_accessed,
                    uploaded_files = excluded.uploaded_files,
                    user_id = COALESCE(excluded.user_id, user_id)
            ''', (session_id, now, now, files_json, user_id))
            conn.commit()
            conn.close()
            logger.debug(f"Session {session_id} added or updated with {len(uploaded_files_data) if uploaded_files_data else 0} files and user_id {user_id}.")
        except sqlite3.Error as e:
             logger.error(f"Error adding/updating session {session_id}: {e}", exc_info=True)

def save_chat_message(session_id: str, role: str, content: str, file_references: list = None):
    """Saves a message to the chat history for a given session."""
    if role not in ['user', 'model']:
        logger.error(f"Invalid role '{role}' for saving message.")
        return
    # Ensure session exists and update its timestamp
    add_or_update_session(session_id)
    with db_lock:
        try:
            conn = sqlite3.connect(DB_NAME, check_same_thread=False)
            cursor = conn.cursor()
            # Serialize file references if they exist
            files_json = json.dumps(file_references) if file_references else None
            cursor.execute('''
                INSERT INTO chat_messages (session_id, role, content, file_references)
                VALUES (?, ?, ?, ?)
            ''', (session_id, role, content, files_json))
            conn.commit()
            conn.close()
            logger.debug(f"Saved {role} message for session {session_id}.")
        except sqlite3.Error as e:
            logger.error(f"Error saving message for session {session_id}: {e}", exc_info=True)

def get_session_files(session_id: str) -> List[dict]:
    """Retrieves uploaded files for a session.
    
    Returns:
        List of dicts: [{"uri": "files/xxx", "mime_type": "application/pdf", "display_name": "doc.pdf"}]
    """
    _prune_old_sessions()
    with db_lock:
        try:
            conn = sqlite3.connect(DB_NAME, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute("SELECT uploaded_files FROM chat_sessions WHERE session_id = ?", (session_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:
                return json.loads(result[0])
            return []
        except sqlite3.Error as e:
            logger.error(f"Error retrieving files for session {session_id}: {e}", exc_info=True)
            return []

def get_user_sessions(user_id: int) -> List[dict]:
    """Retrieves all chat sessions for a specific user.
    
    Returns:
        List of dicts with session info: [{"session_id": "...", "created_at": "...", "last_accessed": "...", "preview": "..."}]
    """
    _prune_old_sessions()
    with db_lock:
        try:
            conn = sqlite3.connect(DB_NAME, check_same_thread=False)
            cursor = conn.cursor()
            
            # Get all sessions for this user, ordered by last_accessed (newest first)
            cursor.execute('''
                SELECT cs.session_id, cs.created_at, cs.last_accessed,
                       (SELECT content FROM chat_messages WHERE session_id = cs.session_id AND role = 'user' ORDER BY timestamp ASC LIMIT 1) as first_message
                FROM chat_sessions cs
                WHERE cs.user_id = ?
                ORDER BY cs.last_accessed DESC
            ''', (user_id,))
            
            sessions = []
            for row in cursor.fetchall():
                session_id, created_at, last_accessed, first_message = row
                # Create a preview from first user message (truncate if too long)
                preview = first_message[:60] + "..." if first_message and len(first_message) > 60 else (first_message or "New Chat")
                
                sessions.append({
                    "session_id": session_id,
                    "created_at": created_at,
                    "last_accessed": last_accessed,
                    "preview": preview
                })
            
            conn.close()
            logger.debug(f"Retrieved {len(sessions)} sessions for user {user_id}")
            return sessions
        except sqlite3.Error as e:
            logger.error(f"Error retrieving sessions for user {user_id}: {e}", exc_info=True)
            return []

def get_chat_history(session_id: str) -> List[dict]:
    """Retrieves chat history for a session formatted for Gemini API, checks timeout."""
    _prune_old_sessions() # Prune before fetching
    history_for_api = []
    with db_lock:
         try:
            conn = sqlite3.connect(DB_NAME, check_same_thread=False)
            cursor = conn.cursor()
            # First check if session still exists (not timed out)
            cursor.execute("SELECT 1 FROM chat_sessions WHERE session_id = ?", (session_id,))
            session_exists = cursor.fetchone()

            if not session_exists:
                logger.warning(f"Session {session_id} not found or expired during history retrieval.")
                conn.close()
                return [] # Return empty history if session expired or never existed

            # Session exists, fetch messages
            cursor.execute('''
                SELECT role, content, file_references
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            ''', (session_id,))
            rows = cursor.fetchall()
            conn.close()

            # Format for Gemini API's 'contents' list
            for role, content, files_json in rows:
                parts = [{"text": content}] # Always include text part
                if files_json:
                    file_refs = json.loads(files_json)
                    # IMPORTANT: Adjust mime_type based on actual uploaded file type if supporting non-PDFs
                    # Assuming file_refs contains URIs like 'files/xxxx'
                    file_parts = [{"file_data": {"mime_type": "application/pdf", "file_uri": ref}} for ref in file_refs]
                    # Prepend file parts before text for better context according to Gemini docs
                    parts = file_parts + parts

                history_for_api.append({"role": role, "parts": parts})
            logger.debug(f"Retrieved {len(history_for_api)} messages for session {session_id}.")

         except sqlite3.Error as e:
             logger.error(f"Error retrieving history for session {session_id}: {e}", exc_info=True)
    return history_for_api

# Initialize DB when module is loaded
# Ensure this runs only once, e.g., during app startup instead
# init_db() # Call this explicitly in app.py startup instead
