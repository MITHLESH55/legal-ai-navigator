import os
import uuid
import logging
import tempfile
import asyncio
import threading
import pickle
import json
import io
import time 
from pathlib import Path
from typing import AsyncIterator, Dict, Any, List, Optional
from flask import Flask, request, jsonify, redirect, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from werkzeug.utils import secure_filename

# Import nest_asyncio to allow nested event loops
try:
    import nest_asyncio
    nest_asyncio.apply()
    _nest_asyncio_available = True
except ImportError:
    _nest_asyncio_available = False
    print("Warning: nest_asyncio not installed. Run 'pip install nest-asyncio' for better async support.")

load_dotenv()


os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

from livekit import api
from livekit.api import LiveKitAPI, ListRoomsRequest
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import google.generativeai as genai # For file API status check

# Import our shared logic and new data stores
from agent_logic import agent_executor
from data_stores import vector_store, graph_store, load_and_split_pdf
import graph_builder
from drive_tool import _get_drive_service, SCOPES, TOKEN_PICKLE_PATH

# --- NEW Imports for /chat-g ---
from gemini_chat_logic import agent_executor_g, upload_file_to_gemini, TOOLS_REQUIRING_APPROVAL_G, invoke_gemini_with_files_directly
from db_utils import init_db, save_chat_message, get_chat_history, add_or_update_session, get_session_files, get_user_sessions # Import DB functions
from langgraph.types import Command # Import Command
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage # Import message types

load_dotenv()

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})



# --- Global event loop for all async operations ---
_global_loop = None
_global_loop_lock = threading.Lock()

# --- Async Route Decorator for Flask ---
def async_route(f):
    """
    Decorator to run async functions in Flask routes using a GLOBAL persistent event loop.
    
    This is necessary because Flask doesn't natively support async routes like FastAPI.
    The grpc library (used by Google Gemini) creates tasks that must ALL stay on the SAME event loop.
    
    By using a single global event loop running in a background thread, we ensure:
    1. All grpc tasks are created on the same loop
    2. No "attached to a different loop" errors occur
    3. The loop persists across multiple requests
    
    Usage:
        @app.route("/my-route", methods=["POST"])
        @async_route
        async def my_handler():
            # Your async code here
            pass
    """
    from functools import wraps
    
    @wraps(f)
    def wrapper(*args, **kwargs):
        global _global_loop
        
        # Initialize the global event loop on first use
        with _global_loop_lock:
            if _global_loop is None:
                logger.debug("Initializing global event loop in background thread...")
                _global_loop = asyncio.new_event_loop()
                
                # Start the event loop in a background thread
                def run_loop():
                    asyncio.set_event_loop(_global_loop)
                    _global_loop.run_forever()
                
                loop_thread = threading.Thread(target=run_loop, daemon=True)
                loop_thread.start()
                logger.debug("Global event loop started successfully")
        
        # Submit the async function to the global event loop
        future = asyncio.run_coroutine_threadsafe(f(*args, **kwargs), _global_loop)
        
        # Wait for the result
        return future.result()
    
    return wrapper


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Authentication Imports ---
import bcrypt
import jwt
import smtplib
import random
import string
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from functools import wraps
from db_utils import get_db_connection

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production-please")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Email Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_EMAIL = "devanku411@gmail.com"
SMTP_PASSWORD = "qovwbrbopzsghezk"

# --- Initialize Database ---
try:
    init_db()
    logger.info("Chat-G Database Initialized.")
except Exception as e:
    logger.critical(f"Failed to initialize Chat-G database: {e}. Endpoint /chat-g might fail.")
    # Decide if the app should stop here. For now, it will continue but /chat-g will likely error.


# --- Authentication Helper Functions ---

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


def generate_otp() -> str:
    """Generate a 6-digit OTP."""
    return ''.join(random.choices(string.digits, k=6))


def create_jwt_token(user_id: int, email: str) -> str:
    """Create a JWT token for authentication."""
    payload = {
        'user_id': user_id,
        'email': email,
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify a JWT token and return the payload."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired")
        return None
    except jwt.InvalidTokenError:
        logger.warning("Invalid JWT token")
        return None


def send_email(to_email: str, subject: str, html_body: str) -> bool:
    """Send an email using Gmail SMTP."""
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = SMTP_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        
        # Attach HTML body
        html_part = MIMEText(html_body, 'html')
        msg.attach(html_part)
        
        # Connect to SMTP server and send
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"Email sent successfully to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {e}")
        return False


def send_otp_email(email: str, otp: str) -> bool:
    """Send OTP email for password reset."""
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .container {{
                background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);
                backdrop-filter: blur(20px);
                border-radius: 20px;
                padding: 40px;
                border: 2px solid rgba(12, 17, 91, 0.1);
                box-shadow: 0 8px 32px rgba(12, 17, 91, 0.15);
            }}
            .header {{
                text-align: center;
                color: #0C115B;
                margin-bottom: 30px;
            }}
            .otp-box {{
                background: rgba(12, 17, 91, 0.05);
                border: 2px solid #0C115B;
                border-radius: 12px;
                padding: 30px;
                text-align: center;
                margin: 30px 0;
            }}
            .otp-code {{
                font-size: 48px;
                font-weight: bold;
                color: #0C115B;
                letter-spacing: 8px;
                margin: 20px 0;
                font-family: 'Courier New', monospace;
            }}
            .warning {{
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px;
                margin: 20px 0;
                border-radius: 8px;
            }}
            .footer {{
                text-align: center;
                color: #666;
                font-size: 14px;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid rgba(12, 17, 91, 0.1);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔐 Password Reset Request</h1>
                <p>Legal Navigator</p>
            </div>
            
            <p>Hello,</p>
            <p>You requested to reset your password. Use the OTP code below to complete the process:</p>
            
            <div class="otp-box">
                <p>Your OTP Code:</p>
                <div class="otp-code">{otp}</div>
            </div>
            
            <div class="warning">
                <strong>⏰ Important:</strong> This OTP will expire in 10 minutes for security reasons.
            </div>
            
            <p>If you didn't request this password reset, please ignore this email. Your password will remain unchanged.</p>
            
            <div class="footer">
                <p>This is an automated message from Legal Navigator</p>
                <p>© 2024 Legal Navigator. All rights reserved.</p>
            </div>
        </div>
    </body>
    </html>
    """
    return send_email(email, "Password Reset OTP - Legal Navigator", html_body)


def send_welcome_email(email: str, name: str) -> bool:
    """Send welcome email for new registration."""
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .container {{
                background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);
                backdrop-filter: blur(20px);
                border-radius: 20px;
                padding: 40px;
                border: 2px solid rgba(12, 17, 91, 0.1);
                box-shadow: 0 8px 32px rgba(12, 17, 91, 0.15);
            }}
            .header {{
                text-align: center;
                color: #0C115B;
                margin-bottom: 30px;
            }}
            .welcome-box {{
                background: rgba(12, 17, 91, 0.05);
                border-radius: 12px;
                padding: 30px;
                text-align: center;
                margin: 30px 0;
            }}
            .button {{
                display: inline-block;
                background: #0C115B;
                color: white;
                padding: 15px 40px;
                border-radius: 10px;
                text-decoration: none;
                font-weight: bold;
                margin: 20px 0;
            }}
            .features {{
                background: white;
                border-radius: 12px;
                padding: 20px;
                margin: 20px 0;
            }}
            .feature-item {{
                margin: 15px 0;
                padding-left: 30px;
                position: relative;
            }}
            .feature-item:before {{
                content: "✓";
                position: absolute;
                left: 0;
                color: #0C115B;
                font-weight: bold;
                font-size: 20px;
            }}
            .footer {{
                text-align: center;
                color: #666;
                font-size: 14px;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid rgba(12, 17, 91, 0.1);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎉 Welcome to Legal Navigator!</h1>
            </div>
            
            <div class="welcome-box">
                <h2>Hi {name}! 👋</h2>
                <p style="font-size: 18px;">Your account has been created successfully!</p>
            </div>
            
            <p>Thank you for joining Legal Navigator. We're excited to have you on board!</p>
            
            <div class="features">
                <h3 style="color: #0C115B;">What you can do with Legal Navigator:</h3>
                <div class="feature-item">AI-powered legal research and analysis</div>
                <div class="feature-item">Intelligent case law search and citations</div>
                <div class="feature-item">Document analysis and summarization</div>
                <div class="feature-item">Real-time legal updates and notifications</div>
                <div class="feature-item">Integrated browser for comprehensive research</div>
            </div>
            
            <div style="text-align: center;">
                <a href="http://localhost:5525" class="button">Get Started →</a>
            </div>
            
            <p style="margin-top: 30px;">If you have any questions or need assistance, our support team is here to help!</p>
            
            <div class="footer">
                <p>This is an automated message from Legal Navigator</p>
                <p>© 2024 Legal Navigator. All rights reserved.</p>
            </div>
        </div>
    </body>
    </html>
    """
    return send_email(email, "Welcome to Legal Navigator! 🎉", html_body)


def require_auth(f):
    """Decorator to require authentication for routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'No token provided'}), 401
        
        token = auth_header.split(' ')[1]
        payload = verify_jwt_token(token)
        
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Add user info to request context
        request.user_id = payload['user_id']
        request.user_email = payload['email']
        
        return f(*args, **kwargs)
    
    return decorated_function


# --- Authentication Endpoints ---

@app.route("/api/auth/register", methods=["POST"])
def register():
    """Register a new user."""
    try:
        data = request.json
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        name = data.get('name', '').strip()
        
        # Validation
        if not email or not password or not name:
            return jsonify({'error': 'Email, password, and name are required'}), 400
        
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters long'}), 400
        
        # Check if user already exists
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT user_id FROM users WHERE email = ?', (email,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            conn.close()
            return jsonify({'error': 'Email already registered'}), 409
        
        # Hash password and create user
        password_hash = hash_password(password)
        cursor.execute(
            'INSERT INTO users (email, password_hash, created_at, is_verified) VALUES (?, ?, ?, ?)',
            (email, password_hash, datetime.utcnow().isoformat(), 1)  # Auto-verify for now
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        
        # Send welcome email
        send_welcome_email(email, name)
        
        logger.info(f"New user registered: {email}")
        
        return jsonify({
            'message': 'Registration successful',
            'user': {
                'user_id': user_id,
                'email': email,
                'name': name
            }
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500


@app.route("/api/auth/login", methods=["POST"])
def login():
    """Login a user and return JWT token."""
    try:
        data = request.json
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        # Find user
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT user_id, email, password_hash FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return jsonify({'error': 'Invalid email or password'}), 401
        
        user_id, user_email, password_hash = user
        
        # Verify password
        if not verify_password(password, password_hash):
            conn.close()
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Update last login
        cursor.execute('UPDATE users SET last_login = ? WHERE user_id = ?',
                      (datetime.utcnow().isoformat(), user_id))
        conn.commit()
        conn.close()
        
        # Create JWT token
        token = create_jwt_token(user_id, user_email)
        
        logger.info(f"User logged in: {email}")
        
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': {
                'user_id': user_id,
                'email': user_email
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500


@app.route("/api/auth/forgot-password", methods=["POST"])
def forgot_password():
    """Send OTP for password reset."""
    try:
        data = request.json
        email = data.get('email', '').strip().lower()
        
        if not email:
            return jsonify({'error': 'Email is required'}), 400
        
        # Check if user exists
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT user_id FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        
        if not user:
            # Don't reveal if email exists or not for security
            return jsonify({'message': 'If the email exists, an OTP has been sent'}), 200
        
        user_id = user[0]
        
        # Generate OTP
        otp = generate_otp()
        expires_at = datetime.utcnow() + timedelta(minutes=10)
        
        # Store OTP in database
        cursor.execute(
            'INSERT INTO password_reset_tokens (user_id, otp_code, created_at, expires_at, is_used) VALUES (?, ?, ?, ?, ?)',
            (user_id, otp, datetime.utcnow().isoformat(), expires_at.isoformat(), 0)
        )
        conn.commit()
        conn.close()
        
        # Send OTP email
        if send_otp_email(email, otp):
            logger.info(f"OTP sent to: {email}")
            return jsonify({'message': 'OTP sent to your email'}), 200
        else:
            return jsonify({'error': 'Failed to send OTP email'}), 500
        
    except Exception as e:
        logger.error(f"Forgot password error: {e}")
        return jsonify({'error': 'Failed to process request'}), 500


@app.route("/api/auth/reset-password", methods=["POST"])
def reset_password():
    """Reset password using OTP."""
    try:
        data = request.json
        email = data.get('email', '').strip().lower()
        otp = data.get('otp', '').strip()
        new_password = data.get('newPassword', '')
        
        if not email or not otp or not new_password:
            return jsonify({'error': 'Email, OTP, and new password are required'}), 400
        
        if len(new_password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters long'}), 400
        
        # Find user
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT user_id FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return jsonify({'error': 'Invalid email or OTP'}), 401
        
        user_id = user[0]
        
        # Verify OTP
        cursor.execute(
            '''SELECT token_id FROM password_reset_tokens 
               WHERE user_id = ? AND otp_code = ? AND is_used = 0 AND expires_at > ?
               ORDER BY created_at DESC LIMIT 1''',
            (user_id, otp, datetime.utcnow().isoformat())
        )
        token = cursor.fetchone()
        
        if not token:
            conn.close()
            return jsonify({'error': 'Invalid or expired OTP'}), 401
        
        token_id = token[0]
        
        # Update password
        password_hash = hash_password(new_password)
        cursor.execute('UPDATE users SET password_hash = ? WHERE user_id = ?',
                      (password_hash, user_id))
        
        # Mark OTP as used
        cursor.execute('UPDATE password_reset_tokens SET is_used = 1 WHERE token_id = ?',
                      (token_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Password reset successful for: {email}")
        
        return jsonify({'message': 'Password reset successful'}), 200
        
    except Exception as e:
        logger.error(f"Reset password error: {e}")
        return jsonify({'error': 'Password reset failed'}), 500


@app.route("/api/sessions", methods=["GET"])
def get_sessions():
    """Get all chat sessions for the authenticated user."""
    try:
        # Extract user_id from JWT token
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'No token provided'}), 401
        
        token = auth_header.split(' ')[1]
        payload = verify_jwt_token(token)
        
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        user_id = payload['user_id']
        
        # Get user's sessions from database
        sessions = get_user_sessions(user_id)
        
        return jsonify({
            'sessions': sessions
        }), 200
        
    except Exception as e:
        logger.error(f"Get sessions error: {e}")
        return jsonify({'error': 'Failed to retrieve sessions'}), 500


@app.route("/api/sessions/<session_id>/messages", methods=["GET"])
def get_session_messages(session_id):
    """Get all messages for a specific chat session."""
    try:
        # Extract user_id from JWT token
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'No token provided'}), 401
        
        token = auth_header.split(' ')[1]
        payload = verify_jwt_token(token)
        
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Get chat history for this session
        messages = get_chat_history(session_id)
        
        return jsonify({
            'messages': messages,
            'session_id': session_id
        }), 200
        
    except Exception as e:
        logger.error(f"Get session messages error: {e}")
        return jsonify({'error': 'Failed to retrieve messages'}), 500


@app.route("/api/auth/google", methods=["POST"])
def google_auth_exchange():
    """Exchange Google OAuth info for JWT token. Creates user if doesn't exist."""
    try:
        data = request.json
        email = data.get('email')
        google_id = data.get('id')
        name = data.get('name', '')
        
        if not email or not google_id:
            return jsonify({'error': 'Email and ID required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute('SELECT user_id FROM users WHERE email = ?', (email,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            user_id = existing_user[0]
            # Update last login
            cursor.execute('UPDATE users SET last_login = ? WHERE user_id = ?', 
                         (datetime.now().isoformat(), user_id))
        else:
            # Create new user with Google ID as password (they won't use password login)
            user_id = str(uuid.uuid4())
            password_hash = bcrypt.hashpw(google_id.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            cursor.execute('''
                INSERT INTO users (user_id, email, password_hash, created_at, last_login, is_verified)
                VALUES (?, ?, ?, ?, ?, 1)
            ''', (user_id, email, password_hash, datetime.now().isoformat(), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        # Generate JWT token
        token = create_jwt_token(user_id, email)
        
        return jsonify({
            'token': token,
            'user': {
                'user_id': user_id,
                'email': email,
                'name': name
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Google auth exchange error: {e}")
        return jsonify({'error': 'Authentication failed'}), 500


# --- Pydantic Models for LangGraph Server API Endpoints ---

class StreamLogRequest(BaseModel):
    """Request model for /stream_log endpoint."""
    input: Dict[str, Any] = Field(..., description="Input data for the agent")
    config: Dict[str, Any] = Field(..., description="Configuration including thread_id")
    stream_mode: List[str] = Field(default=["values"], description="Stream modes to use")
    include_names: Optional[List[str]] = Field(default=None, description="Node names to include")
    include_types: Optional[List[str]] = Field(default=None, description="Types to include")
    include_tags: Optional[List[str]] = Field(default=None, description="Tags to include")
    exclude_names: Optional[List[str]] = Field(default=None, description="Node names to exclude")
    exclude_types: Optional[List[str]] = Field(default=None, description="Types to exclude")
    exclude_tags: Optional[List[str]] = Field(default=None, description="Tags to exclude")


class UpdateStateRequest(BaseModel):
    """Request model for /update_state endpoint."""
    config: Dict[str, Any] = Field(..., description="Configuration including thread_id")
    values: Dict[str, Any] = Field(..., description="State values to update")
    as_node: Optional[str] = Field(default=None, description="Update as if this node made the update")


class InvokeRequest(BaseModel):
    """Request model for /invoke endpoint."""
    input: Dict[str, Any] = Field(..., description="Input data for the agent")
    config: Dict[str, Any] = Field(..., description="Configuration including thread_id")


class StreamRequest(BaseModel):
    """Request model for /stream endpoint."""
    input: Dict[str, Any] = Field(..., description="Input data for the agent")
    config: Dict[str, Any] = Field(..., description="Configuration including thread_id")
    stream_mode: List[str] = Field(default=["values"], description="Stream modes to use")


# --- LangGraph Server API Endpoints ---

@app.route("/stream_log", methods=["POST"])
@async_route
async def stream_log():
    """
    Stream intermediate steps and state patches from the agent execution.
    
    This endpoint streams JSON patches in Server-Sent Events (SSE) format.
    Each patch represents a change in the agent's state during execution.
    """
    try:
        data = request.json
        
        # Validate input using Pydantic
        try:
            req = StreamLogRequest(**data)
        except ValidationError as e:
            logger.error(f"Validation error in /stream_log: {e}")
            return jsonify({"error": "Invalid input", "details": e.errors()}), 400
        
        # Extract thread_id from config
        thread_id = req.config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return jsonify({"error": "thread_id is required in config.configurable"}), 400
        
        logger.info(f"Stream log request for thread: {thread_id}")
        
        # Prepare the input for the agent
        input_data = req.input
        config = req.config
        
        async def generate_sse():
            """Generate Server-Sent Events."""
            try:
                async for chunk in agent_executor.astream(
                    input_data,
                    config=config,
                    stream_mode=req.stream_mode
                ):
                    # Format as SSE: data: {json}\n\n
                    event_data = json.dumps(chunk, default=str)
                    yield f"data: {event_data}\n\n"
                
                # Send completion event
                yield f"data: {json.dumps({'event': 'end'})}\n\n"
                
            except Exception as e:
                logger.error(f"Error during streaming: {e}", exc_info=True)
                error_data = json.dumps({"event": "error", "data": {"error": str(e)}})
                yield f"data: {error_data}\n\n"
        
        # Return SSE response
        return Response(
            generate_sse(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in /stream_log endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/get_state", methods=["GET"])
@async_route
async def get_state():
    """
    Get the current state of a thread.
    
    Query Parameters:
        thread_id (str): The thread ID to retrieve state for
        checkpoint_id (str, optional): Specific checkpoint ID to retrieve
    
    Returns:
        JSON representation of the state snapshot
    """
    try:
        thread_id = request.args.get("thread_id")
        checkpoint_id = request.args.get("checkpoint_id")
        
        if not thread_id:
            return jsonify({"error": "thread_id query parameter is required"}), 400
        
        logger.info(f"Get state request for thread: {thread_id}, checkpoint: {checkpoint_id}")
        
        # Build config
        config = {"configurable": {"thread_id": thread_id}}
        if checkpoint_id:
            config["configurable"]["checkpoint_id"] = checkpoint_id
        
        # Get state from agent executor
        state_snapshot = agent_executor.get_state(config)
        
        if not state_snapshot:
            logger.warning(f"No state found for thread {thread_id}")
            return jsonify({
                "values": {},
                "next": [],
                "config": config,
                "metadata": {},
                "created_at": None,
                "parent_config": None,
                "tasks": []
            }), 404
        
        # Helper function to serialize LangChain messages
        def serialize_value(value):
            """Recursively serialize values, converting LangChain objects to dicts."""
            if hasattr(value, '__dict__') and hasattr(value, 'type'):
                # LangChain message object
                return {
                    "type": getattr(value, "type", "unknown"),
                    "content": value.content if hasattr(value, "content") else str(value),
                    "id": getattr(value, "id", None),
                    "tool_calls": getattr(value, "tool_calls", None),
                }
            elif isinstance(value, list):
                return [serialize_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            else:
                return value
        
        # Convert StateSnapshot to dict for JSON response
        response_data = {
            "values": serialize_value(state_snapshot.values),
            "next": list(state_snapshot.next) if state_snapshot.next else [],
            "config": state_snapshot.config,
            "metadata": state_snapshot.metadata,
            "created_at": state_snapshot.created_at if isinstance(state_snapshot.created_at, str) else (
                state_snapshot.created_at.isoformat() if state_snapshot.created_at else None
            ),
            "parent_config": state_snapshot.parent_config,
            "tasks": [{"id": t.id, "name": t.name} for t in state_snapshot.tasks] if state_snapshot.tasks else []
        }
        
        logger.info(f"Successfully retrieved state for thread {thread_id}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in /get_state endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/update_state", methods=["POST"])
@async_route
async def update_state():
    """
    Update the state of a thread.
    
    Request Body:
        config (dict): Configuration including thread_id
        values (dict): State values to update
        as_node (str, optional): Update as if this node made the update
    
    Returns:
        Updated config with new checkpoint_id
    """
    try:
        data = request.json
        
        # Validate input
        try:
            req = UpdateStateRequest(**data)
        except ValidationError as e:
            logger.error(f"Validation error in /update_state: {e}")
            return jsonify({"error": "Invalid input", "details": e.errors()}), 400
        
        thread_id = req.config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return jsonify({"error": "thread_id is required in config.configurable"}), 400
        
        logger.info(f"Update state request for thread: {thread_id}")
        
        # Update the state using agent executor
        kwargs = {"values": req.values}
        if req.as_node:
            kwargs["as_node"] = req.as_node
        
        updated_config = agent_executor.update_state(req.config, **kwargs)
        
        logger.info(f"Successfully updated state for thread {thread_id}")
        
        return jsonify({
            "success": True,
            "config": updated_config
        })
        
    except Exception as e:
        logger.error(f"Error in /update_state endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/invoke", methods=["POST"])
@async_route
async def invoke():
    """
    Invoke the agent and return the final state.
    
    Request Body:
        input (dict): Input data for the agent
        config (dict): Configuration including thread_id
    
    Returns:
        Final state after execution
    """
    try:
        data = request.json
        
        # Validate input
        try:
            req = InvokeRequest(**data)
        except ValidationError as e:
            logger.error(f"Validation error in /invoke: {e}")
            return jsonify({"error": "Invalid input", "details": e.errors()}), 400
        
        thread_id = req.config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return jsonify({"error": "thread_id is required in config.configurable"}), 400
        
        logger.info(f"Invoke request for thread: {thread_id}")
        
        # Invoke the agent
        result = await agent_executor.ainvoke(req.input, config=req.config)
        
        logger.info(f"Successfully invoked agent for thread {thread_id}")
        
        # Convert result to JSON-serializable format
        # LangChain message objects need to be converted to dicts
        serializable_result = {}
        for key, value in result.items():
            if key == "messages":
                # Convert message objects to dicts
                serializable_result[key] = [
                    {
                        "role": getattr(msg, "type", "unknown"),
                        "content": msg.content if hasattr(msg, "content") else str(msg),
                        "id": getattr(msg, "id", None),
                        "tool_calls": getattr(msg, "tool_calls", None),
                    }
                    for msg in value
                ]
            else:
                serializable_result[key] = value
        
        return jsonify(serializable_result)
        
    except Exception as e:
        logger.error(f"Error in /invoke endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/stream", methods=["POST"])
@async_route
async def stream():
    """
    Stream only the final outputs from the agent execution.
    
    Request Body:
        input (dict): Input data for the agent
        config (dict): Configuration including thread_id
        stream_mode (list): Stream modes (default: ["values"])
    
    Returns:
        SSE stream of state updates
    """
    try:
        data = request.json
        
        # Validate input
        try:
            req = StreamRequest(**data)
        except ValidationError as e:
            logger.error(f"Validation error in /stream: {e}")
            return jsonify({"error": "Invalid input", "details": e.errors()}), 400
        
        thread_id = req.config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return jsonify({"error": "thread_id is required in config.configurable"}), 400
        
        logger.info(f"Stream request for thread: {thread_id}")
        
        async def generate_sse() -> AsyncIterator[str]:
            """Generate Server-Sent Events from astream."""
            try:
                async for chunk in agent_executor.astream(
                    req.input,
                    config=req.config,
                    stream_mode=req.stream_mode
                ):
                    # Format as SSE
                    event_data = json.dumps(chunk, default=str)
                    yield f"data: {event_data}\n\n"
                
                # Send completion event
                yield f"data: {json.dumps({'event': 'end'})}\n\n"
                
            except Exception as e:
                logger.error(f"Error during streaming for thread {thread_id}: {e}", exc_info=True)
                error_data = json.dumps({"error": str(e), "event": "error"})
                yield f"data: {error_data}\n\n"
        
        # Return SSE response
        return Response(
            generate_sse(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in /stream endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# --- LiveKit Token Endpoint (Async) ---

async def get_rooms():
    lk_api = LiveKitAPI(os.getenv("LIVEKIT_URL"), os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET"))
    rooms = await lk_api.room.list_rooms(ListRoomsRequest())
    await lk_api.aclose()
    return [room.name for room in rooms.rooms]

async def generate_room_name():
    name = "room-" + str(uuid.uuid4())[:8]
    rooms = await get_rooms()
    while name in rooms:
        name = "room-" + str(uuid.uuid4())[:8]
    return name

@app.route("/get_token", methods=["GET"])
@async_route
async def get_token():
    """Generates a LiveKit access token."""
    name = request.args.get("name", "user-" + str(uuid.uuid4())[:4])
    room = request.args.get("room", None)
    
    if not room:
        room = await generate_room_name()
        
    token = api.AccessToken(os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET")) \
        .with_identity(name)\
        .with_name(name)\
        .with_grants(api.VideoGrants(room_join=True, room=room))
    
    logger.info(f"Generated token for user {name} in room {room}")
    return jsonify({"token": token.to_jwt(), "room_name": room})

# --- File Upload Endpoint (with Background Graph Building) ---

def background_graph_builder(documents: list, case_id: str, file_name: str):
    """Worker function to run graph extraction in a separate thread."""
    with app.app_context(): # To access app context in thread
        logger.info(f"[Background Thread] Starting graph extraction for {file_name}")
        try:
            graph_builder.extract_and_store_graph(documents, case_id, file_name)
            logger.info(f"[Background Thread] Finished graph extraction for {file_name}")
        except Exception as e:
            logger.error(f"[Background Thread] Error during graph extraction for {file_name}: {e}")

@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Handles file uploads, indexes them in Qdrant (vectors),
    and starts a background task to build a Knowledge Graph in Neo4j.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    case_id = request.form.get("case_id", f"case_{uuid.uuid4().hex[:8]}")
    
    if file.filename == '' or not file.filename.endswith('.pdf'):
        return jsonify({"error": "No file selected or invalid type (must be .pdf)"}), 400
    
    if not vector_store or not graph_store:
        return jsonify({"error": "Data stores are not initialized"}), 500

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_f:
            file.save(temp_f.name)
            temp_path = temp_f.name
        
        logger.info(f"File saved to temp path: {temp_path}")
        
        # 1. Load and split the PDF (this is fast)
        documents = load_and_split_pdf(temp_path)
        if not documents:
            return jsonify({"error": "Failed to load or split PDF"}), 500

        # 2. Index in Qdrant (Vector RAG - this is relatively fast)
        vector_store.index_documents(documents, case_id, file.filename)
        
        # --- 3. Check page count and conditionally start graph building ---
        max_pages = int(os.getenv("MAX_DOCUMENT_PAGES", "20"))
        
        # Determine the number of pages in the document
        num_pages = 0
        if documents:
            for doc in documents:
                page_num = doc.metadata.get("page", 0)
                if page_num > num_pages:
                    num_pages = page_num
        
        logger.info(f"Document '{file.filename}' has {num_pages} pages. Max allowed: {max_pages}")
        
        # Conditionally start graph building based on page count
        if num_pages <= max_pages:
            # Start Knowledge Graph building in a background thread
            graph_thread = threading.Thread(
                target=background_graph_builder,
                args=(documents, case_id, file.filename)
            )
            graph_thread.start()
            
            message = "File accepted. Vector indexing complete. Graph building running in background (batched)."
            logger.info(f"Hybrid indexing started for {file.filename} (case_id: {case_id})")
        else:
            message = "File accepted. Vector indexing complete. Graph building skipped (document exceeds page limit)."
            logger.warning(f"Graph building skipped for {file.filename} - exceeds {max_pages} page limit")
        
        # Return 202 Accepted: The request has been accepted for processing,
        # but the processing has not been completed.
        return jsonify({
            "message": message,
            "filename": file.filename,
            "case_id": case_id,
            "chunks_found": len(documents),
            "pages": num_pages,
            "graph_building": num_pages <= max_pages
        }), 202
            
    except Exception as e:
        logger.error(f"Error during file upload: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# --- Text Chatbot Endpoint (Async) ---

@app.route("/chat", methods=["POST"])
@async_route
async def chat():
    """Handles text-based chat queries using the central agent graph with memory."""
    data = request.json
    query = data.get("query")
    case_id = data.get("case_id")  # Can be None if it's a general query
    session_id = data.get("session_id")  # For conversation memory
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # Generate session_id if not provided (for memory tracking)
    if not session_id:
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        logger.info(f"Generated new session_id: {session_id}")
        
    logger.info(f"Chat query received. Query: '{query}', Case ID: '{case_id}', Session: '{session_id}'")

    try:
        # Pass config with thread_id for memory persistence
        config = {"configurable": {"thread_id": session_id}}
        
        # Use .ainvoke() method with config for memory
        response = await agent_executor.ainvoke({
            "input": query,
            "case_id": case_id or "",  # Empty string if None
            "messages": []  # Initialize empty messages list
        }, config)
        
        # Check if there's an interrupt (human-in-the-loop)
        if "__interrupt__" in response:
            interrupt_data = response["__interrupt__"]
            logger.info(f"Interrupt detected: {interrupt_data}")
            
            # Check if it's a tool approval request
            if isinstance(interrupt_data, list) and len(interrupt_data) > 0:
                # Access the Interrupt object's value attribute
                interrupt_obj = interrupt_data[0]
                interrupt_value = interrupt_obj.value if hasattr(interrupt_obj, 'value') else interrupt_obj
                
                if isinstance(interrupt_value, dict) and interrupt_value.get("type") == "tool_approval_request":
                    # Return approval request to frontend
                    tools = interrupt_value.get("tools", [])
                    return jsonify({
                        "response": interrupt_value.get("message", "Tool approval required"),
                        "needs_approval": True,
                        "approval_requests": tools,
                        "session_id": session_id,
                        "interrupt_type": "tool_approval"
                    })
                else:
                    # Generic clarification request (from court supervisor)
                    question = interrupt_value.get("question", "Please provide more information.") if isinstance(interrupt_value, dict) else str(interrupt_value)
                    return jsonify({
                        "response": question,
                        "needs_clarification": True,
                        "session_id": session_id,
                        "interrupt_data": str(interrupt_data)
                    })
        
        # Get the final answer from the last AI message
        messages = response.get("messages", [])
        if messages:
            # Get the last message (should be the AI's response)
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                final_answer = last_message.content
            else:
                final_answer = str(last_message)
        else:
            final_answer = "Error: Could not get final answer from the agent."
        
        # Check if the response is a browser redirect signal
        try:
            action_data = json.loads(final_answer)
            if isinstance(action_data, dict) and action_data.get("action") == "BROWSER_REDIRECT":
                logger.info(f"Browser redirect detected: {action_data.get('url')}")
                return jsonify({
                    "action": "BROWSER_REDIRECT",
                    "url": action_data.get("url"),
                    "session_id": session_id
                })
        except (json.JSONDecodeError, TypeError):
            # Not a JSON response, proceed normally
            pass
        
        return jsonify({
            "response": final_answer,
            "session_id": session_id
        })
        
    except Exception as e:
        logger.error(f"Error during chat agent invocation: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/chat/approve", methods=["POST"])
@async_route
async def approve_tool():
    """
    Handle user approval/rejection of tool execution from BOTH agent types.
    Resumes the interrupted agent with the user's decision.
    """
    try:
        data = request.json
        session_id = data.get("session_id")
        approved = data.get("approved", False)  # Boolean: True = approve, False = reject
        # Check which agent needs resuming based on session_id prefix or passed flag
        agent_type = data.get("agent_type", "original") # Expect "gemini" or "original"
        
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
        
        logger.info(f"Tool approval response: session={session_id}, approved={approved}, agent={agent_type}")
        
        # Choose executor
        if agent_type == "gemini":
            executor_to_resume = agent_executor_g
            config = {"configurable": {"thread_id": session_id}}
            logger.info(f"Resuming Gemini agent ({session_id}) with approval={approved}")
        else: # Original agent
            executor_to_resume = agent_executor
            config = {"configurable": {"thread_id": session_id}}
            logger.info(f"Resuming Original agent ({session_id}) with approval={approved}")
        
        # Resume the agent with the approval decision
        response = await executor_to_resume.ainvoke(
            Command(resume={"approved": approved}),
            config
        )
        
        # Get the final answer after resumption
        messages = response.get("messages", [])
        final_answer = ""
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                final_answer = last_message.content
            else:
                final_answer = str(last_message)
        else:
            final_answer = "Tool execution completed." if approved else "Action cancelled."
        
        # --- Browser Redirect Check ---
        try:
            action_data = json.loads(final_answer)
            if isinstance(action_data, dict) and action_data.get("action") == "BROWSER_REDIRECT":
                logger.info(f"/chat/approve redirect: {action_data.get('url')}")
                return jsonify({
                    "action": "BROWSER_REDIRECT",
                    "url": action_data.get("url"),
                    "session_id": session_id
                })
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Save final response to DB for Gemini agent if it was resumed
        if agent_type == "gemini":
            save_chat_message(session_id, 'model', final_answer)
        
        return jsonify({
            "response": final_answer,
            "session_id": session_id
        })
        
    except Exception as e:
        logger.error(f"Error handling tool approval for session {session_id}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to process approval: {str(e)}"}), 500


# --- Background Old Pipeline Function ---
def run_old_pipeline_background(temp_file_path: str, case_id: str, original_file_name: str):
    """Runs the existing indexing pipeline in a background thread."""
    logger.info(f"[Background Old Pipeline] Thread starting for {original_file_name} (case: {case_id}) using temp file {temp_file_path}")
    thread_temp_path = temp_file_path # Use the path passed to the thread
    try:
        # Check data stores again within the thread context if necessary
        if not vector_store or not graph_store:
            logger.error("[Background Old Pipeline] Data stores not available in thread. Cannot index.")
            return

        logger.info(f"[Background Old Pipeline] Loading and splitting PDF from {thread_temp_path}...")
        documents = load_and_split_pdf(thread_temp_path)
        if not documents:
            logger.error(f"[Background Old Pipeline] Failed to load/split PDF: {original_file_name}")
            return

        logger.info(f"[Background Old Pipeline] Indexing {len(documents)} chunks in Qdrant for {original_file_name}...")
        vector_store.index_documents(documents, case_id, original_file_name)
        logger.info("[Background Old Pipeline] Qdrant indexing complete.")
        
        # Verify indexing by attempting a simple search
        try:
            test_results = vector_store.search(query="test", case_id=case_id, k=1)
            logger.info(f"[Background Old Pipeline] Verification: Found {len(test_results)} results for case_id={case_id}")
        except Exception as verify_err:
            logger.warning(f"[Background Old Pipeline] Could not verify indexing: {verify_err}")

        logger.info("[Background Old Pipeline] Starting graph building check...")
        # graph_builder handles page limits internally
        graph_builder.extract_and_store_graph(documents, case_id, original_file_name)

        logger.info(f"[Background Old Pipeline] Finished indexing tasks for {original_file_name}")

    except Exception as e:
        logger.error(f"[Background Old Pipeline] Error during indexing for {original_file_name}: {e}", exc_info=True)
    finally:
        # Clean up the specific temporary file used by this thread
        if thread_temp_path and os.path.exists(thread_temp_path):
             try:
                 os.remove(thread_temp_path)
                 logger.info(f"[Background Old Pipeline] Cleaned up temp file: {thread_temp_path}")
             except OSError as e:
                 logger.error(f"[Background Old Pipeline] Error removing temp file {thread_temp_path}: {e}")
        else:
             logger.warning(f"[Background Old Pipeline] Temp file {thread_temp_path} not found for cleanup.")


# --- NEW Gemini Chat Endpoint ---
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'webp'} # Image types added

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define mime types for Gemini based on extension
def get_mime_type(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        return 'application/pdf'
    elif ext in ['png', 'jpg', 'jpeg', 'gif', 'webp']:
        return f'image/{ext}' # Gemini supports common image types
    else:
        return None # Unsupported

@app.route("/", methods=["GET"])
def serve_index():
    """Serve the index.html file for the chat interface."""
    from flask import send_from_directory
    return send_from_directory('.', 'index.html')

@app.route("/chat-g", methods=["POST"])
@async_route
async def chat_gemini():
    """Handles chat queries using Gemini 2.5 Flash with file uploads, grounding, tools, HITL, and background indexing."""
    start_time = time.time()
    
    # Extract user_id from JWT token
    user_id = None
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]
        payload = verify_jwt_token(token)
        if payload:
            user_id = payload['user_id']
            logger.info(f"Authenticated user_id: {user_id}")
        else:
            logger.warning("Invalid or expired token in /chat-g request")
    
    if not request.is_json and 'multipart/form-data' not in request.content_type:
         return jsonify({"error": "Invalid content type. Use application/json or multipart/form-data."}), 415

    session_id = None
    query = None
    uploaded_files = []

    # Handle different content types
    if request.is_json:
        data = request.get_json()
        session_id = data.get("session_id")
        query = data.get("query")
        logger.debug("Received JSON request for /chat-g")
    elif 'multipart/form-data' in request.content_type:
        session_id = request.form.get("session_id")
        query = request.form.get("query")
        uploaded_files = request.files.getlist("files") # Field name for files
        logger.debug("Received multipart/form-data request for /chat-g")
    else:
         # Should not happen due to initial check, but good failsafe
         return jsonify({"error": "Unsupported content type."}), 415

    if not query and not uploaded_files:
        return jsonify({"error": "No query or file provided"}), 400

    # Ensure session_id (check DB first)
    is_new_session = False
    old_session_id = session_id  # Track for logging
    if not session_id:
        is_new_session = True
        session_id = f"gemini_session_{uuid.uuid4().hex[:12]}"
        logger.info(f"No session_id provided - generated new session_id for /chat-g: {session_id}")
    elif not get_chat_history(session_id):
        # Session expired or doesn't exist
        is_new_session = True
        session_id = f"gemini_session_{uuid.uuid4().hex[:12]}"
        logger.warning(f"Session {old_session_id} expired or not found - generated new session_id: {session_id}")
    else:
        logger.info(f"Continuing existing session for /chat-g: {session_id}")
        # Update session timestamp implicitly via save_chat_message later

    # --- File Handling ---
    gemini_file_objects = [] # Store full Gemini file objects temporarily
    gemini_file_uris_for_prompt = [] # Store only URIs for the prompt
    temp_file_paths_for_cleanup = [] # Main thread cleanup list

    if uploaded_files:
        logger.info(f"/chat-g processing {len(uploaded_files)} file(s) for session {session_id}.")

        for file in uploaded_files:
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                mime_type = get_mime_type(filename)
                if not mime_type:
                     logger.warning(f"Skipping unsupported file type: {filename}")
                     continue

                # Create a persistent temp file for this file for potential background use
                temp_suffix = os.path.splitext(filename)[1]
                try:
                    # Use a context manager for safer file handling
                    with tempfile.NamedTemporaryFile(delete=False, suffix=temp_suffix) as temp_f:
                        file.save(temp_f.name) # Save uploaded content to temp file
                        temp_file_path = temp_f.name
                    logger.info(f"Saved uploaded file temporarily to {temp_file_path}")
                    temp_file_paths_for_cleanup.append(temp_file_path)

                    # --- Upload to Gemini ---
                    logger.info(f"Uploading {filename} to Gemini...")
                    uploaded_gemini_file = upload_file_to_gemini(temp_file_path, display_name=filename)

                    if uploaded_gemini_file and uploaded_gemini_file.state.name == "ACTIVE":
                        gemini_file_objects.append(uploaded_gemini_file)
                        gemini_file_uris_for_prompt.append(uploaded_gemini_file.uri) # Use URI for prompt
                        logger.info(f"File {filename} ACTIVE on Gemini, URI: {uploaded_gemini_file.uri}")
                        
                        # Store file metadata in session for later retrieval
                        # This allows subsequent text-only queries to access the files
                        add_or_update_session(session_id, [{
                            "uri": uploaded_gemini_file.uri,
                            "mime_type": mime_type,
                            "display_name": filename
                        }], user_id=user_id)

                        # --- Trigger old pipeline in background for PDFs only ---
                        if mime_type == 'application/pdf':
                            case_id_for_old_pipeline = f"chatg_{session_id}_{filename.replace('.', '_')[:30]}" # Keep reasonable length
                            # IMPORTANT: Pass the *path* to the background thread. The thread is responsible for cleanup.
                            bg_thread = threading.Thread(
                                 target=run_old_pipeline_background,
                                 args=(temp_file_path, case_id_for_old_pipeline, filename),
                                 daemon=True
                            )
                            bg_thread.start()
                            logger.info(f"Started background indexing thread for {filename} using {temp_file_path}")
                            # Remove from main thread cleanup list as background thread now owns it
                            temp_file_paths_for_cleanup.remove(temp_file_path)

                    else:
                        logger.error(f"Failed to upload or process {filename} on Gemini API.")
                        # Keep the temp file in the main cleanup list if Gemini upload failed
                        # The background thread wasn't started for this file.

                except Exception as file_err:
                    logger.error(f"Error handling file {filename}: {file_err}", exc_info=True)
                    # Ensure temp file is cleaned up if created before error
                    if 'temp_file_path' in locals() and temp_file_path in temp_file_paths_for_cleanup:
                        if os.path.exists(temp_file_path):
                            try:
                                os.remove(temp_file_path)
                                logger.info(f"Cleaned up temp file {temp_file_path} after error.")
                            except OSError as rm_err:
                                logger.error(f"Error removing temp file {temp_file_path} after error: {rm_err}")
                        temp_file_paths_for_cleanup.remove(temp_file_path) # Avoid double cleanup attempt

            else:
                logger.warning(f"Skipped invalid or disallowed file: {file.filename if file else 'No File'}")
    # --- End File Handling ---

    # Save user message to DB (only if query or files were actually processed)
    if query or gemini_file_uris_for_prompt:
        save_chat_message(session_id, 'user', query or "[File Upload]", file_references=gemini_file_uris_for_prompt)

    # Determine the text content
    text_content = ""
    if query:
        text_content = query
    elif gemini_file_uris_for_prompt:
        file_names = [f.display_name for f in gemini_file_objects]
        file_list = ", ".join(file_names)
        text_content = f"I have uploaded the file(s): {file_list}. Please analyze the content and provide a comprehensive summary."
    else:
        text_content = "Hello."
    
    logger.info(f"Processing query with text='{text_content[:100]}...' and {len(gemini_file_objects)} file(s) attached")
    
    # HYBRID APPROACH: When files are present, use direct Gemini SDK with tool support
    # This is because LangChain's message system doesn't properly support Gemini File objects
    if gemini_file_objects:
        logger.info("Files detected - using direct Gemini SDK with tool support")
        try:
            # Use the new function that handles files AND tools
            final_answer = await invoke_gemini_with_files_directly(
                user_message=text_content,
                gemini_file_objs=gemini_file_objects,
                session_id=session_id
            )
            
            logger.info(f"Direct Gemini response with tools received: {final_answer[:100]}...")
            
            # Save final model response to DB (user message already saved earlier)
            save_chat_message(session_id, 'model', final_answer)
            
            # Get session files to return to frontend
            all_session_files = get_session_files(session_id)
            
            response_data = {
                "response": final_answer,
                "session_id": session_id,
                "tool_status": [],
                "session_files": all_session_files
            }
            
            # Clean up temp files that weren't handed to background thread
            cleanup_delay = 1 # seconds
            logger.debug(f"Waiting {cleanup_delay}s before cleaning main thread temp files...")
            time.sleep(cleanup_delay)
            for temp_path in temp_file_paths_for_cleanup:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                        logger.info(f"Cleaned up main thread temp file: {temp_path}")
                    except OSError as e:
                        logger.error(f"Error removing main thread temp file {temp_path}: {e}")
            
            end_time = time.time()
            logger.info(f"/chat-g request for session {session_id} completed in {end_time - start_time:.2f} seconds (direct SDK with tools).")
            return jsonify(response_data)
            
        except Exception as direct_err:
            logger.error(f"Direct Gemini SDK invocation with tools failed: {direct_err}", exc_info=True)
            # Clean up temp files on error too
            for temp_path in temp_file_paths_for_cleanup:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                        logger.info(f"Cleaned up temp file {temp_path} after error.")
                    except OSError as e:
                        logger.error(f"Error removing temp file {temp_path}: {e}")
            return jsonify({"error": f"Failed to process file with Gemini: {str(direct_err)}"}), 500
    
    # NO FILES IN THIS REQUEST: Check if session has previously uploaded files
    # If yes, use direct Gemini with those files; otherwise use LangGraph
    session_files = get_session_files(session_id)
    
    if session_files:
        # Session has uploaded files from previous messages - use direct Gemini with context
        logger.info(f"No new files but session has {len(session_files)} existing file(s) - using direct Gemini SDK with context")
        try:
            # Reconstruct file objects from stored metadata
            import google.generativeai as genai_direct
            genai_direct.configure(api_key=os.getenv("GEMINI_API_KEY"))
            
            # Get file objects by URI
            gemini_file_objects_from_session = []
            for file_info in session_files:
                try:
                    file_obj = genai_direct.get_file(name=file_info['uri'].split('/')[-1])
                    gemini_file_objects_from_session.append(file_obj)
                except Exception as file_err:
                    logger.warning(f"Could not retrieve file {file_info['uri']}: {file_err}")
            
            if gemini_file_objects_from_session:
                # Use direct SDK with existing files
                final_answer = await invoke_gemini_with_files_directly(
                    user_message=text_content,
                    gemini_file_objs=gemini_file_objects_from_session,
                    session_id=session_id
                )
                
                logger.info(f"Direct Gemini response with session files: {final_answer[:100]}...")
                save_chat_message(session_id, 'model', final_answer)
                
                response_data = {
                    "response": final_answer,
                    "session_id": session_id,
                    "tool_status": [],
                    "session_files": session_files  # Return file info to frontend
                }
                
                end_time = time.time()
                logger.info(f"/chat-g request completed in {end_time - start_time:.2f} seconds (direct SDK with session files).")
                return jsonify(response_data)
        except Exception as session_file_err:
            logger.error(f"Error using session files: {session_file_err}", exc_info=True)
            # Fall through to LangGraph if session file retrieval fails
    
    # NO FILES AT ALL: Use LangGraph as normal
    current_input_message = HumanMessage(content=text_content)
    logger.info(f"No files - using LangGraph agent")

    # --- LangGraph Interaction ---
    config = {"configurable": {"thread_id": session_id}}
    logger.info(f"Invoking Gemini agent graph for session: {session_id}")
    try:
        final_state = None
        interrupt_response = None
        tool_execution_status = [] # List to track tool calls

        # Use astream with stream_mode="values" (default) to get full state after each step
        async for event in agent_executor_g.astream(
            {"messages": [current_input_message]}, # Pass input as part of messages
            config=config,
            stream_mode="values"  # Explicitly set to get full state
        ):
            # Log the event structure for debugging
            logger.debug(f"Received event: {type(event)}, Keys: {event.keys() if isinstance(event, dict) else 'N/A'}")
            
            # Check if we've reached an interrupt point
            if "__interrupt__" in event:
                # Extract pending tool calls from state
                state = event
                messages = state.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                        needs_approval = any(
                            tc["name"] in TOOLS_REQUIRING_APPROVAL_G for tc in last_msg.tool_calls
                        )
                        if needs_approval:
                            logger.info("Interrupting before tool execution for approval.")
                            # Generate the approval request data
                            approval_requests_data = []
                            for tc in last_msg.tool_calls:
                                if tc["name"] in TOOLS_REQUIRING_APPROVAL_G:
                                    approval_msg = f"Allow me to use the tool '{tc['name']}' with arguments {tc.get('args', {})}. (Yes/No)"
                                    approval_requests_data.append({
                                        "tool_call_id": tc["id"],
                                        "tool_name": tc["name"],
                                        "tool_args": tc.get("args", {}),
                                        "approval_message": approval_msg
                                    })
                            
                            # Clean up temp files before returning
                            cleanup_delay = 1 # seconds
                            logger.debug(f"Waiting {cleanup_delay}s before cleaning main thread temp files...")
                            time.sleep(cleanup_delay)
                            for temp_path in temp_file_paths_for_cleanup:
                                if os.path.exists(temp_path):
                                    try:
                                        os.remove(temp_path)
                                        logger.info(f"Cleaned up main thread temp file: {temp_path}")
                                    except OSError as e:
                                        logger.error(f"Error removing main thread temp file {temp_path}: {e}")
                            
                            return jsonify({
                                "response": "Tool approval required",
                                "needs_approval": True,
                                "approval_agent_type": "gemini",
                                "approval_requests": approval_requests_data,
                                "session_id": session_id
                            })
            
            # Store the final state - event IS the state in values mode
            final_state = event
            logger.debug(f"Updated final_state with event containing {len(event.get('messages', [])) if isinstance(event, dict) else 0} messages")

        # --- Process Final Response ---
        response_data = {}
        if final_state and isinstance(final_state, dict) and final_state.get("messages"):
            messages = final_state["messages"]
            logger.info(f"Final state has {len(messages)} messages")
            last_message = messages[-1]
            final_answer = ""
            if isinstance(last_message, AIMessage):
                final_answer = last_message.content
                logger.info(f"Extracted AI message content: {final_answer[:100] if final_answer else 'Empty'}...")
            elif hasattr(last_message, 'content'): # Handle other message types
                final_answer = last_message.content
                logger.info(f"Extracted message content: {final_answer[:100] if final_answer else 'Empty'}...")
            else:
                 final_answer = str(last_message)
                 logger.info(f"Converted message to string: {final_answer[:100]}...")

            # --- Browser Redirect Check ---
            is_redirect = False
            try:
                action_data = json.loads(final_answer)
                if isinstance(action_data, dict) and action_data.get("action") == "BROWSER_REDIRECT":
                    logger.info(f"/chat-g Browser redirect: {action_data.get('url')}")
                    response_data = {
                        "action": "BROWSER_REDIRECT",
                        "url": action_data.get("url"),
                        "session_id": session_id
                    }
                    is_redirect = True
            except (json.JSONDecodeError, TypeError):
                pass

            if not is_redirect:
                 # Get session files to return to frontend
                 all_session_files = get_session_files(session_id)
                 
                 response_data = {
                    "response": final_answer or "Sorry, I couldn't generate a response.",
                    "session_id": session_id,
                    "tool_status": tool_execution_status, # Include tool execution messages
                    "session_files": all_session_files
                 }
                 # Save final model response to DB
                 save_chat_message(session_id, 'model', final_answer)
        else:
             # Handle cases where agent finishes without a final message (e.g., error)
             logger.error(f"Agent finished for session {session_id} but no final message found in state.")
             response_data = {
                 "response": "Sorry, an error occurred while processing your request.",
                 "session_id": session_id,
                 "tool_status": tool_execution_status
             }

        end_time = time.time()
        logger.info(f"/chat-g request for session {session_id} completed in {end_time - start_time:.2f} seconds.")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in /chat-g endpoint for session {session_id}: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
    finally:
        # Clean up any *main thread* temp files that weren't passed to background threads
        cleanup_delay = 1 # seconds
        logger.debug(f"Waiting {cleanup_delay}s before cleaning main thread temp files...")
        time.sleep(cleanup_delay)
        cleaned_count = 0
        for temp_path in temp_file_paths_for_cleanup:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.info(f"Cleaned up main thread temp file: {temp_path}")
                    cleaned_count += 1
                except OSError as e:
                     logger.error(f"Error removing main thread temp file {temp_path}: {e}")
        if temp_file_paths_for_cleanup:
             logger.info(f"Main thread cleanup finished. Removed {cleaned_count}/{len(temp_file_paths_for_cleanup)} files.")


# --- Google Drive OAuth Endpoints ---

@app.route("/auth/google", methods=["GET"])
def auth_google():
    """Initiates Google OAuth flow for Drive access."""
    try:
        credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "./credentials.json")
        redirect_uri = os.getenv("GOOGLE_OAUTH_REDIRECT_URI", "http://localhost:7860/oauth2callback")
        
        if not os.path.exists(credentials_path):
            return jsonify({
                "error": "Google credentials file not found. Please configure GOOGLE_CREDENTIALS_PATH."
            }), 500
        
        flow = InstalledAppFlow.from_client_secrets_file(
            credentials_path,
            scopes=SCOPES,
            redirect_uri=redirect_uri
        )
        
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true'
        )
        
        logger.info(f"Redirecting to Google OAuth: {authorization_url}")
        return redirect(authorization_url)
        
    except Exception as e:
        logger.error(f"Error initiating Google OAuth: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/oauth2callback", methods=["GET"])
def oauth2callback():
    """Handles the OAuth2 callback from Google."""
    try:
        credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "./credentials.json")
        redirect_uri = os.getenv("GOOGLE_OAUTH_REDIRECT_URI", "http://localhost:7860/oauth2callback")
        
        flow = InstalledAppFlow.from_client_secrets_file(
            credentials_path,
            scopes=SCOPES,
            redirect_uri=redirect_uri
        )
        
        # Fetch token using the authorization response URL
        flow.fetch_token(authorization_response=request.url)
        
        # Save credentials to pickle file
        credentials = flow.credentials
        with open(TOKEN_PICKLE_PATH, 'wb') as token:
            pickle.dump(credentials, token)
        
        logger.info("Google Drive authentication successful. Token saved.")
        return "Authentication successful! You can close this tab and return to the application.", 200
        
    except Exception as e:
        logger.error(f"Error in OAuth callback: {e}")
        return f"Authentication failed: {str(e)}", 500


@app.route("/sync_drive", methods=["POST"])
def sync_drive():
    """
    Syncs PDF files from the configured Google Drive folder.
    Downloads and processes each PDF using the existing upload pipeline.
    """
    try:
        # Get Drive service (checks for valid token)
        service = _get_drive_service()
        
        if service is None:
            return jsonify({
                "error": "Google Drive not authenticated. Please visit /auth/google first."
            }), 401
        
        # Get folder ID from environment
        folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        if not folder_id:
            return jsonify({
                "error": "GOOGLE_DRIVE_FOLDER_ID not configured in environment."
            }), 500
        
        # Get case_id from request (optional)
        data = request.json or {}
        base_case_id = data.get("case_id", f"drive_sync_{uuid.uuid4().hex[:8]}")
        
        # Search for PDF files in the folder
        query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
        logger.info(f"Searching Google Drive folder {folder_id} for PDFs...")
        
        results = service.files().list(
            q=query,
            fields="files(id, name)",
            pageSize=100
        ).execute()
        
        files = results.get('files', [])
        
        if not files:
            return jsonify({
                "status": "Success",
                "message": "No PDF files found in the Drive folder.",
                "processed": [],
                "skipped": [],
                "errors": []
            })
        
        logger.info(f"Found {len(files)} PDF files. Starting download and processing...")
        
        processed = []
        skipped = []
        errors = []
        max_pages = int(os.getenv("MAX_DOCUMENT_PAGES", "20"))
        
        # Process each file
        for file in files:
            file_name = file['name']
            file_id = file['id']
            
            try:
                logger.info(f"Processing: {file_name} (ID: {file_id})")
                
                # Download file content
                request_obj = service.files().get_media(fileId=file_id)
                file_buffer = io.BytesIO()
                downloader = MediaIoBaseDownload(file_buffer, request_obj)
                
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        logger.info(f"Download progress: {int(status.progress() * 100)}%")
                
                file_buffer.seek(0)
                logger.info(f"Downloaded {file_name} successfully")
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_f:
                    temp_f.write(file_buffer.read())
                    temp_path = temp_f.name
                
                try:
                    # Load and split PDF
                    documents = load_and_split_pdf(temp_path)
                    
                    if not documents:
                        errors.append(f"{file_name}: Failed to load/split PDF")
                        continue
                    
                    # Generate unique case_id for this file
                    case_id = f"{base_case_id}_{file_name.replace('.pdf', '').replace(' ', '_')}"
                    
                    # Index in vector store
                    vector_store.index_documents(documents, case_id, file_name)
                    
                    # Check page count for graph building
                    num_pages = 0
                    for doc in documents:
                        page_num = doc.metadata.get("page", 0)
                        if page_num > num_pages:
                            num_pages = page_num
                    
                    # Conditionally start graph building
                    if num_pages <= max_pages:
                        # Run graph building in current thread (synchronous for sync operation)
                        graph_builder.extract_and_store_graph(documents, case_id, file_name)
                        processed.append(f"{file_name} (indexed + graph built)")
                    else:
                        skipped.append(f"{file_name} (indexed, graph skipped - {num_pages} pages exceeds limit)")
                    
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")
                errors.append(f"{file_name}: {str(e)}")
        
        logger.info(f"Drive sync complete. Processed: {len(processed)}, Skipped: {len(skipped)}, Errors: {len(errors)}")
        
        return jsonify({
            "status": "Sync Complete",
            "message": f"Processed {len(processed)} files, skipped {len(skipped)}, encountered {len(errors)} errors.",
            "processed": processed,
            "skipped": skipped,
            "errors": errors
        })
        
    except HttpError as e:
        logger.error(f"Google Drive API error: {e}")
        return jsonify({"error": f"Drive API error: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Error during drive sync: {e}")
        return jsonify({"error": str(e)}), 500

# --- Cases Management Endpoint ---

@app.route("/cases", methods=["GET"])
def list_cases():
    """
    Lists all processed cases/documents.
    Returns a list of case_ids with their associated files.
    """
    try:
        if not graph_store:
            return jsonify({
                "error": "Graph store is not initialized. Cases list unavailable."
            }), 503
        
        logger.info("Fetching cases list from Neo4j...")
        
        # Query to get all Document nodes grouped by case_id
        query = """
        MATCH (d:Document)
        WHERE d.case_id IS NOT NULL AND d.file_name IS NOT NULL
        RETURN d.case_id AS caseId, collect(DISTINCT d.file_name) AS files
        ORDER BY caseId
        """
        
        results = graph_store.run_query(query)
        
        if not results:
            logger.info("No cases found in the database")
            return jsonify([])
        
        # Format results
        formatted_results = [
            {
                "case_id": record["caseId"],
                "files": record["files"]
            }
            for record in results
        ]
        
        logger.info(f"Found {len(formatted_results)} cases")
        return jsonify(formatted_results)
        
    except Exception as e:
        logger.error(f"Error fetching cases list: {e}")
        return jsonify({"error": f"Failed to retrieve cases: {str(e)}"}), 500


@app.route("/auth/status", methods=["GET"])
def get_auth_status():
    """
    Checks if the backend has valid Google Drive credentials.
    Returns authentication status.
    """
    try:
        logger.info("Checking Google Drive authentication status...")
        
        is_authenticated = False
        
        # Check if token file exists
        if TOKEN_PICKLE_PATH.exists():
            try:
                # Load credentials
                with open(TOKEN_PICKLE_PATH, 'rb') as token:
                    creds = pickle.load(token)
                
                logger.info("Token file found, checking validity...")
                
                # Check if credentials are valid
                if creds and creds.valid:
                    is_authenticated = True
                    logger.info("Credentials are valid")
                elif creds and creds.expired and creds.refresh_token:
                    # Try to refresh
                    logger.info("Credentials expired, attempting refresh...")
                    try:
                        creds.refresh(Request())
                        
                        # Save refreshed credentials
                        with open(TOKEN_PICKLE_PATH, 'wb') as token:
                            pickle.dump(creds, token)
                        
                        is_authenticated = creds.valid
                        logger.info(f"Refresh {'successful' if is_authenticated else 'failed'}")
                    except Exception as refresh_error:
                        logger.warning(f"Failed to refresh credentials: {refresh_error}")
                        is_authenticated = False
                else:
                    logger.info("Credentials are invalid and cannot be refreshed")
                    is_authenticated = False
                    
            except (FileNotFoundError, pickle.UnpicklingError) as e:
                logger.warning(f"Error loading token file: {e}")
                is_authenticated = False
        else:
            logger.info("Token file does not exist")
            is_authenticated = False
        
        return jsonify({
            "authenticated": is_authenticated,
            "message": "Google Drive is connected" if is_authenticated else "Google Drive not connected. Visit /auth/google to authorize."
        })
        
    except Exception as e:
        logger.error(f"Error checking auth status: {e}")
        return jsonify({
            "authenticated": False,
            "error": str(e)
        }), 500


# --- Health Check Endpoint ---

@app.route("/health", methods=["GET"])
def health_check():
    """Health check for Docker."""
    return "OK", 200


if __name__ == "__main__":
    # This is for local dev only. Gunicorn is used in production (see start.sh)
    port = int(os.getenv("PORT", 7860))
    logger.info(f"Starting Flask app in debug mode on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)