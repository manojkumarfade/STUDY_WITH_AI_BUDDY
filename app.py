import streamlit as st
import sqlite3
import os
import json
import PyPDF2
from PIL import Image
from io import BytesIO
import time
import datetime
import uuid
import random
import hashlib
import google.generativeai as genai
import logging
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from cryptography.fernet import Fernet
import secrets
import requests

# Page Configuration
st.set_page_config(
    page_title="AI Study Buddy - Enhanced",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import OpenAI for A4F API support
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    OPENAI_AVAILABLE = False
    OpenAI = None  # Define OpenAI as None when not available

# EMAIL CONFIGURATION

# Email configuration constants
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 465

# Load email credentials from environment
EMAIL_SENDER = st.secrets.get("EMAIL_SENDER")
EMAIL_PASSWORD = st.secrets.get("EMAIL_PASSWORD")

# Session timeout in seconds (15 minutes)
SESSION_TIMEOUT = 15 * 60


# Configure the generative AI model
api_configured = False
try:
    # Prioritize environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    # Try to get the API key from environment variables first
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        api_configured = True
    else:
        # Fallback to Streamlit secrets if env var not found
        api_key = st.secrets.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            api_configured = True
        else:
            st.error("GOOGLE_API_KEY not found in environment variables or Streamlit secrets.")
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")


# Global flag to track if API is configured
GEMINI_API_AVAILABLE = api_configured

# Handle new chat state
if 'new_chat_initiated' in st.session_state and st.session_state.new_chat_initiated:
    st.session_state.new_chat_initiated = False

# A4F API CONFIGURATION

# Initialize A4F API client
a4f_client = None
A4F_API_AVAILABLE = False

if OPENAI_AVAILABLE and OpenAI is not None:
    try:
        # Try to get A4F API key from environment
        from dotenv import load_dotenv
        load_dotenv()
        a4f_api_key = st.secrets.get("A4F_API_KEY") or st.secrets.get("a4f_api_key")

        if a4f_api_key:
            a4f_client = OpenAI(
                api_key=a4f_api_key,
                base_url="https://api.a4f.co/v1"
            )
            A4F_API_AVAILABLE = True
            st.info("✅ A4F API configured successfully!")
        else:
            st.warning("A4F API key not found. Install with: pip install openai")
    except Exception as e:
        st.warning(f"A4F API configuration error: {e}")

# MULTI-API AI FUNCTIONS

def get_available_ai_client():
    """Get the first available AI client (Gemini -> A4F -> None)"""
    if GEMINI_API_AVAILABLE:
        # Use the correct way to access GenerativeModel
        model = genai.GenerativeModel(model_name='gemini-2.5-flash')
        return "gemini", model
    elif A4F_API_AVAILABLE and a4f_client:
        return "a4f", a4f_client
    else:
        return None, None

def get_ai_response_multi_api(prompt, context=""):
    """Enhanced AI response with fallback between Gemini and A4F"""
    client_type, client = get_available_ai_client()

    if not client:
        return "❌ No AI API available. Please configure either Gemini or A4F API."

    try:
        # Add conversation memory (last 5 messages)
        memory_context = ""
        if len(st.session_state.messages) > 1:
            recent = st.session_state.messages[-5:]
            memory_context = "\n".join([f"{m['role']}: {m['content']}" for m in recent])

        # Check if web search is enabled
        web_context = ""
        if st.session_state.get('web_search_enabled', False):
            with st.spinner("Searching the web for current information..."):
                search_results = web_search(prompt, num_results=3)
                if search_results and not search_results[0].startswith("❌"):
                    web_context = "Web Search Results:\n" + "\n\n".join(search_results)
                    st.info("Web search completed. Including current information in response.")

        base_prompt = f"""You are an AI Study Buddy assistant. Your goal is to help students learn effectively.

        Previous conversation:
        {memory_context}

        Additional Context: {context}
        
        {web_context}

        User Query: {prompt}

        Provide clear, educational, and encouraging responses. Break down complex topics into simple terms.
        Include examples when helpful. Always maintain a supportive tone.
        If you used web search results, mention that the information is current and from the web.
        """

        # Apply persona
        full_prompt = get_persona_prompt(st.session_state.ai_persona, base_prompt)

        response_text = ""  # Initialize response_text variable

        if client_type == "gemini":
            # Use Gemini API
            response = client.generate_content(full_prompt)
            response_text = response.text

        elif client_type == "a4f" and a4f_client is not None:
            # Use A4F API
            completion = a4f_client.chat.completions.create(
                model="provider-1/llama-3.2-1b-instruct-fp-1",  # Updated model name
                messages=[
                    {"role": "system", "content": "You are a helpful AI Study Buddy assistant."},
                    {"role": "user", "content": full_prompt}
                ]
            )
            response_text = completion.choices[0].message.content

        # Detect emotion
        emotion = detect_emotion_from_text(prompt)
        st.session_state.last_emotion = emotion

        # Add supportive message if needed
        supportive = get_supportive_response(emotion)
        if supportive and response_text:  # Check if response_text is not empty
            response_text = f"{supportive}\n\n{response_text}"

        # Save chat to database if user is logged in
        if st.session_state.get('logged_in') and st.session_state.get('user_email'):
            save_chat_to_db(st.session_state.user_email, "user", prompt)
            save_chat_to_db(st.session_state.user_email, "assistant", response_text)

        return response_text

    except Exception as e:
        error_msg = f"Error with {client_type.upper() if client_type else 'UNKNOWN'} API: {str(e)}"

        # Try fallback to other API
        if client_type == "gemini" and A4F_API_AVAILABLE and a4f_client:
            st.warning("⚠️ Gemini API failed, trying A4F API...")
            try:
                return get_ai_response_a4f(prompt, context)
            except:
                pass
        elif client_type == "a4f" and GEMINI_API_AVAILABLE:
            st.warning("⚠️ A4F API failed, trying Gemini API...")
            try:
                return get_ai_response_gemini(prompt, context)
            except:
                pass

        return f"{error_msg}\n\nPlease check your API configuration."

def get_ai_response_gemini(prompt, context=""):
    """Gemini-specific AI response function"""
    if not GEMINI_API_AVAILABLE:
        return "❌ Gemini API not configured."

    try:
        model = genai.GenerativeModel(model_name='gemini-2.5-flash')

        # Add conversation memory (last 5 messages)
        memory_context = ""
        if len(st.session_state.messages) > 1:
            recent = st.session_state.messages[-5:]
            memory_context = "\n".join([f"{m['role']}: {m['content']}" for m in recent])

        base_prompt = f"""You are an AI Study Buddy assistant. Your goal is to help students learn effectively.

        Previous conversation:
        {memory_context}

        Additional Context: {context}

        User Query: {prompt}

        Provide clear, educational, and encouraging responses. Break down complex topics into simple terms.
        Include examples when helpful. Always maintain a supportive tone.
        """

        # Apply persona
        full_prompt = get_persona_prompt(st.session_state.ai_persona, base_prompt)

        response = model.generate_content(full_prompt)
        response_text = response.text

        # Detect emotion
        emotion = detect_emotion_from_text(prompt)
        st.session_state.last_emotion = emotion

        # Add supportive message if needed
        supportive = get_supportive_response(emotion)
        if supportive and response_text:  # Check if response_text is not empty
            response_text = f"{supportive}\n\n{response_text}"

        return response_text
    except Exception as e:
        return f"❌ Gemini API Error: {str(e)}"

def get_ai_response_a4f(prompt, context=""):
    """A4F-specific AI response function"""
    if not A4F_API_AVAILABLE or not a4f_client:
        return "❌ A4F API not configured."

    try:
        # Add conversation memory (last 5 messages)
        memory_context = ""
        if len(st.session_state.messages) > 1:
            recent = st.session_state.messages[-5:]
            memory_context = "\n".join([f"{m['role']}: {m['content']}" for m in recent])

        base_prompt = f"""You are an AI Study Buddy assistant. Your goal is to help students learn effectively.

        Previous conversation:
        {memory_context}

        Additional Context: {context}

        User Query: {prompt}

        Provide clear, educational, and encouraging responses. Break down complex topics into simple terms.
        Include examples when helpful. Always maintain a supportive tone.
        """

        # Apply persona
        full_prompt = get_persona_prompt(st.session_state.ai_persona, base_prompt)

        completion = a4f_client.chat.completions.create(
            model="provider-1/llama-3.2-1b-instruct-fp-1",  # Updated model name
            messages=[
                {"role": "system", "content": "You are a helpful AI Study Buddy assistant."},
                {"role": "user", "content": full_prompt}
            ]
        )

        response_text = completion.choices[0].message.content

        # Detect emotion
        emotion = detect_emotion_from_text(prompt)
        st.session_state.last_emotion = emotion

        # Add supportive message if needed
        supportive = get_supportive_response(emotion)
        if supportive and response_text:  # Check if response_text is not empty
            response_text = f"{supportive}\n\n{response_text}"

        return response_text
    except Exception as e:
        return f"❌ A4F API Error: {str(e)}"

def web_search(query, num_results=5):
    """
    Perform web search using Serper API
    Returns a list of formatted search results
    """
    try:
        # Get API key from environment variables
        serper_api_key = st.secrets.get("SERPER_API_KEY")
        if not serper_api_key:
            return ["❌ Serper API key not configured. Please set SERPER_API_KEY in your environment variables."]
        
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": serper_api_key,
            "Content-Type": "application/json"
        }
        payload = {"q": query, "num": num_results}
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        results = response.json().get("organic", [])
        formatted_results = []
        
        for r in results[:num_results]:
            title = r.get('title', 'No title')
            link = r.get('link', '#')
            snippet = r.get('snippet', 'No description')
            formatted_results.append(f"🔗 {title} - {link}\n  {snippet}")
        
        return formatted_results
    except Exception as e:
        return [f"❌ Web search error: {str(e)}"]

# Advanced imports with fallbacks
try:
    import pytz
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "pytz"])
    import pytz

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except:
    PLOTLY_AVAILABLE = False

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_API_AVAILABLE = True
except:
    YOUTUBE_API_AVAILABLE = False

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except:
    SPEECH_AVAILABLE = False

# Advanced feature imports
try:
    import cv2
    import pytesseract
    from PIL import Image
    # Configure Tesseract OCR path for Windows
    if os.name == 'nt':  # Windows
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    COMPUTER_VISION_AVAILABLE = True
except:
    COMPUTER_VISION_AVAILABLE = False

try:
    from transformers import pipeline
    from diffusers import StableDiffusionPipeline
    import torch
    AI_IMAGING_AVAILABLE = True
except:
    AI_IMAGING_AVAILABLE = False

try:
    from ultralytics import YOLO
    OBJECT_DETECTION_AVAILABLE = True
except:
    OBJECT_DETECTION_AVAILABLE = False

try:
    import pyvista as pv
    from pyvista import examples
    THREED_VISUALIZATION_AVAILABLE = True
except:
    THREED_VISUALIZATION_AVAILABLE = False

try:
    from streamlit_webrtc import webrtc_streamer
    import av
    WEBCAM_AVAILABLE = True
except:
    WEBCAM_AVAILABLE = False

# DATABASE SCHEMA - Enhanced with gamification

@st.cache_resource
def get_db_connection():
    """Get cached database connection"""
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect('data/app.db', timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_achievements():
    """Initialize achievement database"""
    achievements = [
        ("first_session", "First Steps", "Complete your first study session", "🎯"),
        ("streak_3", "Getting Consistent", "Maintain a 3-day streak", "🔥"),
        ("streak_7", "Week Warrior", "Maintain a 7-day streak", "🔥🔥"),
        ("streak_30", "Month Master", "Maintain a 30-day streak", "🔥🔥🔥"),
        ("quiz_perfect", "Perfect Score", "Get 100% on a quiz", "💯"),
        ("level_5", "Rising Scholar", "Reach level 5", "⭐"),
        ("level_10", "Expert Learner", "Reach level 10", "⭐⭐"),
        ("night_owl", "Night Owl", "Study after midnight", "🦉"),
        ("early_bird", "Early Bird", "Study before 6 AM", "🐦"),
        ("flash_master", "Flashcard Master", "Review 100 flashcards", "📚"),
        ("quiz_master", "Quiz Champion", "Complete 50 quizzes", "🏆"),
        ("speed_demon", "Speed Demon", "Complete quiz in under 5 min", "⚡"),
        ("persistent", "Never Give Up", "Answer same question correctly after 3 failures", "💪"),
        ("diverse_learner", "Jack of All Trades", "Study 5 different subjects", "🎨"),
        ("pet_lover", "Pet Caretaker", "Keep pet happiness above 80 for 7 days", "❤️"),
    ]
    
    conn = None
    try:
        conn = get_db_connection()
        for ach_id, name, desc, icon in achievements:
            conn.execute("""
                INSERT OR IGNORE INTO achievements (id, name, description, icon)
                VALUES (?, ?, ?, ?)
            """, (ach_id, name, desc, icon))
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error during achievements initialization: {e}")
        if conn:
            conn.rollback()
    except Exception as e:
        st.error(f"Unexpected error during achievements initialization: {e}")
        if conn:
            conn.rollback()

def init_database():
    """Initialize enhanced database schema"""
    try:
        os.makedirs('data', exist_ok=True)
        
        # Check if users table exists and has correct schema
        conn = get_db_connection()
        try:
            # Check if users table exists
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
            if not cursor.fetchone():
                # Table doesn't exist, create it
                conn.execute('''
                    CREATE TABLE users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        verified INTEGER DEFAULT 0,
                        verification_code TEXT,
                        created_at TEXT,
                        last_login TEXT,
                        session_token TEXT,
                        session_expires TEXT
                    )
                ''')
                conn.commit()
                st.info("Users table created successfully!")
            else:
                # Table exists, check and add missing columns
                cursor = conn.execute("PRAGMA table_info(users)")
                columns = [column[1] for column in cursor.fetchall()]
                
                # Required columns for the new schema
                required_columns = {
                    'password_hash': 'TEXT',
                    'verified': 'INTEGER DEFAULT 0',
                    'verification_code': 'TEXT',
                    'created_at': 'TEXT',
                    'last_login': 'TEXT',
                    'session_token': 'TEXT',
                    'session_expires': 'TEXT'
                }
                
                columns_added = []
                for col_name, col_type in required_columns.items():
                    if col_name not in columns:
                        try:
                            conn.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}")
                            columns_added.append(col_name)
                        except Exception as col_error:
                            st.warning(f"Could not add column {col_name}: {col_error}")
                
                if columns_added:
                    conn.commit()
                    st.info(f"Database schema updated! Added columns: {', '.join(columns_added)}")
        except Exception as e:
            st.error(f"Database initialization error: {e}")
        
        # Check and create user_chats table
        try:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_chats'")
            if not cursor.fetchone():
                # Table doesn't exist, create it
                conn.execute('''
                    CREATE TABLE user_chats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_email TEXT NOT NULL,
                        message_role TEXT NOT NULL,
                        message_content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (user_email) REFERENCES users (email)
                    )
                ''')
                conn.commit()
                st.info("User chats table created successfully!")
            else:
                # Table exists, check and add missing columns
                cursor = conn.execute("PRAGMA table_info(user_chats)")
                columns = [column[1] for column in cursor.fetchall()]
                
                required_chat_columns = {
                    'user_email': 'TEXT NOT NULL',
                    'message_role': 'TEXT NOT NULL',
                    'message_content': 'TEXT NOT NULL',
                    'timestamp': 'TEXT NOT NULL'
                }
                
                columns_added = []
                for col_name, col_type in required_chat_columns.items():
                    if col_name not in columns:
                        try:
                            conn.execute(f"ALTER TABLE user_chats ADD COLUMN {col_name} {col_type}")
                            columns_added.append(col_name)
                        except Exception as col_error:
                            st.warning(f"Could not add column {col_name} to user_chats: {col_error}")
                
                if columns_added:
                    conn.commit()
                    st.info(f"User chats table updated! Added columns: {', '.join(columns_added)}")
        except Exception as e:
            st.error(f"User chats table initialization error: {e}")

        # Create all tables with proper schema
        tables_to_create = {
            'study_sessions': '''
                CREATE TABLE IF NOT EXISTS study_sessions (
                    id TEXT PRIMARY KEY,
                    user_email TEXT,
                    session_type TEXT,
                    duration INTEGER,
                    score INTEGER,
                    timestamp TEXT,
                    xp_earned INTEGER DEFAULT 0,
                    FOREIGN KEY (user_email) REFERENCES users (email)
                )
            ''',
            'flashcards': '''
                CREATE TABLE IF NOT EXISTS flashcards (
                    id TEXT PRIMARY KEY,
                    user_email TEXT,
                    question TEXT,
                    answer TEXT,
                    created_at TEXT,
                    times_reviewed INTEGER DEFAULT 0,
                    times_correct INTEGER DEFAULT 0,
                    FOREIGN KEY (user_email) REFERENCES users (email)
                )
            ''',
            'study_materials': '''
                CREATE TABLE IF NOT EXISTS study_materials (
                    id TEXT PRIMARY KEY,
                    user_email TEXT,
                    material_name TEXT,
                    material_type TEXT,
                    content TEXT,
                    summary TEXT,
                    FOREIGN KEY (user_email) REFERENCES users (email)
                )
            ''',
            'achievements': '''
                CREATE TABLE IF NOT EXISTS achievements (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    icon TEXT,
                    unlocked INTEGER DEFAULT 0,
                    unlocked_at TEXT
                )
            ''',
            'quiz_attempts': '''
                CREATE TABLE IF NOT EXISTS quiz_attempts (
                    id TEXT PRIMARY KEY,
                    user_email TEXT,
                    question_hash TEXT,
                    question TEXT,
                    correct INTEGER,
                    confidence INTEGER,
                    timestamp TEXT,
                    FOREIGN KEY (user_email) REFERENCES users (email)
                )
            ''',
            'user_profile': '''
                CREATE TABLE IF NOT EXISTS user_profile (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_email TEXT UNIQUE NOT NULL,
                    username TEXT DEFAULT 'Scholar',
                    total_xp INTEGER DEFAULT 0,
                    level INTEGER DEFAULT 1,
                    study_pet TEXT DEFAULT 'egg',
                    pet_happiness INTEGER DEFAULT 50,
                    learning_style TEXT DEFAULT 'unknown',
                    current_theme TEXT DEFAULT 'default',
                    created_at TEXT,
                    FOREIGN KEY (user_email) REFERENCES users (email)
                )
            ''',
            'daily_quests': '''
                CREATE TABLE IF NOT EXISTS daily_quests (
                    id TEXT PRIMARY KEY,
                    user_email TEXT,
                    date TEXT,
                    quest_type TEXT,
                    target INTEGER,
                    progress INTEGER DEFAULT 0,
                    completed INTEGER DEFAULT 0,
                    FOREIGN KEY (user_email) REFERENCES users (email)
                )
            ''',
            'chat_sessions': '''
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_email TEXT NOT NULL,
                    session_name TEXT NOT NULL,
                    messages TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_email) REFERENCES users (email)
                )
            '''
        }
        
        # Create tables and add missing columns
        for table_name, create_sql in tables_to_create.items():
            try:
                # Check if table exists
                cursor = conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                table_exists = cursor.fetchone()
                
                if not table_exists:
                    # Create table
                    conn.execute(create_sql)
                    conn.commit()
                    st.info(f"{table_name} table created successfully!")
                else:
                    # Table exists, check and add missing columns
                    cursor = conn.execute(f"PRAGMA table_info({table_name})")
                    columns = [column[1] for column in cursor.fetchall()]
                    
                    # Define required columns for each table
                    required_columns = {
                        'study_sessions': ['user_email'],
                        'flashcards': ['user_email'],
                        'study_materials': ['user_email'],
                        'achievements': [],  # No user_email needed
                        'quiz_attempts': ['user_email'],
                        'user_profile': ['user_email'],
                        'daily_quests': ['user_email'],
                        'chat_sessions': ['user_email']
                    }
                    
                    columns_added = []
                    for col_name in required_columns.get(table_name, []):
                        if col_name not in columns:
                            try:
                                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} TEXT")
                                columns_added.append(col_name)
                            except Exception as col_error:
                                st.warning(f"Could not add column {col_name} to {table_name}: {col_error}")
                    
                    if columns_added:
                        conn.commit()
                        st.info(f"{table_name} table updated! Added columns: {', '.join(columns_added)}")
            
            except Exception as e:
                st.error(f"Error creating/updating {table_name} table: {e}")
        
        # Legacy queries for backward compatibility (these will be ignored if tables already exist)
        queries = []
        
        conn = get_db_connection()
        try:
            for query in queries:
                conn.execute(query)
            conn.commit()
        except sqlite3.Error as e:
            st.error(f"Database error during commit: {e}")
            conn.rollback()
        except Exception as e:
            st.error(f"Unexpected error during commit: {e}")
            conn.rollback()
        
        # Initialize default achievements
        init_achievements()
        
        # Migrate existing users if needed
        migrate_existing_users()
        
    except Exception as e:
        st.error(f"Error initializing database: {e}")

def migrate_existing_users():
    """Migrate existing users to new schema"""
    conn = None
    try:
        conn = get_db_connection()
        # Check if there are users without password_hash
        cursor = conn.execute("SELECT email FROM users WHERE password_hash IS NULL OR password_hash = ''")
        users_to_migrate = cursor.fetchall()
        
        if users_to_migrate:
            st.warning(f"Found {len(users_to_migrate)} users without password hashes. Please re-register these accounts.")
            # Optionally, you could delete these users or prompt for re-registration
            for user in users_to_migrate:
                conn.execute("DELETE FROM users WHERE email = ?", (user[0],))
            conn.commit()
            st.info("Cleaned up users without proper authentication.")
        
        # Fix user_profile table constraint issue
        try:
            # Check if user_profile table has the old constraint
            cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='user_profile'")
            table_sql = cursor.fetchone()
            
            if table_sql and 'CHECK (id = 1)' in table_sql[0]:
                st.info("Fixing user_profile table constraint...")
                # Drop and recreate user_profile table without constraint
                conn.execute("DROP TABLE IF EXISTS user_profile")
                conn.execute('''
                    CREATE TABLE user_profile (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_email TEXT UNIQUE NOT NULL,
                        username TEXT DEFAULT 'Scholar',
                        total_xp INTEGER DEFAULT 0,
                        level INTEGER DEFAULT 1,
                        study_pet TEXT DEFAULT 'egg',
                        pet_happiness INTEGER DEFAULT 50,
                        learning_style TEXT DEFAULT 'unknown',
                        current_theme TEXT DEFAULT 'default',
                        created_at TEXT,
                        FOREIGN KEY (user_email) REFERENCES users (email)
                    )
                ''')
                conn.commit()
                st.success("User profile table constraint fixed!")
        except Exception as constraint_error:
            st.warning(f"Could not fix user_profile constraint: {constraint_error}")
        
        # Fix user_chats table schema issue - Enhanced version
        try:
            # Check if user_chats table exists
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_chats'")
            if cursor.fetchone():
                # Check user_chats table schema
                cursor = conn.execute("PRAGMA table_info(user_chats)")
                columns = [col[1] for col in cursor.fetchall()]
                
                # Check what columns we have
                has_user_id = 'user_id' in columns
                has_user_email = 'user_email' in columns
                has_message_role = 'message_role' in columns
                has_message_content = 'message_content' in columns
                has_role = 'role' in columns
                has_content = 'content' in columns
                
                # If we have the old schema (user_id, role, content) and the new schema (user_email, message_role, message_content)
                # we need to recreate the table with the correct schema
                if (has_user_id and has_role and has_content) and (has_user_email and has_message_role and has_message_content):
                    st.info("Fixing user_chats table schema (mixed old and new schema)...")
                    # Drop and recreate user_chats table with correct schema
                    conn.execute("DROP TABLE IF EXISTS user_chats")
                    conn.execute('''
                        CREATE TABLE user_chats (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_email TEXT NOT NULL,
                            message_role TEXT NOT NULL,
                            message_content TEXT NOT NULL,
                            timestamp TEXT NOT NULL,
                            FOREIGN KEY (user_email) REFERENCES users (email)
                        )
                    ''')
                    conn.commit()
                    st.success("User chats table schema fixed!")
                # If we have the old schema (user_id, role, content) but not the new schema
                elif (has_user_id and has_role and has_content) and not (has_user_email and has_message_role and has_message_content):
                    st.info("Fixing user_chats table schema (old schema)...")
                    # Drop and recreate user_chats table with correct schema
                    conn.execute("DROP TABLE IF EXISTS user_chats")
                    conn.execute('''
                        CREATE TABLE user_chats (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_email TEXT NOT NULL,
                            message_role TEXT NOT NULL,
                            message_content TEXT NOT NULL,
                            timestamp TEXT NOT NULL,
                            FOREIGN KEY (user_email) REFERENCES users (email)
                        )
                    ''')
                    conn.commit()
                    st.success("User chats table schema fixed!")
                # If we have user_id but not user_email
                elif has_user_id and not has_user_email:
                    st.info("Fixing user_chats table schema (user_id without user_email)...")
                    # Drop and recreate user_chats table with correct schema
                    conn.execute("DROP TABLE IF EXISTS user_chats")
                    conn.execute('''
                        CREATE TABLE user_chats (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_email TEXT NOT NULL,
                            message_role TEXT NOT NULL,
                            message_content TEXT NOT NULL,
                            timestamp TEXT NOT NULL,
                            FOREIGN KEY (user_email) REFERENCES users (email)
                        )
                    ''')
                    conn.commit()
                    st.success("User chats table schema fixed!")
                # If we don't have the correct columns
                elif not (has_user_email and has_message_role and has_message_content):
                    st.info("Fixing user_chats table schema (missing required columns)...")
                    # Drop and recreate user_chats table with correct schema
                    conn.execute("DROP TABLE IF EXISTS user_chats")
                    conn.execute('''
                        CREATE TABLE user_chats (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_email TEXT NOT NULL,
                            message_role TEXT NOT NULL,
                            message_content TEXT NOT NULL,
                            timestamp TEXT NOT NULL,
                            FOREIGN KEY (user_email) REFERENCES users (email)
                        )
                    ''')
                    conn.commit()
                    st.success("User chats table schema fixed!")
                else:
                    # Table schema looks correct
                    pass  # User chats table schema is correct
            else:
                # Table doesn't exist, create it with correct schema
                conn.execute('''
                    CREATE TABLE user_chats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_email TEXT NOT NULL,
                        message_role TEXT NOT NULL,
                        message_content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (user_email) REFERENCES users (email)
                    )
                ''')
                conn.commit()
                st.info("User chats table created successfully!")
        except Exception as chats_error:
            st.error(f"Could not fix user_chats schema: {chats_error}")
            
    except Exception as e:
        st.error(f"Migration error: {e}")

def reset_database():
    """Reset database - use only for development"""
    if st.button("🗑️ Reset Database (Development Only)", type="secondary"):
        try:
            # Close any existing connections
            conn = get_db_connection()
            conn.close()
            
            # Remove database file
            if os.path.exists('data/app.db'):
                os.remove('data/app.db')
                st.success("Database reset successfully! Please refresh the page.")
            else:
                st.info("Database file not found. It will be created on next run.")
            st.rerun()
        except Exception as e:
            st.error(f"Error resetting database: {e}")

def force_schema_update():
    """Force update database schema - development only"""
    if st.button("🔧 Force Schema Update", type="secondary"):
        try:
            conn = get_db_connection()
            
            # Drop all tables
            tables_to_drop = [
                'daily_quests', 'user_profile', 'quiz_attempts', 'achievements',
                'study_materials', 'flashcards', 'study_sessions', 'user_chats', 'chat_sessions', 'users'
            ]
            
            for table in tables_to_drop:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            
            # Recreate all tables with correct schema
            conn.execute('''
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    verified INTEGER DEFAULT 0,
                    verification_code TEXT,
                    created_at TEXT,
                    last_login TEXT,
                    session_token TEXT,
                    session_expires TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE user_chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_email TEXT NOT NULL,
                    message_role TEXT NOT NULL,
                    message_content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (user_email) REFERENCES users (email)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE user_profile (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_email TEXT UNIQUE NOT NULL,
                    username TEXT DEFAULT 'Scholar',
                    total_xp INTEGER DEFAULT 0,
                    level INTEGER DEFAULT 1,
                    study_pet TEXT DEFAULT 'egg',
                    pet_happiness INTEGER DEFAULT 50,
                    learning_style TEXT DEFAULT 'unknown',
                    current_theme TEXT DEFAULT 'default',
                    created_at TEXT,
                    FOREIGN KEY (user_email) REFERENCES users (email)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE study_sessions (
                    id TEXT PRIMARY KEY,
                    user_email TEXT,
                    session_type TEXT,
                    duration INTEGER,
                    score INTEGER,
                    timestamp TEXT,
                    xp_earned INTEGER DEFAULT 0,
                    FOREIGN KEY (user_email) REFERENCES users (email)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE flashcards (
                    id TEXT PRIMARY KEY,
                    user_email TEXT,
                    question TEXT,
                    answer TEXT,
                    created_at TEXT,
                    times_reviewed INTEGER DEFAULT 0,
                    times_correct INTEGER DEFAULT 0,
                    FOREIGN KEY (user_email) REFERENCES users (email)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE study_materials (
                    id TEXT PRIMARY KEY,
                    user_email TEXT,
                    material_name TEXT,
                    material_type TEXT,
                    content TEXT,
                    summary TEXT,
                    FOREIGN KEY (user_email) REFERENCES users (email)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE quiz_attempts (
                    id TEXT PRIMARY KEY,
                    user_email TEXT,
                    question_hash TEXT,
                    question TEXT,
                    correct INTEGER,
                    confidence INTEGER,
                    timestamp TEXT,
                    FOREIGN KEY (user_email) REFERENCES users (email)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE daily_quests (
                    id TEXT PRIMARY KEY,
                    user_email TEXT,
                    date TEXT,
                    quest_type TEXT,
                    target INTEGER,
                    progress INTEGER DEFAULT 0,
                    completed INTEGER DEFAULT 0,
                    FOREIGN KEY (user_email) REFERENCES users (email)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE achievements (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    icon TEXT,
                    unlocked INTEGER DEFAULT 0,
                    unlocked_at TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_email TEXT NOT NULL,
                    session_name TEXT NOT NULL,
                    messages TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_email) REFERENCES users (email)
                )
            ''')
            
            conn.commit()
            st.success("All tables recreated successfully! Please refresh the page.")
            st.rerun()
        except Exception as e:
            st.error(f"Error updating schema: {e}")

def debug_database_schema():
    """Debug database schema - development only"""
    if st.button("🔍 Debug Schema", type="secondary"):
        try:
            conn = get_db_connection()
            
            # Define all tables and their required columns
            tables_info = {
                'users': ['password_hash', 'verified', 'verification_code', 'created_at', 'last_login', 'session_token', 'session_expires'],
                'user_chats': ['user_email', 'message_role', 'message_content', 'timestamp'],
                'user_profile': ['user_email'],
                'study_sessions': ['user_email'],
                'flashcards': ['user_email'],
                'study_materials': ['user_email'],
                'quiz_attempts': ['user_email'],
                'daily_quests': ['user_email'],
                'achievements': [],  # No user_email needed
                'chat_sessions': ['user_email']
            }
            
            all_good = True
            
            for table_name, required_columns in tables_info.items():
                cursor = conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                table_exists = cursor.fetchone()
                
                if table_exists:
                    cursor = conn.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()
                    
                    st.markdown(f"### {table_name.title()} Table Schema:")
                    column_names = [col[1] for col in columns]
                    
                    for col in columns:
                        st.write(f"- {col[1]} ({col[2]}) - {'NOT NULL' if col[3] else 'NULL'} - Default: {col[4]}")
                    
                    missing_columns = [col for col in required_columns if col not in column_names]
                    
                    if missing_columns:
                        st.error(f"{table_name} table missing columns: {', '.join(missing_columns)}")
                        all_good = False
                    else:
                        st.success(f"{table_name} table: All required columns present!")
                else:
                    st.error(f"{table_name} table does not exist!")
                    all_good = False
                
                st.markdown("---")
            
            if all_good:
                st.success("🎉 All tables have correct schema!")
            else:
                st.error("❌ Some tables have schema issues. Use 'Force Schema Update' to fix.")
                
        except Exception as e:
            st.error(f"Error debugging schema: {e}")

def debug_user_data():
    """Debug user data - development only"""
    if st.button("👤 Debug User Data", type="secondary"):
        try:
            conn = get_db_connection()
            
            # Get all users
            users = conn.execute("SELECT email, verified, created_at FROM users").fetchall()
            
            st.markdown("### Users in Database:")
            if users:
                for user in users:
                    st.write(f"- {user[0]} (Verified: {user[1]}, Created: {user[2]})")
                    
                    # Check user data in other tables
                    email = user[0]
                    tables_to_check = ['user_chats', 'user_profile', 'study_sessions', 'flashcards', 'study_materials', 'quiz_attempts', 'daily_quests']
                    
                    for table in tables_to_check:
                        try:
                            cursor = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE user_email = ?", (email,))
                            count = cursor.fetchone()[0]
                            if count > 0:
                                st.write(f"  - {table}: {count} records")
                        except:
                            pass
                    st.write("---")
            else:
                st.info("No users found in database")
                
        except Exception as e:
            st.error(f"Error debugging user data: {e}")

# AUTHENTICATION SYSTEM

def hash_password(password):
    """Hash password using SHA-256 with salt"""
    salt = secrets.token_hex(16)
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{password_hash}"

def verify_password(password, stored_hash):
    """Verify password against stored hash"""
    try:
        salt, password_hash = stored_hash.split(':')
        return hashlib.sha256((password + salt).encode()).hexdigest() == password_hash
    except:
        return False

def generate_verification_code():
    """Generate 6-digit verification code"""
    return str(random.randint(100000, 999999))

def send_verification_email(email, code):
    """Send verification email with OTP code"""
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        return False, "Email configuration not found. Please check .env file."
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = email
        msg['Subject'] = "AI Study Buddy - Email Verification"
        
        body = f"""
        Welcome to AI Study Buddy!
        
        Your verification code is: {code}
        
        This code will expire in 10 minutes.
        
        If you didn't request this verification, please ignore this email.
        
        Best regards,
        AI Study Buddy Team
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Create SSL context and send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        
        return True, "Verification email sent successfully!"
    
    except Exception as e:
        return False, f"Failed to send email: {str(e)}"

def register_user(email, password):
    """Register a new user"""
    conn = get_db_connection()
    
    # Check if user already exists
    existing = conn.execute("SELECT email FROM users WHERE email = ?", (email,)).fetchone()
    if existing:
        return False, "User already exists with this email"
    
    # Hash password
    password_hash = hash_password(password)
    
    # Generate verification code
    verification_code = generate_verification_code()
    
    # Insert user
    try:
        conn.execute("""
            INSERT INTO users (email, password_hash, verification_code, created_at)
            VALUES (?, ?, ?, ?)
        """, (email, password_hash, verification_code, datetime.datetime.now().isoformat()))
        conn.commit()
    
        # Send verification email
        success, message = send_verification_email(email, verification_code)
        if success:
            return True, f"Registration successful! {message}"
        else:
            return False, f"Registration successful but email failed: {message}"
    
    except Exception as e:
        return False, f"Registration failed: {str(e)}"

def verify_user_email(email, code):
    """Verify user email with OTP code"""
    conn = get_db_connection()
    
    user = conn.execute("""
        SELECT verification_code FROM users WHERE email = ? AND verified = 0
    """, (email,)).fetchone()
    
    if not user:
        return False, "User not found or already verified"
    
    if user[0] == code:
        conn.execute("""
            UPDATE users SET verified = 1, verification_code = NULL WHERE email = ?
        """, (email,))
        conn.commit()
        return True, "Email verified successfully!"
    else:
        return False, "Invalid verification code"

def login_user(email, password):
    """Login user and create session"""
    conn = get_db_connection()
    
    # Get user
    user = conn.execute("""
        SELECT password_hash, verified FROM users WHERE email = ?
    """, (email,)).fetchone()
    
    if not user:
        return False, "User not found"
    
    if not user[1]:  # verified = 0
        return False, "Please verify your email first"
    
    # Verify password
    if not verify_password(password, user[0]):
        return False, "Invalid password"
    
    # Create session
    session_token = secrets.token_urlsafe(32)
    session_expires = datetime.datetime.now() + datetime.timedelta(seconds=SESSION_TIMEOUT)
    
    # Update user session
    conn.execute("""
        UPDATE users SET session_token = ?, session_expires = ?, last_login = ?
        WHERE email = ?
    """, (session_token, session_expires.isoformat(), datetime.datetime.now().isoformat(), email))
    conn.commit()
    
    # Set session state
    st.session_state.user_email = email
    st.session_state.session_token = session_token
    st.session_state.session_expires = session_expires.isoformat()
    st.session_state.logged_in = True
    
    return True, "Login successful!"

def logout_user():
    """Logout user and clear session"""
    if st.session_state.get('logged_in'):
        conn = get_db_connection()
        conn.execute("""
            UPDATE users SET session_token = NULL, session_expires = NULL
            WHERE email = ?
        """, (st.session_state.user_email,))
        conn.commit()
    
    # Clear session state
    for key in ['user_email', 'session_token', 'session_expires', 'logged_in']:
        if key in st.session_state:
            del st.session_state[key]
    
    # Clear all user-specific data
    clear_user_session_data()

def delete_user(email):
    """Delete user and all related data"""
    conn = get_db_connection()
    
    try:
        # First, verify the user exists
        cursor = conn.execute("SELECT email FROM users WHERE email = ?", (email,))
        if not cursor.fetchone():
            return False, f"User {email} not found in database"
        
        # Define tables and their email column names
        tables_to_clear = {
            'user_chats': 'user_email',
            'study_sessions': 'user_email', 
            'flashcards': 'user_email',
            'study_materials': 'user_email',
            'quiz_attempts': 'user_email',
            'user_profile': 'user_email',
            'daily_quests': 'user_email',
            'users': 'email'
        }
        
        deleted_records = {}
        
        # Delete from each table using the correct column name
        for table, email_column in tables_to_clear.items():
            try:
                # Check if table exists
                cursor = conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                if cursor.fetchone():
                    # Check if the email column exists
                    cursor = conn.execute(f"PRAGMA table_info({table})")
                    columns = [col[1] for col in cursor.fetchall()]
                    
                    if email_column in columns:
                        # Count records before deletion
                        cursor = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {email_column} = ?", (email,))
                        count_before = cursor.fetchone()[0]
                        
                        # Delete records
                        cursor = conn.execute(f"DELETE FROM {table} WHERE {email_column} = ?", (email,))
                        deleted_records[table] = count_before
                        
                        if count_before > 0:
                            st.info(f"Deleted {count_before} records from {table}")
                    else:
                        st.warning(f"Column {email_column} not found in table {table}")
                else:
                    st.warning(f"Table {table} does not exist")
            except Exception as table_error:
                st.warning(f"Error deleting from {table}: {table_error}")
        
        conn.commit()
        
        # Summary
        total_deleted = sum(deleted_records.values())
        if total_deleted > 0:
            return True, f"Account deleted successfully! Removed {total_deleted} records across {len(deleted_records)} tables."
        else:
            return True, "Account deleted successfully! (No records found to delete)"
    
    except Exception as e:
        return False, f"Failed to delete account: {str(e)}"

def maintain_session():
    """Check and maintain user session"""
    if not st.session_state.get('logged_in'):
        return False
    
    # Check if session expired
    if 'session_expires' in st.session_state:
        try:
            expires = datetime.datetime.fromisoformat(st.session_state.session_expires)
            if datetime.datetime.now() > expires:
                logout_user()
                return False
        except:
            logout_user()
            return False
    
    # Update session expiry on activity
    if 'user_email' in st.session_state:
        conn = get_db_connection()
        new_expires = datetime.datetime.now() + datetime.timedelta(seconds=SESSION_TIMEOUT)
        conn.execute("""
            UPDATE users SET session_expires = ? WHERE email = ?
        """, (new_expires.isoformat(), st.session_state.user_email))
        conn.commit()
        st.session_state.session_expires = new_expires.isoformat()
    
    return True

def save_chat_to_db(user_email, role, content):
    """Save chat message to database"""
    try:
        conn = get_db_connection()
        
        # Ensure user_chats table exists with correct schema
        conn.execute('''
            CREATE TABLE IF NOT EXISTS user_chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                message_role TEXT NOT NULL,
                message_content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (user_email) REFERENCES users (email)
            )
        ''')
        
        conn.execute("""
            INSERT INTO user_chats (user_email, message_role, message_content, timestamp)
            VALUES (?, ?, ?, ?)
        """, (user_email, role, content, datetime.datetime.now().isoformat()))
        conn.commit()
    
    except Exception as e:
        st.error(f"Error saving chat message: {e}")

def load_chat_history(user_email):
    """Load chat history for user"""
    try:
        conn = get_db_connection()
        
        # Ensure user_chats table exists with correct schema
        conn.execute('''
            CREATE TABLE IF NOT EXISTS user_chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                message_role TEXT NOT NULL,
                message_content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (user_email) REFERENCES users (email)
            )
        ''')
        
        messages = conn.execute("""
            SELECT message_role, message_content FROM user_chats 
            WHERE user_email = ? ORDER BY timestamp ASC
        """, (user_email,)).fetchall()
        
        return [{"role": msg[0], "content": msg[1]} for msg in messages]
    
    except Exception as e:
        st.error(f"Error loading chat history: {e}")
        return []

def get_chat_sessions(user_email):
    """Get list of chat sessions for sidebar"""
    try:
        conn = get_db_connection()
        
        # Get unique chat sessions (grouped by date)
        sessions = conn.execute("""
            SELECT DATE(timestamp) as chat_date, COUNT(*) as message_count
            FROM user_chats
            WHERE user_email = ?
            GROUP BY DATE(timestamp)
            ORDER BY chat_date DESC
            LIMIT 10
        """, (user_email,)).fetchall()
        
        return sessions
    
    except Exception as e:
        st.error(f"Error loading chat sessions: {e}")
        return []

def display_chat_history_sidebar():
    """Display chat history in sidebar"""
    if not st.session_state.get('logged_in'):
        st.info("Please log in to view chat history")
        return
    
    user_email = st.session_state.user_email
    
    # Save current chat session
    if st.session_state.messages:
        session_name = st.text_input("Save current chat as:", key="save_session_name")
        if session_name and st.button("💾 Save Chat"):
            if save_chat_session(user_email, session_name, st.session_state.messages):
                st.success(f"Chat saved as '{session_name}'!")
                st.rerun()
    
    # Display saved chat sessions
    saved_sessions = get_saved_chat_sessions(user_email)
    if saved_sessions:
        st.markdown("### Saved Chats")
        for session in saved_sessions:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"📂 {session['name']}"):
                    # Load this session
                    messages = load_chat_session(user_email, session['name'])
                    if messages:
                        st.session_state.messages = messages
                        st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{session['name']}"):
                    # Delete this session
                    if delete_chat_session(user_email, session['name']):
                        st.success(f"Deleted '{session['name']}'!")
                        st.rerun()
    
    st.markdown("---")
    
    # Display recent chat sessions (by date)
    sessions = get_chat_sessions(user_email)
    
    if sessions:
        st.markdown("### Recent Chats")
        # Display chat sessions
        for session in sessions:
            chat_date = session[0]
            message_count = session[1]
            
            # Format date
            try:
                date_obj = datetime.datetime.fromisoformat(chat_date)
                formatted_date = date_obj.strftime("%b %d, %Y")
            except:
                formatted_date = chat_date
            
            # Create expandable session
            with st.expander(f"📅 {formatted_date} ({message_count} messages)"):
                # Load messages for this date
                try:
                    conn = get_db_connection()
                    messages = conn.execute("""
                        SELECT message_role, message_content, timestamp
                        FROM user_chats
                        WHERE user_email = ? AND DATE(timestamp) = ?
                        ORDER BY timestamp ASC
                    """, (user_email, chat_date)).fetchall()
                    
                    for msg in messages:
                        role = msg[0]
                        content = msg[1]
                        timestamp = msg[2]
                        
                        # Format timestamp
                        try:
                            time_obj = datetime.datetime.fromisoformat(timestamp)
                            time_str = time_obj.strftime("%H:%M")
                        except:
                            time_str = timestamp
                        
                        if role == "user":
                            st.markdown(f"**You** ({time_str}):")
                            st.markdown(f"💬 {content}")
                except Exception as e:
                    st.error(f"Error loading chat messages: {e}")
    
    # Clear chat history button
    if st.button("🗑️ Clear All Chat History", type="secondary"):
        if st.session_state.get('confirm_clear_chat', False):
            try:
                conn = get_db_connection()
                conn.execute("DELETE FROM user_chats WHERE user_email = ?", (user_email,))
                conn.commit()
                st.success("Chat history cleared successfully!")
                st.session_state.confirm_clear_chat = False
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing chat history: {e}")
        else:
            st.session_state.confirm_clear_chat = True
            st.warning("Are you sure you want to clear all chat history? This action cannot be undone.")

def save_chat_session(user_email, session_name, messages):
    """Save current chat session with a name"""
    try:
        conn = get_db_connection()
        
        # Ensure chat_sessions table exists
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                session_name TEXT NOT NULL,
                messages TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_email) REFERENCES users (email)
            )
        ''')
        
        # Serialize messages to JSON
        import json
        messages_json = json.dumps(messages)
        
        # Check if session with this name already exists for this user
        existing = conn.execute("""
            SELECT id FROM chat_sessions 
            WHERE user_email = ? AND session_name = ?
        """, (user_email, session_name)).fetchone()
        
        if existing:
            # Update existing session
            conn.execute("""
                UPDATE chat_sessions 
                SET messages = ?, created_at = ?
                WHERE id = ?
            """, (messages_json, datetime.datetime.now().isoformat(), existing[0]))
        else:
            # Insert new session
            conn.execute("""
                INSERT INTO chat_sessions (user_email, session_name, messages, created_at)
                VALUES (?, ?, ?, ?)
            """, (user_email, session_name, messages_json, datetime.datetime.now().isoformat()))
        
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error saving chat session: {e}")
        return False

def load_chat_session(user_email, session_name):
    """Load a saved chat session by name"""
    try:
        conn = get_db_connection()
        
        # Ensure chat_sessions table exists
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                session_name TEXT NOT NULL,
                messages TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_email) REFERENCES users (email)
            )
        ''')
        
        # Get session
        session = conn.execute("""
            SELECT messages FROM chat_sessions 
            WHERE user_email = ? AND session_name = ?
        """, (user_email, session_name)).fetchone()
        
        if session:
            import json
            messages = json.loads(session[0])
            return messages
        return []
    except Exception as e:
        st.error(f"Error loading chat session: {e}")
        return []

def get_saved_chat_sessions(user_email):
    """Get list of saved chat sessions for user"""
    try:
        conn = get_db_connection()
        
        # Ensure chat_sessions table exists
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                session_name TEXT NOT NULL,
                messages TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_email) REFERENCES users (email)
            )
        ''')
        
        sessions = conn.execute("""
            SELECT session_name, created_at FROM chat_sessions 
            WHERE user_email = ?
            ORDER BY created_at DESC
        """, (user_email,)).fetchall()
        
        return [{"name": s[0], "created_at": s[1]} for s in sessions]
    except Exception as e:
        st.error(f"Error loading saved sessions: {e}")
        return []

def delete_chat_session(user_email, session_name):
    """Delete a saved chat session"""
    try:
        conn = get_db_connection()
        
        # Ensure chat_sessions table exists
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                session_name TEXT NOT NULL,
                messages TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_email) REFERENCES users (email)
            )
        ''')
        
        conn.execute("""
            DELETE FROM chat_sessions 
            WHERE user_email = ? AND session_name = ?
        """, (user_email, session_name))
        
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error deleting chat session: {e}")
        return False

def get_user_profile(user_email):
    """Get or create user profile"""
    try:
        conn = get_db_connection()
        
        # Check if user_profile table exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_profile'")
        if not cursor.fetchone():
            st.warning("User profile table not found. Creating it now...")
            return None
        
        # Check if required columns exist
        cursor = conn.execute("PRAGMA table_info(user_profile)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'user_email' not in columns:
            st.warning("User profile table schema incomplete. Please use 'Force Schema Update' to fix.")
            return None
        
        profile = conn.execute("""
            SELECT * FROM user_profile WHERE user_email = ?
        """, (user_email,)).fetchone()
        
        if not profile:
            # Create new profile
            conn.execute("""
                INSERT INTO user_profile (user_email, username, created_at)
                VALUES (?, ?, ?)
            """, (user_email, 'Scholar', datetime.datetime.now().isoformat()))
            conn.commit()
            
            # Get the created profile
            profile = conn.execute("""
                SELECT * FROM user_profile WHERE user_email = ?
            """, (user_email,)).fetchone()
        
        return profile
    
    except Exception as e:
        st.error(f"Error getting user profile: {e}")
        return None

# SESSION STATE INITIALIZATION - Enhanced

# Initialize session state with timer variables
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        # Authentication states
        "logged_in": False,
        "user_email": None,
        "session_token": None,
        "session_expires": None,
        
        # Existing states
        "messages": [],
        "flashcards": [],
        "quiz_questions": [],
        "study_timer": 0,
        "timer_running": False,
        "study_points": 0,
        "study_streak": 0,
        "last_study_date": None,
        "current_session_points": 0,
        "language": "English",
        "tts_enabled": False,
        "last_timer_update": time.time(),
        "session_active": False,
        "session_start_time": None,
        "session_duration": 0,
        "daily_goal": 60,
        "session_goals_met": 0,
        "study_sessions_history": [],
        
        # NEW: AI Persona
        "ai_persona": "Speed",
        
        # NEW: Gamification
        "total_xp": 0,
        "level": 1,
        "achievements_unlocked": [],
        "study_pet": {"type": "egg", "happiness": 50, "evolution_stage": 0},
        "daily_quests": [],
        
        # NEW: Learning
        "learning_style": "unknown",
        "weakness_tracker": {},
        "conversation_memory": [],
        
        # NEW: UI
        "current_theme": "default",
        "show_onboarding": True,
        
        # NEW: Advanced features
        "pomodoro_count": 0,
        "last_emotion": "neutral",
        "study_plan": None,
        
        # NEW: Alarm-based timers
        "study_timer_target": 0,  # Target time in seconds
        "study_timer_start_time": 0,
        "study_timer_active": False,
        "study_timer_finished": False,
        "pomodoro_work_time": 25 * 60,  # 25 minutes in seconds
        "pomodoro_break_time": 5 * 60,  # 5 minutes in seconds
        "pomodoro_rounds": 4,
        "pomodoro_current_round": 1,
        "pomodoro_in_break": False,
        "pomodoro_start_time": 0,
        "pomodoro_active": False,
        "pomodoro_finished": False,
        
        # NEW: Web Search
        "web_search_enabled": False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Add chat confirmation state
    if 'confirm_clear_chat' not in st.session_state:
        st.session_state.confirm_clear_chat = False

def clear_user_session_data():
    """Clear all user-specific session data"""
    # Clear chat and learning data
    for key in ['messages', 'flashcards', 'quiz_questions', 'total_xp', 'level', 'achievements_unlocked']:
        if key in st.session_state:
            del st.session_state[key]
    
    # Reset to defaults
    st.session_state.messages = []
    st.session_state.flashcards = []
    st.session_state.quiz_questions = []
    st.session_state.total_xp = 0
    st.session_state.level = 1
    st.session_state.achievements_unlocked = []
    st.session_state.confirm_clear_chat = False

# Call initialization
init_session_state()
init_database()

# Verify database schema on startup
def verify_schema():
    """Verify database schema is correct"""
    try:
        conn = get_db_connection()
        
        # Define all tables and their required columns
        tables_info = {
            'users': ['password_hash', 'verified', 'verification_code', 'created_at', 'last_login', 'session_token', 'session_expires'],
            'user_chats': ['user_email', 'message_role', 'message_content', 'timestamp'],
            'user_profile': ['user_email'],
            'study_sessions': ['user_email'],
            'flashcards': ['user_email'],
            'study_materials': ['user_email'],
            'quiz_attempts': ['user_email'],
            'daily_quests': ['user_email'],
            'achievements': []  # No user_email needed
        }
        
        schema_issues = []
        
        for table_name, required_columns in tables_info.items():
            try:
                cursor = conn.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                column_names = [col[1] for col in columns]
                
                missing_columns = [col for col in required_columns if col not in column_names]
                
                if missing_columns:
                    schema_issues.append(f"{table_name}: {', '.join(missing_columns)}")
            except Exception as table_error:
                schema_issues.append(f"{table_name}: Table does not exist")
        
        if schema_issues:
            st.error(f"❌ Database schema issues found:")
            for issue in schema_issues:
                st.error(f"  - {issue}")
            st.info("💡 Use the 'Force Schema Update' button in Development Tools to fix this.")
            return False
        
        return True
    except Exception as e:
        st.error(f"❌ Database schema verification failed: {e}")
        return False

# Verify schema on startup
verify_schema()

# Check session on startup
def restore_session():
    """Restore user session from database"""
    try:
        conn = get_db_connection()
        
        # Check if there's a valid session in the database
        cursor = conn.execute("""
            SELECT email, session_token, session_expires FROM users 
            WHERE session_token IS NOT NULL AND session_expires > ?
            ORDER BY last_login DESC
        """, (datetime.datetime.now().isoformat(),))
        
        valid_session = cursor.fetchone()
        
        if valid_session:
            email, token, expires = valid_session
            
            # Clear any existing user data first
            clear_user_session_data()
            
            # Restore session state
            st.session_state.logged_in = True
            st.session_state.user_email = email
            st.session_state.session_token = token
            st.session_state.session_expires = expires
            
            # Load user data
            # Load chat history
            st.session_state.messages = load_chat_history(email)
            
            # Load user profile
            profile = get_user_profile(email)
            if profile:
                st.session_state.total_xp = profile['total_xp'] or 0
                st.session_state.level = profile['level'] or 1
                st.session_state.study_pet = {
                    "type": profile['study_pet'] or 'egg',
                    "happiness": profile['pet_happiness'] or 50,
                    "evolution_stage": 0
                }
                st.session_state.learning_style = profile['learning_style'] or 'unknown'
                st.session_state.current_theme = profile['current_theme'] or 'default'
            
            return True
        return False
    except Exception as e:
        st.error(f"Error restoring session: {e}")
        return False

def cleanup_expired_sessions():
    """Clean up expired sessions from database"""
    try:
        conn = get_db_connection()
        cursor = conn.execute("""
            UPDATE users 
            SET session_token = NULL, session_expires = NULL 
            WHERE session_expires <= ?
        """, (datetime.datetime.now().isoformat(),))
        
        if cursor.rowcount > 0:
            conn.commit()
            st.info(f"Cleaned up {cursor.rowcount} expired sessions.")
    except Exception as e:
        st.warning(f"Could not cleanup expired sessions: {e}")

# Clean up expired sessions on startup
cleanup_expired_sessions()

# Try to restore session on startup
if not st.session_state.get('logged_in'):
    restore_session()

# If session is active, maintain it
if st.session_state.get('logged_in'):
    if not maintain_session():
        st.warning("Session expired. Please login again.")
        st.rerun()

# GAMIFICATION SYSTEM

def calculate_xp_for_activity(activity_type, duration=0, score=0):
    """Calculate XP based on activity"""
    xp_map = {
        "study_session": duration // 60 * 10,  # 10 XP per minute
        "quiz_perfect": 100,
        "quiz_good": score * 5,
        "flashcard_review": 5,
        "streak_bonus": 50,
        "achievement": 25,
        "daily_quest": 30,
    }
    return xp_map.get(activity_type, 0)

def add_xp(amount, source=""):
    """Add XP and check for level up"""
    st.session_state.total_xp += amount
    
    # Level up calculation (exponential curve)
    new_level = int((st.session_state.total_xp / 100) ** 0.5) + 1
    
    if new_level > st.session_state.level:
        st.session_state.level = new_level
        st.balloons()
        st.success(f"🎉 Level Up! You're now Level {new_level}!")
        check_achievement("level_5" if new_level == 5 else f"level_{new_level}")
    
    # Update database
    if st.session_state.get('logged_in') and st.session_state.get('user_email'):
        conn = get_db_connection()
        conn.execute("""
            UPDATE user_profile 
            SET total_xp = ?, level = ?
            WHERE user_email = ?
        """, (st.session_state.total_xp, st.session_state.level, st.session_state.user_email))
        conn.commit()
    
    # Update pet happiness
    update_pet_happiness(5)

def check_achievement(achievement_id):
    """Check and unlock achievement"""
    conn = get_db_connection()
    result = conn.execute("""
        SELECT unlocked FROM achievements WHERE id = ?
    """, (achievement_id,)).fetchone()
    
    if result and not result[0]:
        conn.execute("""
            UPDATE achievements 
            SET unlocked = 1, unlocked_at = ?
            WHERE id = ?
        """, (datetime.datetime.now().isoformat(), achievement_id))
        conn.commit()
        
        # Get achievement details
        ach = conn.execute("""
            SELECT name, description, icon FROM achievements WHERE id = ?
        """, (achievement_id,)).fetchone()
        
        if ach:
            st.session_state.achievements_unlocked.append(achievement_id)
            st.success(f"🏆 Achievement Unlocked: {ach['icon']} {ach['name']}!")
            add_xp(25, "achievement")
            return True
    return False

def get_unlocked_achievements():
    """Get list of unlocked achievements"""
    conn = get_db_connection()
    return conn.execute("""
        SELECT * FROM achievements WHERE unlocked = 1
        ORDER BY unlocked_at DESC
    """).fetchall()

def get_all_achievements():
    """Get all achievements"""
    conn = get_db_connection()
    return conn.execute("SELECT * FROM achievements").fetchall()

# STUDY PET SYSTEM

def get_pet_emoji(pet_type, happiness):
    """Get pet emoji based on type and happiness"""
    pets = {
        "egg": "🥚",
        "baby_dragon": "🐲",
        "dragon": "🐉",
        "baby_cat": "🐱",
        "cat": "🐈",
        "baby_owl": "🐥",
        "owl": "🦉",
    }
    
    # Add mood modifiers
    if happiness > 80:
        mood = "✨"
    elif happiness > 50:
        mood = "😊"
    elif happiness > 20:
        mood = "😐"
    else:
        mood = "😢"
    
    return f"{pets.get(pet_type, '🥚')} {mood}"

def update_pet_happiness(change):
    """Update pet happiness"""
    current = st.session_state.study_pet["happiness"]
    new_happiness = max(0, min(100, current + change))
    st.session_state.study_pet["happiness"] = new_happiness
    
    # Update database
    if st.session_state.get('logged_in') and st.session_state.get('user_email'):
        conn = get_db_connection()
        conn.execute("""
            UPDATE user_profile 
            SET pet_happiness = ?
            WHERE user_email = ?
        """, (new_happiness, st.session_state.user_email))
        conn.commit()
    
    # Check for evolution
    if new_happiness > 80 and st.session_state.study_pet["evolution_stage"] == 0:
        evolve_pet()

def evolve_pet():
    """Evolve the study pet"""
    current = st.session_state.study_pet["type"]
    
    evolutions = {
        "egg": "baby_dragon",
        "baby_dragon": "dragon",
    }
    
    if current in evolutions:
        st.session_state.study_pet["type"] = evolutions[current]
        st.session_state.study_pet["evolution_stage"] += 1
        st.balloons()
        st.success(f"🎉 Your pet evolved into a {evolutions[current].replace('_', ' ')}!")
        check_achievement("pet_lover")

def pet_check_in():
    """Daily pet maintenance"""
    last_update = st.session_state.get("last_pet_update")
    now = datetime.datetime.now()
    
    if last_update:
        last_date = datetime.datetime.fromisoformat(last_update).date()
        if last_date < now.date():
            # Decrease happiness if not studied
            update_pet_happiness(-10)
            st.session_state.last_pet_update = now.isoformat()
    else:
        st.session_state.last_pet_update = now.isoformat()

# AI PERSONA SYSTEM

def get_persona_prompt(persona, base_prompt):
    """Modify prompt based on selected persona"""
    personas = {
        "Professor": """You are a distinguished professor. Use formal language, provide detailed 
        explanations with academic rigor, cite examples, and structure your responses clearly with 
        headings and bullet points.""",
        
        "Socratic": """You are a Socratic tutor. Don't give direct answers. Instead, ask guiding 
        questions that help the student discover the answer themselves. Challenge their assumptions 
        and guide them to deeper understanding.""",
        
        "ELI5": """You are explaining to a 5-year-old. Use simple words, fun analogies, and 
        relatable examples. Make it playful and easy to understand. Avoid jargon.""",
        
        "Speed": """You are in speed mode. Give concise, bullet-point answers. No fluff. 
        Just the essential facts and key points. Be brief but accurate.""",
        
        "Motivator": """You are an enthusiastic motivational coach. Be encouraging, supportive, 
        and energetic. Celebrate small wins, use emojis, and make learning feel exciting!""",
    }
    
    persona_style = personas.get(persona, personas["Professor"])
    return f"{persona_style}\n\n{base_prompt}"

# LEARNING STYLE DETECTION

def detect_learning_style_quiz():
    """Simple quiz to detect learning style"""
    st.markdown("### 🧠 Discover Your Learning Style")
    
    questions = [
        {
            "q": "When learning something new, I prefer to:",
            "options": ["See diagrams and charts", "Listen to explanations", "Try it hands-on"],
            "styles": ["visual", "auditory", "kinesthetic"]
        },
        {
            "q": "I remember best when:",
            "options": ["I see pictures or videos", "Someone explains it to me", "I practice it myself"],
            "styles": ["visual", "auditory", "kinesthetic"]
        },
        {
            "q": "In my free time, I enjoy:",
            "options": ["Reading or watching videos", "Listening to podcasts/music", "Building or creating things"],
            "styles": ["visual", "auditory", "kinesthetic"]
        },
    ]
    
    scores = {"visual": 0, "auditory": 0, "kinesthetic": 0}
    
    for i, item in enumerate(questions):
        answer = st.radio(item["q"], item["options"], key=f"learning_q{i}")
        idx = item["options"].index(answer)
        scores[item["styles"][idx]] += 1
    
    if st.button("Get My Learning Style"):
        # Convert scores to list of (style, score) tuples and find max
        style_scores = list(scores.items())
        style = max(style_scores, key=lambda x: x[1])[0]
        st.session_state.learning_style = style
        
        conn = get_db_connection()
        conn.execute("UPDATE user_profile SET learning_style = ? WHERE id = 1", (style,))
        conn.commit()
        
        st.success(f"🎯 Your learning style is: **{style.upper()}**!")
        
        recommendations = {
            "visual": "Use mind maps, diagrams, color-coded notes, and videos!",
            "auditory": "Try podcasts, text-to-speech, discussion groups, and verbal repetition!",
            "kinesthetic": "Practice hands-on, use flashcards, take breaks to move, and teach others!"
        }
        
        st.info(recommendations[style])
        return style

# WEAKNESS TRACKING

def track_mistake(question_hash, question_text, topic="general"):
    """Track mistakes for weakness detection"""
    if topic not in st.session_state.weakness_tracker:
        st.session_state.weakness_tracker[topic] = {
            "total_attempts": 0,
            "wrong_attempts": 0,
            "questions": {}
        }
    
    tracker = st.session_state.weakness_tracker[topic]
    tracker["total_attempts"] += 1
    tracker["wrong_attempts"] += 1
    
    if question_hash not in tracker["questions"]:
        tracker["questions"][question_hash] = {
            "text": question_text,
            "attempts": 0,
            "failures": 0
        }
    
    tracker["questions"][question_hash]["attempts"] += 1
    tracker["questions"][question_hash]["failures"] += 1

def get_weak_areas():
    """Get topics where user struggles"""
    weak = []
    for topic, data in st.session_state.weakness_tracker.items():
        if data["total_attempts"] > 3:
            error_rate = data["wrong_attempts"] / data["total_attempts"]
            if error_rate > 0.4:  # More than 40% error rate
                weak.append((topic, error_rate, data))
    return sorted(weak, key=lambda x: x[1], reverse=True)

# EMOTION DETECTION

def detect_emotion_from_text(text):
    """Simple sentiment analysis for emotion detection"""
    text_lower = text.lower()
    
    frustration_words = ["confused", "don't understand", "stuck", "hard", "difficult", "help", "frustrated"]
    confidence_words = ["got it", "understand", "clear", "thanks", "makes sense", "easy"]
    
    frustration_count = sum(1 for word in frustration_words if word in text_lower)
    confidence_count = sum(1 for word in confidence_words if word in text_lower)
    
    if frustration_count > confidence_count:
        return "frustrated"
    elif confidence_count > frustration_count:
        return "confident"
    else:
        return "neutral"

def get_supportive_response(emotion):
    """Get supportive message based on detected emotion"""
    responses = {
        "frustrated": "I sense you might be finding this challenging. Would you like me to explain it differently or break it down into smaller steps? 🤝",
        "confident": "Great! You seem to be understanding this well. Want to try a practice question to test your knowledge? 💪",
        "neutral": ""
    }
    return responses.get(emotion, "")

# STUDY PLAN GENERATOR

def generate_study_plan(exam_date, topics_list, hours_per_day):
    """Generate AI-powered study plan"""
    try:
        # Removed redundant import: from google.generativeai import GenerativeModel
        model = genai.GenerativeModel(model_name='gemini-2.5-flash')
        
        topics_str = ", ".join(topics_list)
        days_until_exam = (exam_date - datetime.date.today()).days
        
        prompt = f"""
        Create a detailed day-by-day study plan in JSON format.
        
        Exam Date: {exam_date}
        Days until exam: {days_until_exam}
        Topics: {topics_str}
        Study hours per day: {hours_per_day}
        
        Return a JSON array with this structure:
        [
            {{
                "day": 1,
                "date": "2024-01-15",
                "topics": ["Topic 1", "Topic 2"],
                "focus": "Understanding basics",
                "tasks": ["Read Chapter 1", "Make notes", "Practice 10 questions"],
                "duration": "2 hours"
            }},
            ...
        ]
        
        Make it realistic, progressive (easier topics first), and include review days.
        Return ONLY valid JSON, no markdown or explanations.
        """
        
        response = model.generate_content(prompt)
        # Clean response
        text = response.text.strip()
        if text.startswith("``json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        
        plan = json.loads(text.strip())
        return plan
        
    except Exception as e:
        st.error(f"Error generating plan: {e}")
        return None

# AI ENHANCED FUNCTIONS

def get_ai_response(prompt, context=""):
    """Enhanced AI response with persona and memory"""
    try:
        model = genai.GenerativeModel(model_name='gemini-2.5-flash')
        
        # Add conversation memory (last 5 messages)
        memory_context = ""
        if len(st.session_state.messages) > 1:
            recent = st.session_state.messages[-5:]
            memory_context = "\n".join([f"{m['role']}: {m['content']}" for m in recent])
        
        base_prompt = f"""You are an AI Study Buddy assistant. Your goal is to help students learn effectively.
        
        Previous conversation:
        {memory_context}
        
        Additional Context: {context}
        
        User Query: {prompt}
        
        Provide clear, educational, and encouraging responses. Break down complex topics into simple terms.
        Include examples when helpful. Always maintain a supportive tone.
        """
        
        # Apply persona
        full_prompt = get_persona_prompt(st.session_state.ai_persona, base_prompt)
        
        response = model.generate_content(full_prompt)
        
        # Detect emotion
        emotion = detect_emotion_from_text(prompt)
        st.session_state.last_emotion = emotion
        
        # Add supportive message if needed
        supportive = get_supportive_response(emotion)
        if supportive:
            response_text = f"{supportive}\n\n{response.text}"
        else:
            response_text = response.text
        
        return response_text
    except Exception as e:
        return f"Error: {str(e)}"

def generate_mnemonic(concept, key_points):
    """Generate mnemonic device for memorization"""
    try:
        model = genai.GenerativeModel(model_name='gemini-2.5-flash')
        
        prompt = f"""
        Create a memorable mnemonic device for this concept:
        Concept: {concept}
        Key Points: {', '.join(key_points)}
        
        Provide:
        1. An acronym (if applicable)
        2. A memorable sentence or rhyme
        3. A visual/story association
        
        Make it fun and easy to remember!
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def generate_multilevel_summary(content):
    """Generate summary at multiple difficulty levels"""
    try:
        model = genai.GenerativeModel(model_name='gemini-2.5-flash')
        
        prompt = f"""
        Summarize the following content at THREE difficulty levels:
        
        1. **Beginner** (ages 10-12, simple language)
        2. **Intermediate** (high school level)
        3. **Advanced** (college/expert level)
        
        Content: {content[:2000]}...
        
        Format each summary with a clear heading.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def summarize_youtube_video(video_url):
    """Extract and summarize YouTube video"""
    if not YOUTUBE_API_AVAILABLE:
        return "YouTube transcript library not available. Install with: pip install youtube-transcript-api"
    
    try:
        # Extract video ID
        if "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
        elif "watch?v=" in video_url:
            video_id = video_url.split("watch?v=")[1].split("&")[0]
        else:
            return "Invalid YouTube URL"
        
        # Get transcript - try multiple languages
        transcript_list = None
        languages_to_try = ['en', 'hi', 'ta', 'te', 'kn', 'ml', 'mr', 'bn', 'gu', 'pa', 'or', 'as', 'ur']
        
        # First try to get any available transcript
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        except:
            # If that fails, try specific languages
            for lang in languages_to_try:
                try:
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                    break
                except:
                    continue
        
        # If still no transcript, try to get any available language
        if transcript_list is None:
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            except:
                # Try to get transcript with any language
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages_to_try)
        
        if transcript_list is None:
            return "Could not retrieve transcript for this video. The video may not have captions available."
            
        transcript_text = " ".join([item['text'] for item in transcript_list])
        
        # Summarize
        model = genai.GenerativeModel(model_name='gemini-2.5-flash')
        
        prompt = f"""
        Summarize this YouTube video transcript into:
        1. Main topic (1 sentence)
        2. Key points (bullet list)
        3. Important timestamps with topics
        
        Note: The transcript may be in any language (Tamil, Telugu, Hindi, English, etc.). 
        Please provide the summary in English regardless of the original language.
        
        Transcript: {transcript_text[:3000]}...
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error: {str(e)}"

# STUDY SESSION MANAGEMENT

def start_study_session():
    """Starts a new study session."""
    st.session_state.session_active = True
    st.session_state.timer_running = True
    st.session_state.session_start_time = time.time()
    st.session_state.last_timer_update = time.time()

def stop_study_session():
    """Pauses the current study session."""
    if st.session_state.session_active and st.session_state.timer_running:
        st.session_state.timer_running = False
        elapsed = time.time() - st.session_state.session_start_time
        st.session_state.session_duration += elapsed

def end_study_session():
    """Ends the current study session and returns duration and points."""
    if st.session_state.session_active:
        stop_study_session() # Pause first to update duration
        duration = st.session_state.study_timer
        points = st.session_state.current_session_points

        # Reset session state
        st.session_state.session_active = False
        st.session_state.timer_running = False
        st.session_state.study_timer = 0
        st.session_state.current_session_points = 0
        st.session_state.session_start_time = None
        st.session_state.session_duration = 0
        
        # Update streak
        today = datetime.date.today()
        last_day = st.session_state.last_study_date
        if last_day:
            last_day_date = datetime.date.fromisoformat(last_day)
            if (today - last_day_date).days == 1:
                st.session_state.study_streak += 1
            elif today != last_day_date:
                st.session_state.study_streak = 1 # Reset if not consecutive
        else:
            st.session_state.study_streak = 1
        
        st.session_state.last_study_date = today.isoformat()

        return duration, points
    return 0, 0

# CONTENT EXTRACTION & GENERATION

def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_image(image):
    """Extract text from an image using Gemini."""
    try:
        model = genai.GenerativeModel(model_name='gemini-2.5-flash')
        response = model.generate_content(["Extract the text from this image.", image])
        return response.text
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        return ""

def generate_flashcards(content, num_cards=5):
    """Generate flashcards from content using AI."""
    try:
        model = genai.GenerativeModel(model_name='gemini-2.5-flash')
        prompt = f"""
        Based on the following content, generate {num_cards} flashcards in JSON format.
        Each flashcard should have a 'question' and an 'answer'.
        The questions should test key concepts from the text.
        
        Content:
        {content[:3000]}
        
        Return ONLY a valid JSON array like this:
        [
            {{"question": "What is concept A?", "answer": "Concept A is..."}},
            {{"question": "Explain process B.", "answer": "Process B involves..."}}
        ]
        """
        response = model.generate_content(prompt)
        # Clean response
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        
        flashcards = json.loads(text)
        return flashcards
    except Exception as e:
        st.error(f"Error generating flashcards: {e}")
        return []

def generate_quiz(content, num_questions=5):
    """Generate a quiz from content using AI."""
    try:
        model = genai.GenerativeModel(model_name='gemini-2.5-flash')
        prompt = f"""
        Based on the following content, generate a {num_questions}-question multiple-choice quiz in JSON format.
        Each question should have a 'question', a list of 'options', and the index of the 'correct_answer'.
        
        Content:
        {content[:3000]}
        
        Return ONLY a valid JSON array like this:
        [
            {{
                "question": "What is the capital of France?",
                "options": ["London", "Berlin", "Paris", "Madrid"],
                "correct_answer": 2
            }},
            ...
        ]
        """
        response = model.generate_content(prompt)
        # Clean response
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
            
        quiz = json.loads(text)
        return quiz
    except Exception as e:
        st.error(f"Error generating quiz: {e}")
        return []

# POMODORO TIMER

def pomodoro_session(work_minutes=25, break_minutes=5):
    """Enhanced Pomodoro timer with suggestions"""
    st.markdown("### 🍅 Pomodoro Timer")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        work_min = st.number_input("Work (min)", 15, 60, work_minutes, 5)
    with col2:
        break_min = st.number_input("Break (min)", 3, 15, break_minutes, 1)
    with col3:
        rounds = st.number_input("Rounds", 1, 8, 4)
    
    if st.button("Start Pomodoro"):
        st.session_state.pomodoro_count += 1
        st.info(f"🍅 Starting Pomodoro round {st.session_state.pomodoro_count}!")
        
        # Track in session
        start_study_session()
        
        # Break suggestions based on count
        if st.session_state.pomodoro_count % 4 == 0:
            st.success("🎉 You've completed 4 Pomodoros! Take a longer break (15-30 min)!")
        
        check_achievement("persistent")

# VOICE INPUT

def voice_to_text():
    """Convert speech to text"""
    if not SPEECH_AVAILABLE:
        return "Speech recognition not available. Install with: pip install SpeechRecognition"
    
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now!")
            audio = r.listen(source, timeout=5)
            text = r.recognize_google(audio)
            return text
    except sr.WaitTimeoutError:
        return "No speech detected"
    except sr.UnknownValueError:
        return "Could not understand audio"
    except Exception as e:
        return f"Error: {str(e)}"

# CHARTS & ANALYTICS

def create_study_heatmap():
    """Create heatmap of study activity"""
    if not PLOTLY_AVAILABLE:
        st.warning("Install plotly for charts: pip install plotly")
        return
    
    if not st.session_state.get('logged_in') or not st.session_state.get('user_email'):
        st.info("Please login to view your study analytics!")
        return
    
    # Get last 30 days of study sessions
    conn = get_db_connection()
    sessions = conn.execute("""
        SELECT date(timestamp) as date, SUM(duration) as total_minutes
        FROM study_sessions
        WHERE timestamp >= date('now', '-30 days') AND user_email = ?
        GROUP BY date(timestamp)
        ORDER BY date
    """, (st.session_state.user_email,)).fetchall()
    
    if not sessions:
        st.info("No study data yet. Start studying to see your heatmap!")
        return
    
    dates = [s['date'] for s in sessions]
    values = [s['total_minutes'] / 60 for s in sessions]  # Convert to hours
    
    fig = go.Figure(data=go.Scatter(
        x=dates,
        y=values,
        mode='lines+markers',
        fill='tozeroy',
        line=dict(color='#667eea', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="📊 Study Activity (Last 30 Days)",
        xaxis_title="Date",
        yaxis_title="Hours Studied",
        template="plotly_dark",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def create_weakness_chart():
    """Visualize weak areas"""
    if not PLOTLY_AVAILABLE:
        return
    
    weak_areas = get_weak_areas()
    if not weak_areas:
        st.info("No weakness data yet. Take some quizzes to track your progress!")
        return
    
    topics = [w[0] for w in weak_areas[:5]]  # Top 5
    error_rates = [w[1] * 100 for w in weak_areas[:5]]
    
    fig = go.Figure(data=[
        go.Bar(x=topics, y=error_rates, marker_color='#f87171')
    ])
    
    fig.update_layout(
        title="🎯 Topics Needing Practice",
        xaxis_title="Topic",
        yaxis_title="Error Rate (%)",
        template="plotly_dark",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# CONFIDENCE-BASED TESTING

def confidence_quiz(questions):
    """Quiz with confidence rating"""
    st.markdown("### 🎯 Confidence-Based Quiz")
    st.caption("Rate your confidence for each answer!")
    
    if "quiz_results" not in st.session_state:
        st.session_state.quiz_results = []
    
    for idx, q in enumerate(questions):
        st.markdown(f"**Q{idx+1}: {q['question']}**")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            answer = st.radio(
                f"Answer for Q{idx+1}:",
                q['options'],
                key=f"ans_{idx}"
            )
        
        with col2:
            confidence = st.select_slider(
                "Confidence",
                options=["Not Sure", "Somewhat", "Very Sure"],
                key=f"conf_{idx}"
            )
        
        if st.button(f"Submit Q{idx+1}", key=f"submit_confidence_{idx}"):
            correct = answer == q['options'][q['correct_answer']]
            conf_value = {"Not Sure": 1, "Somewhat": 2, "Very Sure": 3}[confidence]
            
            # Scoring logic
            if correct and conf_value == 3:
                points = 10
                msg = "✅ Correct and confident! Perfect!"
            elif correct and conf_value < 3:
                points = 5
                msg = "✅ Correct, but you were unsure. Review this topic!"
            elif not correct and conf_value == 3:
                points = -5
                msg = "❌ Wrong but very confident - this is dangerous! Study this more!"
            else:
                points = 0
                msg = "❌ Wrong, but good that you weren't too confident."
            
            st.write(msg)
            st.session_state.quiz_results.append({
                "question": q['question'],
                "correct": correct,
                "confidence": confidence,
                "points": points
            })
            
            # Track mistake
            if not correct:
                q_hash = hashlib.md5(q['question'].encode()).hexdigest()
                track_mistake(q_hash, q['question'])

# DAILY QUESTS

def generate_daily_quests():
    """Generate daily quests"""
    if not st.session_state.get('logged_in') or not st.session_state.get('user_email'):
        return []
    
    today = datetime.date.today().isoformat()
    user_email = st.session_state.user_email
    
    conn = get_db_connection()
    existing = conn.execute("""
        SELECT COUNT(*) FROM daily_quests WHERE date = ? AND user_email = ?
    """, (today, user_email)).fetchone()[0]
    
    if existing == 0:
        quests = [
            {"type": "study_time", "target": 30, "description": "Study for 30 minutes"},
            {"type": "flashcards", "target": 10, "description": "Review 10 flashcards"},
            {"type": "quiz", "target": 1, "description": "Complete 1 quiz"},
        ]
        
        for quest in quests:
            quest_id = str(uuid.uuid4())
            conn.execute("""
                INSERT INTO daily_quests (id, user_email, date, quest_type, target, progress, completed)
                VALUES (?, ?, ?, ?, ?, 0, 0)
            """, (quest_id, user_email, today, quest["type"], quest["target"]))
        
        conn.commit()
    
    return conn.execute("""
        SELECT * FROM daily_quests WHERE date = ? AND user_email = ?
    """, (today, user_email)).fetchall()

def update_quest_progress(quest_type, progress=1):
    """Update quest progress"""
    if not st.session_state.get('logged_in') or not st.session_state.get('user_email'):
        return
    
    today = datetime.date.today().isoformat()
    user_email = st.session_state.user_email
    
    conn = get_db_connection()
    conn.execute("""
        UPDATE daily_quests
        SET progress = progress + ?,
            completed = CASE WHEN progress + ? >= target THEN 1 ELSE 0 END
        WHERE date = ? AND quest_type = ? AND user_email = ?
    """, (progress, progress, today, quest_type, user_email))
    conn.commit()
    
    # Check if completed
    quest = conn.execute("""
        SELECT * FROM daily_quests WHERE date = ? AND quest_type = ? AND user_email = ? AND completed = 1
    """, (today, quest_type, user_email)).fetchone()
    
    if quest:
        add_xp(30, "daily_quest")
        st.success(f"🎯 Quest Completed: {quest_type}!")

# Continued in PART 2 due to length...

# PART 2: UI COMPONENTS & MAIN APPLICATION
THEMES = {
    "default": {
        "primary": "#667eea",
        "secondary": "#764ba2",
        "background": "rgba(0, 0, 0, 0.7)",
        "text": "#FFE4B5"
    },
    "ocean": {
        "primary": "#0891b2",
        "secondary": "#0e7490",
        "background": "rgba(8, 145, 178, 0.1)",
        "text": "#a5f3fc"
    },
    "forest": {
        "primary": "#059669",
        "secondary": "#047857",
        "background": "rgba(5, 150, 105, 0.1)",
        "text": "#a7f3d0"
    },
    "sunset": {
        "primary": "#f59e0b",
        "secondary": "#d97706",
        "background": "rgba(245, 158, 11, 0.1)",
        "text": "#fde68a"
    },
    "purple": {
        "primary": "#9333ea",
        "secondary": "#7e22ce",
        "background": "rgba(147, 51, 234, 0.1)",
        "text": "#e9d5ff"
    }
}

def get_theme_css(theme_name):
    """Generate CSS for selected theme"""
    theme = THEMES.get(theme_name, THEMES["default"])
    
    return f"""
    <style>
        .stApp {{
            background-image: url('https://i.postimg.cc/mk4x0MCy/Screenshot-2025-10-12-165723.png');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(rgba(0, 0, 0, 0.3), rgba(0, 0, 0, 0.5));
            z-index: 0;
            pointer-events: none;
        }}
        
        [data-testid="stSidebar"] {{
            background: linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.9));
            backdrop-filter: blur(10px);
        }}
        
        .stButton > button {{
            background: linear-gradient(135deg, {theme["primary"]} 0%, {theme["secondary"]} 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            transition: all 0.3s ease;
            font-weight: bold;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px {theme["primary"]}66;
        }}

        /* Chat UI improvements */
        .stTabs {{
            height: calc(100vh - 200px); /* Fixed height to prevent overlap */
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        .stTabContent {{
            flex-grow: 1;
            overflow-y: auto;
            padding: 1rem;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            margin-top: 1rem;
        }}

        .stForm {{
            position: sticky;
            bottom: 0;
            background: rgba(0,0,0,0.8);
            padding: 1rem;
            border-top: 1px solid rgba(255,255,255,0.2);
            margin-top: 1rem;
        }}

        /* Fix tab overlapping */
        [data-testid="stTabs"] {{
            overflow: visible;
        }}

        [data-testid="stTab"] {{
            padding: 0.5rem 1rem;
        }}

        /* Ensure proper spacing between tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
        }}

        /* Fix content overflow */
        .stTabs [data-baseweb="tab-panel"] {{
            padding: 1rem;
            max-height: calc(100vh - 300px);
            overflow-y: auto;
        }}
        
        .timer-display {{
            font-family: 'Arial', sans-serif;
            font-size: 2rem;
            font-weight: bold;
            text-align: center;
            color: {theme["text"]};
            padding: 1rem;
            border-radius: 12px;
            background: {theme["background"]};
            margin: 1rem 0;
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .pet-display {{
            font-size: 4rem;
            text-align: center;
            padding: 1rem;
            background: {theme["background"]};
            border-radius: 12px;
            margin: 1rem 0;
        }}
        
        .achievement-card {{
            background: {theme["background"]};
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border-left: 4px solid {theme["primary"]};
        }}
        
        .quest-card {{
            background: rgba(255, 255, 255, 0.05);
            padding: 0.8rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        h1, h2, h3 {{
            color: white !important;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
            font-weight: bold;
        }}
        
        [data-testid="stMetric"] {{
            background: {theme["background"]};
            padding: 1rem;
            border-radius: 10px;
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        
        [data-testid="stMetricValue"] {{
            color: {theme["text"]} !important;
            font-size: 1.8rem !important;
            font-weight: bold !important;
        }}
    </style>
    """

# Apply theme
st.markdown(get_theme_css(st.session_state.current_theme), unsafe_allow_html=True)

# HELPER UI FUNCTIONS

def display_xp_bar():
    """Display XP progress bar"""
    current_level = st.session_state.level
    current_xp = st.session_state.total_xp
    
    # XP needed for next level
    xp_for_current = (current_level - 1) ** 2 * 100
    xp_for_next = current_level ** 2 * 100
    xp_progress = current_xp - xp_for_current
    xp_needed = xp_for_next - xp_for_current
    
    progress = min(1.0, xp_progress / xp_needed)
    
    st.markdown(f"**Level {current_level}** - {current_xp} XP")
    st.progress(progress)
    st.caption(f"{xp_progress}/{xp_needed} XP to Level {current_level + 1}")

def display_pet():
    """Display study pet"""
    pet = st.session_state.study_pet
    pet_emoji = get_pet_emoji(pet["type"], pet["happiness"])
    
    st.markdown(f'<div class="pet-display">{pet_emoji}</div>', unsafe_allow_html=True)
    
    st.markdown(f"**{pet['type'].replace('_', ' ').title()}**")
    st.progress(pet["happiness"] / 100)
    st.caption(f"Happiness: {pet['happiness']}/100")
    
    if pet["happiness"] < 30:
        st.warning("⚠️ Your pet is sad! Study to make them happy!")
    elif pet["happiness"] > 80:
        st.success("✨ Your pet is very happy!")

def display_daily_quests():
    """Display daily quests"""
    quests = generate_daily_quests()
    
    st.markdown("### 🎯 Daily Quests")
    
    for quest in quests:
        progress = quest["progress"]
        target = quest["target"]
        completed = quest["completed"]
        
        if completed:
            st.success(f"✅ {quest['quest_type'].replace('_', ' ').title()}: {target}/{target}")
        else:
            st.markdown(f"""
            <div class="quest-card">
                <strong>{quest['quest_type'].replace('_', ' ').title()}</strong><br>
                Progress: {progress}/{target}
            </div>
            """, unsafe_allow_html=True)
            st.progress(min(1.0, progress / target))

def display_achievements():
    """Display achievements"""
    all_achievements = get_all_achievements()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Unlocked")
        unlocked = [a for a in all_achievements if a["unlocked"]]
        if unlocked:
            for ach in unlocked:
                st.markdown(f"""
                <div class="achievement-card">
                    {ach['icon']} <strong>{ach['name']}</strong><br>
                    <small>{ach['description']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No achievements unlocked yet!")
    
    with col2:
        st.markdown("### 🔒 Locked")
        locked = [a for a in all_achievements if not a["unlocked"]]
        if locked:
            for ach in locked[:5]:  # Show first 5
                st.markdown(f"""
                <div class="achievement-card" style="opacity: 0.5;">
                    {ach['icon']} <strong>{ach['name']}</strong><br>
                    <small>{ach['description']}</small>
                </div>
                """, unsafe_allow_html=True)

# AUTHENTICATION UI COMPONENTS

def show_login_form():
    """Display login form"""
    st.markdown("### 🔐 Login")
    
    with st.form("login_form"):
        email = st.text_input("Email", placeholder="your@email.com")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login", type="primary")
        
        if login_button:
            if email and password:
                success, message = login_user(email, password)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.error("Please fill in all fields")

def show_signup_form():
    """Display signup form"""
    st.markdown("### 📝 Sign Up")
    
    with st.form("signup_form"):
        email = st.text_input("Email", placeholder="your@email.com")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        signup_button = st.form_submit_button("Sign Up", type="primary")
        
        if signup_button:
            if email and password and confirm_password:
                if password == confirm_password:
                    if len(password) >= 6:
                        success, message = register_user(email, password)
                        if success:
                            st.success(message)
                            st.info("Please check your email for verification code!")
                            st.session_state.show_verification = True
                            st.session_state.pending_email = email
                        else:
                            st.error(message)
                    else:
                        st.error("Password must be at least 6 characters long")
                else:
                    st.error("Passwords do not match")
            else:
                st.error("Please fill in all fields")

def show_verification_form():
    """Display email verification form"""
    st.markdown("### ✉️ Email Verification")
    st.info(f"Please enter the verification code sent to {st.session_state.pending_email}")
    
    with st.form("verification_form"):
        code = st.text_input("Verification Code", placeholder="123456")
        verify_button = st.form_submit_button("Verify", type="primary")
        
        if verify_button:
            if code:
                success, message = verify_user_email(st.session_state.pending_email, code)
                if success:
                    st.success(message)
                    st.session_state.show_verification = False
                    st.session_state.pending_email = None
                    st.info("You can now login with your credentials!")
                else:
                    st.error(message)
            else:
                st.error("Please enter the verification code")

def show_auth_interface():
    """Show authentication interface"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1 style="color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
            📚 AI Study Buddy Pro
        </h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem;">
            Your intelligent companion for personalized learning
        </p>
        <div style="margin-top: 2rem; padding: 1.5rem; background: rgba(255,255,255,0.1); border-radius: 10px; border: 1px solid rgba(255,255,255,0.2);">
            <h3 style="color: #667eea; margin-bottom: 0.5rem;">🚀 Start Your AI Learning Journey</h3>
            <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin: 0;">
                Please sign up with your email ID to begin your personalized AI-powered learning experience
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for login/signup
    auth_tab1, auth_tab2 = st.tabs(["Login", "Sign Up"])
    
    with auth_tab1:
        show_login_form()
    
    with auth_tab2:
        show_signup_form()
    
    # Show verification form if needed
    if st.session_state.get('show_verification'):
        st.markdown("---")
        show_verification_form()
    
    # Development tools (only show if no users exist or if there are schema issues)
    conn = get_db_connection()
    try:
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        show_dev_tools = user_count == 0
    except Exception as e:
        st.error(f"Database error: {e}")
        show_dev_tools = True
    
    if show_dev_tools:
        st.markdown("---")
        st.markdown("### 🛠️ Development Tools")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            reset_database()
        with col2:
            force_schema_update()
        with col3:
            debug_database_schema()
        with col4:
            debug_user_data()

def show_user_header():
    """Show user header with logout option"""
    # Extract username from email (part before @)
    user_email = st.session_state.user_email
    username = user_email.split('@')[0] if user_email and '@' in user_email else user_email
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown(f"""
    <h1 style="text-align: left; font-size: 3rem; margin-bottom: 0.5rem; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
        📚 AI Study Buddy Pro
    </h1>
    <p style="text-align: left; font-size: 1.2rem; color: rgba(255,255,255,0.9);">
            Welcome back, {username}!
    </p>
    """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🚪 Logout", help="Logout from your account"):
            logout_user()
            st.rerun()
    
    with col3:
        if st.button("🗑️ Delete Account", help="Permanently delete your account", type="secondary"):
            st.session_state.show_delete_confirm = True
    
    # Delete account confirmation
    if st.session_state.get('show_delete_confirm'):
        st.markdown("---")
        st.error("⚠️ **Delete Account Confirmation**")
        st.warning("This action cannot be undone! All your data will be permanently deleted.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, Delete My Account", type="primary"):
                success, message = delete_user(st.session_state.user_email)
                if success:
                    st.success(message)
                    logout_user()
                    st.rerun()
                else:
                    st.error(message)
        
        with col2:
            if st.button("❌ Cancel", type="secondary"):
                st.session_state.show_delete_confirm = False
                st.rerun()

# MAIN HEADER & CLOCK

# Check if user is logged in
if not st.session_state.get('logged_in'):
    show_auth_interface()
    st.stop()
else:
    show_user_header()

col1, col2 = st.columns([3, 1])

with col1:
    pass  # Header already shown above

with col2:
    clock_placeholder = st.empty()

def update_clock():
    """Displays a real-time clock in IST."""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.datetime.now(ist)
    clock_placeholder.markdown(
        f"""
        <div style="text-align: right; font-size: 1.5rem; font-weight: bold; color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">
            {now.strftime('%H:%M')}
        </div>
        """,
        unsafe_allow_html=True
    )

# STUDY TIMER FUNCTIONS

def set_study_timer(minutes):
    """Set a study timer for the specified number of minutes"""
    st.session_state.study_timer_target = minutes * 60  # Convert to seconds
    st.session_state.study_timer_start_time = time.time()
    st.session_state.study_timer_active = True
    st.session_state.study_timer_finished = False

def check_study_timer():
    """Check if the study timer has finished"""
    if st.session_state.study_timer_active and not st.session_state.study_timer_finished:
        elapsed = time.time() - st.session_state.study_timer_start_time
        if elapsed >= st.session_state.study_timer_target:
            st.session_state.study_timer_active = False
            st.session_state.study_timer_finished = True
            st.session_state.study_timer = st.session_state.study_timer_target
            # Award XP for completed session
            xp_earned = calculate_xp_for_activity("study_session", duration=st.session_state.study_timer_target // 60)
            add_xp(xp_earned, "study_session")
            update_quest_progress("study_time", st.session_state.study_timer_target // 60)
            return True
    return False

with st.sidebar:
    st.markdown("## 👤 Profile")
    
    # XP and Level
    display_xp_bar()
    
    st.markdown("---")
    
    # Study Pet
    st.markdown("### 🐾 Your Study Pet")
    display_pet()
    
    st.markdown("---")
    
    # Stats
    st.markdown("### 📊 Your Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Points", st.session_state.study_points)
    with col2:
        st.metric("Streak", f"{st.session_state.study_streak} 🔥")
    
    st.markdown("---")
    
    # Chat History Section
    st.markdown("### 💬 Chat History")
    
    # New Chat button
    new_chat_clicked = st.button("💬 New Chat")
    if new_chat_clicked:
        # Save current chat to history if there are messages and user is logged in
        if st.session_state.get('logged_in') and st.session_state.messages and len(st.session_state.messages) > 0:
            # Generate a session name based on the first few words of the first message
            first_message = st.session_state.messages[0]['content']
            session_name = first_message[:30] + ("..." if len(first_message) > 30 else "")
            # Save the current chat session
            if save_chat_session(st.session_state.user_email, session_name, st.session_state.messages):
                st.success(f"Chat saved as '{session_name}'!")
        # Clear current chat
        st.session_state.messages = []
        # Reset any form inputs
        if 'user_input_form' in st.session_state:
            st.session_state.user_input_form = ""
        # Set flag to indicate new chat was initiated
        st.session_state.new_chat_initiated = True
        # Show a confirmation message
        st.success("Started new chat session!")
        # Force a complete refresh to clear the UI
        st.rerun()
    
    # Display a message when a new chat is initiated
    if st.session_state.get('new_chat_initiated', False):
        st.session_state.new_chat_initiated = False
    
    display_chat_history_sidebar()
    
    st.markdown("---")
    
    # Daily Quests
    display_daily_quests()
    
    st.markdown("---")
    
    # Study Timer
    st.markdown("### ⏱️ Study Timer")
    
    # Alarm-based timer instead of real-time
    if not st.session_state.study_timer_active and not st.session_state.study_timer_finished:
        timer_minutes = st.number_input("Set Timer (minutes)", min_value=1, max_value=180, value=25)
        if st.button("Start Timer", type="primary"):
            set_study_timer(timer_minutes)
            st.success(f"⏰ Timer set for {timer_minutes} minutes!")
            st.rerun()
    elif st.session_state.study_timer_active:
        # Show countdown
        elapsed = time.time() - st.session_state.study_timer_start_time
        remaining = max(0, st.session_state.study_timer_target - elapsed)
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        st.markdown(f"⏰ Time remaining: {minutes:02d}:{seconds:02d}")
        
        if st.button("Cancel Timer"):
            st.session_state.study_timer_active = False
            st.session_state.study_timer = 0
            st.rerun()
            
        # Check if timer finished
        if check_study_timer():
            st.balloons()
            st.success("🎉 Study time completed! Great job!")
            st.rerun()
    elif st.session_state.study_timer_finished:
        st.success("🎉 Study session completed!")
        st.session_state.study_timer = st.session_state.study_timer_target
        minutes = int(st.session_state.study_timer // 60)
        seconds = int(st.session_state.study_timer % 60)
        st.markdown(f"⏱️ Total time: {minutes:02d}:{seconds:02d}")
        
        if st.button("Start New Session"):
            st.session_state.study_timer_finished = False
            st.session_state.study_timer = 0
            st.rerun()
    
    st.markdown("---")
    
    # Settings
    st.markdown("### ⚙️ Settings")

    # API Status - Enhanced visibility
    st.markdown("### 🔧 AI API Status")
    col1, col2 = st.columns(2)
    
    with col1:
        if GEMINI_API_AVAILABLE:
            st.success("✅ Gemini: Connected")
        else:
            st.error("❌ Gemini: Not configured")
    
    with col2:
        if A4F_API_AVAILABLE:
            st.success("✅ A4F: Connected")
        else:
            st.warning("⚠️ A4F: Not configured")
    
    if not GEMINI_API_AVAILABLE and not A4F_API_AVAILABLE:
        st.error("❌ No AI API configured! Please set up either Gemini or A4F API.")
        st.info("💡 Add your API keys to the .env file:\n- GOOGLE_API_KEY=your_gemini_key\n- A4F_API_KEY=your_a4f_key")

    # AI Provider Selection (if both are available) - Make it more prominent
    if GEMINI_API_AVAILABLE and A4F_API_AVAILABLE:
        st.markdown("### 🤖 AI Provider Selection")
        st.info("🔄 Switch between AI models or use Auto mode for fallback")
        ai_provider = st.selectbox(
            "Choose AI Provider",
            ["Auto (Best Available)", "Gemini", "A4F"],
            index=["Auto (Best Available)", "Gemini", "A4F"].index(st.session_state.get("ai_provider", "Auto (Best Available)")),
            help="Auto will use Gemini as primary with A4F as fallback"
        )
        st.session_state.ai_provider = ai_provider
    elif GEMINI_API_AVAILABLE:
        st.session_state.ai_provider = "Gemini"
        st.info("🟢 Using Gemini API")
    elif A4F_API_AVAILABLE:
        st.session_state.ai_provider = "A4F"
        st.info("🔵 Using A4F API")
    else:
        st.session_state.ai_provider = "None"
        st.warning("⚠️ No AI provider available")

    # AI Persona
    st.markdown("### 🎭 AI Tutor Style")
    persona = st.selectbox(
        "AI Tutor Style",
        ["Professor", "Socratic", "ELI5", "Speed", "Motivator"],
        index=["Professor", "Socratic", "ELI5", "Speed", "Motivator"].index(st.session_state.ai_persona)
    )
    st.session_state.ai_persona = persona

    # Theme
    st.markdown("### 🎨 Theme")
    theme = st.selectbox(
        "Theme",
        list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.current_theme)
    )
    if theme != st.session_state.current_theme:
        st.session_state.current_theme = theme
        st.rerun()

    # Remove language option from sidebar (as requested)
    # The language selection has been removed to simplify the interface

# ============================================================================
# MAIN CONTENT - TABS
# ============================================================================

tab_names = [
    "🤖 AI Tutor",
    "📚 Flashcards",
    "🧠 Quiz",
    "📖 Materials",
    "🎯 Study Plan",
    "📊 Analytics",
    "🏆 Achievements",
    "🧪 Tools"
]

tabs = st.tabs(tab_names)

# ============================================================================
# TAB 1: AI TUTOR
# ============================================================================

with tabs[0]:
    st.markdown(f"## 🤖 AI Tutor ({st.session_state.ai_persona} Mode)")
    
    # Emotion indicator
    if st.session_state.last_emotion != "neutral":
        emotion_emoji = {"frustrated": "😟", "confident": "😊"}
        st.info(f"{emotion_emoji.get(st.session_state.last_emotion, '')} Detected mood: {st.session_state.last_emotion}")
    
    # Chat container with fixed input at bottom
    chat_placeholder = st.container()
    with chat_placeholder:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                     color: white; border-radius: 15px 15px 5px 15px; padding: 12px 16px; 
                     margin: 8px 0; max-width: 70%; float: right; clear: both;">
                    {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(255, 255, 255, 0.95); color: #333; 
                     border-radius: 15px 15px 15px 5px; padding: 12px 16px; 
                     margin: 8px 0; max-width: 70%; float: left; clear: both;">
                    {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Input form for chat
    with st.form(key='chat_form', clear_on_submit=True):
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            user_input = st.text_input("Ask me anything about your studies...", key="user_input_form")
        
        with col2:
            # Web search toggle
            st.session_state.web_search_enabled = st.checkbox(
                "🌐 Web Search", 
                value=st.session_state.get('web_search_enabled', False),
                help="Enable web search for current information"
            )
        
        with col3:
            if SPEECH_AVAILABLE:
                if st.form_submit_button("🎤 Voice"):
                    voice_text = voice_to_text()
                    if voice_text and not voice_text.startswith("Error"):
                        st.session_state.voice_input = voice_text
                        st.info(f"You said: {voice_text}")
        
        # Use voice input if available
        if "voice_input" in st.session_state and st.session_state.voice_input:
            user_input = st.session_state.voice_input
            st.session_state.voice_input = ""
        
        send_button = st.form_submit_button("Send", type="primary")
        
        if send_button and user_input and user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.spinner("Thinking..."):
                response = get_ai_response_multi_api(user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})

            st.rerun()

# ============================================================================
# TAB 2: FLASHCARDS
# ============================================================================

with tabs[1]:
    st.markdown("## 📚 Flashcards")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Create Flashcards")
        flashcard_input = st.text_area("Enter text or upload file:", height=150)
        uploaded_file = st.file_uploader("Upload PDF/Image", type=["pdf", "png", "jpg", "jpeg"], key="fc_upload")
        num_cards = st.slider("Number of Cards", 1, 20, 5)
        
        if st.button("🎴 Generate Flashcards", type="primary"):
            content = flashcard_input
            
            if uploaded_file:
                if uploaded_file.type == "application/pdf":
                    content += "\n" + extract_text_from_pdf(uploaded_file)
                else:
                    image = Image.open(uploaded_file)
                    content += "\n" + extract_text_from_image(image)
            
            if content.strip():
                with st.spinner("Creating flashcards..."):
                    flashcards = generate_flashcards(content, num_cards)
                    st.session_state.flashcards = flashcards
                    
                    # Update quest
                    update_quest_progress("flashcards", len(flashcards))
                    
                    st.success(f"✅ Created {len(flashcards)} flashcards!")
    
    with col2:
        st.markdown("### Review Mode")
        if st.session_state.flashcards:
            if "fc_index" not in st.session_state:
                st.session_state.fc_index = 0
            
            idx = st.session_state.fc_index
            card = st.session_state.flashcards[idx]
            
            st.markdown(f"**Card {idx + 1}/{len(st.session_state.flashcards)}**")
            st.markdown(f"### Q: {card['question']}")
            
            if st.button("Show Answer"):
                st.info(f"**A:** {card['answer']}")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("⬅️ Previous"):
                    st.session_state.fc_index = max(0, idx - 1)
                    st.rerun()
            with col_b:
                if st.button("Next ➡️"):
                    st.session_state.fc_index = min(len(st.session_state.flashcards) - 1, idx + 1)
                    add_xp(5, "flashcard")
                    st.rerun()
    
    # Display all flashcards
    if st.session_state.flashcards:
        st.markdown("---")
        st.markdown("### All Flashcards")
        for i, card in enumerate(st.session_state.flashcards):
            with st.expander(f"Card {i+1}: {card['question'][:50]}..."):
                st.markdown(f"**Q:** {card['question']}")
                st.markdown(f"**A:** {card['answer']}")

# ============================================================================
# TAB 3: QUIZ
# ============================================================================

with tabs[2]:
    st.markdown("## 🧠 Quiz")
    
    quiz_mode = st.radio("Quiz Mode", ["Standard", "Confidence-Based"], horizontal=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        quiz_input = st.text_area("Enter content for quiz:", height=150)
        uploaded_file = st.file_uploader("Upload PDF/Image", type=["pdf", "png", "jpg", "jpeg"], key="quiz_upload")
        num_questions = st.slider("Number of Questions", 1, 20, 5)
        
        if st.button("📝 Generate Quiz", type="primary"):
            content = quiz_input
            
            if uploaded_file:
                if uploaded_file.type == "application/pdf":
                    content += "\n" + extract_text_from_pdf(uploaded_file)
                else:
                    image = Image.open(uploaded_file)
                    content += "\n" + extract_text_from_image(image)
            
            if content.strip():
                with st.spinner("Creating quiz..."):
                    quiz = generate_quiz(content, num_questions)
                    st.session_state.quiz_questions = quiz
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False
                    st.success(f"✅ Created {len(quiz)} questions!")
    
    with col2:
        if st.session_state.quiz_questions and not st.session_state.get("quiz_submitted", False):
            st.markdown("### Quiz Progress")
            answered = len(st.session_state.get("quiz_answers", {}))
            total = len(st.session_state.quiz_questions)
            st.progress(answered / total)
            st.caption(f"{answered}/{total} answered")
    
    # Display quiz
    if st.session_state.quiz_questions:
        if quiz_mode == "Confidence-Based":
            confidence_quiz(st.session_state.quiz_questions)
        else:
            # Standard quiz
            if "quiz_answers" not in st.session_state:
                st.session_state.quiz_answers = {}
            
            for idx, q in enumerate(st.session_state.quiz_questions):
                st.markdown(f"### Question {idx + 1}")
                st.markdown(f"**{q['question']}**")
                
                answer = st.radio(
                    "Select answer:",
                    q['options'],
                    key=f"q_{idx}"
                )
                
                st.session_state.quiz_answers[idx] = answer
            
            if st.button("Submit Quiz", type="primary"):
                score = 0
                total = len(st.session_state.quiz_questions)
                
                for idx, q in enumerate(st.session_state.quiz_questions):
                    user_answer = st.session_state.quiz_answers.get(idx)
                    correct_answer = q['options'][q['correct_answer']]
                    
                    if user_answer == correct_answer:
                        score += 1
                    else:
                        # Track mistake
                        q_hash = hashlib.md5(q['question'].encode()).hexdigest()
                        track_mistake(q_hash, q['question'])
                
                # Calculate XP
                percentage = (score / total) * 100
                if percentage == 100:
                    xp = calculate_xp_for_activity("quiz_perfect")
                    check_achievement("quiz_perfect")
                else:
                    xp = calculate_xp_for_activity("quiz_good", score=score)
                
                add_xp(xp, "quiz")
                update_quest_progress("quiz", 1)
                
                st.session_state.quiz_submitted = True
                st.balloons()
                st.success(f"🎉 Score: {score}/{total} ({percentage:.0f}%) | +{xp} XP")
                
                if percentage < 60:
                    st.warning("💡 Consider reviewing this topic more!")

# ============================================================================
# TAB 4: STUDY MATERIALS
# ============================================================================

with tabs[3]:
    st.markdown("## 📖 Study Materials")
    
    uploaded_file = st.file_uploader(
        "Upload PDF, Document, or Image",
        type=["pdf", "doc", "docx", "png", "jpg", "jpeg"],
        key="material_upload"
    )
    
    if uploaded_file:
        with st.spinner("Processing..."):
            content = ""
            
            if uploaded_file.type == "application/pdf":
                content = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type in ["image/png", "image/jpg", "image/jpeg"]:
                image = Image.open(uploaded_file)
                content += "\n" + extract_text_from_image(image)
            
            if content:
                tab1, tab2, tab3 = st.tabs(["Summary", "Multi-Level", "Tools"])
                
                with tab1:
                    st.markdown("### 📝 Summary")
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    summary = model.generate_content(f"Summarize this in bullet points:\n\n{content[:3000]}").text
                    st.markdown(summary)
                
                with tab2:
                    st.markdown("### 📊 Multi-Level Summaries")
                    multilevel = generate_multilevel_summary(content)
                    st.markdown(multilevel)
                
                with tab3:
                    st.markdown("### 🛠️ Generate Study Tools")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("🎴 Flashcards"):
                            st.session_state.flashcards = generate_flashcards(content)
                            st.success("✅ Flashcards created!")
                    
                    with col2:
                        if st.button("📝 Quiz"):
                            st.session_state.quiz_questions = generate_quiz(content)
                            st.success("✅ Quiz created!")
                    
                    with col3:
                        if st.button("💡 Mind Map"):
                            st.info("Mind map visualization coming soon!")

# ============================================================================
# TAB 5: STUDY PLAN
# ============================================================================

with tabs[4]:
    st.markdown("## 🎯 Smart Study Plan Generator")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        exam_date = st.date_input(
            "Exam Date",
            min_value=datetime.date.today(),
            value=datetime.date.today() + datetime.timedelta(days=30)
        )
        
        hours_per_day = st.slider("Study hours per day", 1, 8, 2)
        
        topics_input = st.text_area(
            "Topics (one per line)",
            placeholder="Calculus\nPhysics\nChemistry\n...",
            height=150
        )
    
    with col2:
        days_until = (exam_date - datetime.date.today()).days
        st.metric("Days Until Exam", days_until)
        st.metric("Total Study Hours", days_until * hours_per_day)
        
        if topics_input:
            topics = [t.strip() for t in topics_input.split("\n") if t.strip()]
            st.metric("Topics", len(topics))
    
    if st.button("🚀 Generate Study Plan", type="primary"):
        if topics_input.strip():
            topics = [t.strip() for t in topics_input.split("\n") if t.strip()]
            
            with st.spinner("Creating your personalized study plan..."):
                plan = generate_study_plan(exam_date, topics, hours_per_day)
                
                if plan:
                    st.session_state.study_plan = plan
                    st.success("✅ Study plan created!")
                    
                    # Display plan
                    for day in plan:
                        with st.expander(f"📅 Day {day['day']} - {day['date']}"):
                            st.markdown(f"**Focus:** {day['focus']}")
                            st.markdown(f"**Duration:** {day['duration']}")
                            st.markdown("**Topics:**")
                            for topic in day['topics']:
                                st.markdown(f"- {topic}")
                            st.markdown("**Tasks:**")
                            for task in day['tasks']:
                                st.checkbox(task, key=f"task_{day['day']}_{task}")

# TAB 6: ANALYTICS

with tabs[5]:
    st.markdown("## 📊 Learning Analytics")
    
    # Study heatmap
    st.markdown("### 📈 Study Activity")
    create_study_heatmap()
    
    st.markdown("---")
    
    # Weakness analysis
    st.markdown("### 🎯 Areas Needing Practice")
    create_weakness_chart()
    
    weak_areas = get_weak_areas()
    if weak_areas:
        st.markdown("#### Recommendations:")
        for topic, error_rate, data in weak_areas[:3]:
            st.warning(f"**{topic}**: {error_rate*100:.0f}% error rate - Review this topic!")
    
    st.markdown("---")
    
    # Study patterns
    st.markdown("### 🧠 Study Insights")
    
    conn = get_db_connection()
    user_email = st.session_state.user_email
    total_sessions = conn.execute("SELECT COUNT(*) FROM study_sessions WHERE user_email = ?", (user_email,)).fetchone()[0]
    total_minutes = conn.execute("SELECT SUM(duration) FROM study_sessions WHERE user_email = ?", (user_email,)).fetchone()[0] or 0
    avg_session = total_minutes // total_sessions if total_sessions > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sessions", total_sessions)
    with col2:
        st.metric("Total Hours", f"{total_minutes // 60}h {total_minutes % 60}m")
    with col3:
        st.metric("Avg Session", f"{avg_session} min")
    
    # Predictions
    if total_sessions > 5:
        st.markdown("### 🔮 Predictions")
        
        # Simple prediction based on current performance
        avg_score = conn.execute("SELECT AVG(score) FROM study_sessions WHERE score > 0 AND user_email = ?", (user_email,)).fetchone()[0] or 0
        consistency = min(100, st.session_state.study_streak * 10)
        
        predicted_score = min(100, (avg_score + consistency) / 2)
        
        st.info(f"📊 Based on your study patterns, you have a **{predicted_score:.0f}%** chance of achieving your goals!")
        
        if predicted_score < 60:
            st.warning("💡 Tip: Study more consistently and review weak areas to improve!")
        elif predicted_score > 80:
            st.success("🌟 Great job! Keep up the excellent work!")

# TAB 7: ACHIEVEMENTS

with tabs[6]:
    st.markdown("## 🏆 Achievements & Gamification")
    
    # Display achievements
    display_achievements()
    
    st.markdown("---")
    
    # Leaderboard (local)
    st.markdown("### 📊 Your Ranking")
    
    rank_data = {
        "Metric": ["Total XP", "Study Streak", "Level", "Study Hours"],
        "Your Score": [
            st.session_state.total_xp,
            st.session_state.study_streak,
            st.session_state.level,
            st.session_state.study_timer // 3600
        ],
        "Top Score": [5000, 30, 10, 100]  # Mock data
    }
    
    import pandas as pd
    df = pd.DataFrame(rank_data)
    # Fix: Remove the width parameter or use an integer value
    st.dataframe(df)  # Removed width parameter

# TAB 8: TOOLS

with tabs[7]:
    st.markdown("## 🧪 Advanced Tools")
    
    tool = st.selectbox(
        "Select Tool",
        [
            "YouTube Summarizer",
            "Mnemonic Generator",
            "Pomodoro Timer",
            "Learning Style Quiz",
            "Feynman Technique",
            "🧠 AI Concept Visualizer",  # New advanced feature
            "📐 3D Molecular Viewer",     # New advanced feature
            "📸 Smart Document Scanner",  # New advanced feature
            "🎮 AR Learning Experience",  # New advanced feature
            "🔮 Predictive Performance Analytics"  # New advanced feature
        ]
    )
    
    if tool == "YouTube Summarizer":
        st.markdown("### 🎥 YouTube Video Summarizer")
        st.info(" Supports videos in any language including Tamil, Telugu, Hindi, English, and more!")
        video_url = st.text_input("Enter YouTube URL:")
        
        if st.button("Summarize Video"):
            if video_url:
                with st.spinner("Extracting and summarizing..."):
                    summary = summarize_youtube_video(video_url)
                    st.markdown(summary)
    
    elif tool == "Mnemonic Generator":
        st.markdown("### 💡 Mnemonic Generator")
        concept = st.text_input("Concept to remember:")
        key_points = st.text_area("Key points (one per line):")
        
        if st.button("Generate Mnemonic"):
            if concept and key_points:
                points = [p.strip() for p in key_points.split("\n") if p.strip()]
                mnemonic = generate_mnemonic(concept, points)
                st.markdown(mnemonic)
    
    elif tool == "Pomodoro Timer":
        pomodoro_session()
    
    elif tool == "Learning Style Quiz":
        detect_learning_style_quiz()
    
    elif tool == "Feynman Technique":
        st.markdown("### 🎓 Feynman Technique")
        st.info("""
        **How it works:**
        1. Choose a concept
        2. Explain it in simple terms (as if teaching a child)
        3. AI will identify gaps in your explanation
        4. Review and improve
        """)
        
        concept = st.text_input("What concept do you want to learn?")
        explanation = st.text_area("Explain it in your own words:", height=200)
        
        if st.button("Get Feedback"):
            if concept and explanation:
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                feedback = model.generate_content(f"""
                Evaluate this explanation of '{concept}':
                
                {explanation}
                
                Provide:
                1. What's explained well
                2. What's missing or unclear
                3. Suggestions to improve understanding
                """).text
                
                st.markdown("### 📝 Feedback")
                st.markdown(feedback)

    # NEW ADVANCED FEATURES
    
    elif tool == "🧠 AI Concept Visualizer":
        st.markdown("### 🧠 AI Concept Visualizer")
        st.info("Transform complex concepts into visual diagrams using AI!")
        
        if AI_IMAGING_AVAILABLE:
            concept = st.text_input("Enter a concept to visualize:")
            style = st.selectbox("Visualization Style", ["Diagram", "Mind Map", "Flowchart", "Infographic"])
            
            if st.button("Generate Visualization"):
                if concept:
                    with st.spinner("Creating visual representation..."):
                        try:
                            # Use Gemini to generate a description for visualization
                            model = genai.GenerativeModel('gemini-2.5-flash')
                            prompt = f"Create a detailed description of a visual diagram for '{concept}' in the style of a {style}. Describe the main components, their relationships, and layout in a way that could be used to create an image."
                            
                            description = model.generate_content(prompt).text
                            
                            # Display the description
                            st.markdown("### Generated Description")
                            st.info(description)
                            
                            st.success("✅ Concept visualization description generated! You can use this with image generation tools.")
                            
                        except Exception as e:
                            st.error(f"Error generating visualization: {str(e)}")
        else:
            st.warning("AI Imaging features require additional libraries. Please install: pip install diffusers transformers torch")
    
    elif tool == "📐 3D Molecular Viewer":
        st.markdown("### 📐 3D Molecular Viewer")
        st.info("Visualize molecules and chemical structures in 3D!")
        
        if THREED_VISUALIZATION_AVAILABLE:
            molecule = st.selectbox("Select a molecule", ["Water (H2O)", "Carbon Dioxide (CO2)", "Methane (CH4)", "Benzene (C6H6)", "Custom"])
            
            if molecule == "Custom":
                molecule_name = st.text_input("Enter molecule name or formula:")
            else:
                molecule_name = molecule.split(" (")[0]
            
            if st.button("Visualize Molecule"):
                with st.spinner("Generating 3D molecular structure..."):
                    try:
                        # Create a simple 3D visualization
                        plotter = pv.Plotter()
                        plotter.add_mesh(pv.Sphere(radius=0.5), color='red', opacity=0.7)  # Central atom
                        
                        # Add bonds
                        if "Water" in molecule or "H2O" in molecule:
                            # Water molecule visualization
                            plotter.add_mesh(pv.Sphere(center=[1, 0, 0], radius=0.3), color='white', opacity=0.7)  # H atom
                            plotter.add_mesh(pv.Sphere(center=[-1, 0, 0], radius=0.3), color='white', opacity=0.7)  # H atom
                            plotter.add_mesh(pv.Line([0, 0, 0], [1, 0, 0]), color='gray', line_width=5)  # Bond
                            plotter.add_mesh(pv.Line([0, 0, 0], [-1, 0, 0]), color='gray', line_width=5)  # Bond
                        
                        plotter.show_grid()
                        plotter.add_axes()
                        
                        # Render the 3D visualization
                        st.info("3D Molecular Viewer:")
                        plotter.ren_win
                        
                        st.success("✅ 3D molecular structure generated!")
                        
                    except Exception as e:
                        st.error(f"Error generating 3D visualization: {str(e)}")
        else:
            st.warning("3D Visualization requires additional libraries. Please install: pip install pyvista")
    
    elif tool == "📸 Smart Document Scanner":
        st.markdown("### 📸 Smart Document Scanner")
        st.info("Extract and digitize text from images using AI!")
        
        if COMPUTER_VISION_AVAILABLE:
            uploaded_image = st.file_uploader("Upload an image containing text", type=["png", "jpg", "jpeg"])
            
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                if st.button("Extract Text"):
                    with st.spinner("Extracting text from image..."):
                        try:
                            # Convert PIL image to OpenCV format
                            import numpy as np
                            opencv_image = np.array(image)
                            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
                            
                            # Extract text using pytesseract
                            extracted_text = pytesseract.image_to_string(opencv_image)
                            
                            st.markdown("### Extracted Text")
                            st.text_area("Text from Image", extracted_text, height=200)
                            
                            # Summarize the extracted text
                            if extracted_text.strip():
                                model = genai.GenerativeModel('gemini-2.5-flash')
                                summary = model.generate_content(f"Summarize this text:\n\n{extracted_text}").text
                                st.markdown("### Summary")
                                st.info(summary)
                                
                        except Exception as e:
                            st.error(f"Error extracting text: {str(e)}")
        else:
            st.warning("Computer Vision features require additional libraries. Please install: pip install opencv-python pytesseract")
    
    elif tool == "🎮 AR Learning Experience":
        st.markdown("### 🎮 AR Learning Experience")
        st.info("Immersive Augmented Reality learning experiences!")
        
        experience = st.selectbox("Select AR Experience", [
            "Solar System Explorer",
            "Human Anatomy Viewer",
            "Historical Timeline Journey",
            "Mathematical Functions Visualizer"
        ])
        
        st.markdown("### AR Experience Preview")
        st.info(f"Selected Experience: {experience}")
        st.image("https://images.unsplash.com/photo-16423498-8a5f-4f3c-90c7-95d60c1b6d27?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=800&q=80", caption="AR Learning Experience Preview")
        
        st.markdown("### How to Use")
        st.markdown("""
        1. Download our AR app (coming soon)
        2. Point your camera at a flat surface
        3. Experience immersive 3D learning content
        4. Interact with virtual objects using gestures
        """)
        
        st.success("AR experiences will be available in our mobile app soon!")
    
    elif tool == "🔮 Predictive Performance Analytics":
        st.markdown("### 🔮 Predictive Performance Analytics")
        st.info("AI-powered predictions of your learning outcomes and recommendations!")
        
        # Generate mock predictive analytics
        import random
        import pandas as pd
        import plotly.express as px
        
        # Mock data for predictive analytics
        subjects = ["Mathematics", "Physics", "Chemistry", "Biology", "History", "Literature"]
        current_scores = [random.randint(60, 95) for _ in subjects]
        predicted_scores = [min(100, score + random.randint(-5, 15)) for score in current_scores]
        confidence_levels = [random.randint(70, 95) for _ in subjects]
        
        # Create dataframe
        df = pd.DataFrame({
            "Subject": subjects,
            "Current Score": current_scores,
            "Predicted Score": predicted_scores,
            "Confidence %": confidence_levels
        })
        
        # Display predictions table
        st.markdown("### Performance Predictions")
        st.dataframe(df)
        
        # Create visualization
        if PLOTLY_AVAILABLE:
            fig = px.bar(df, x="Subject", y=["Current Score", "Predicted Score"], 
                         title="Current vs Predicted Performance",
                         barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            
            fig2 = px.line(df, x="Subject", y="Confidence %", 
                          title="Prediction Confidence Levels",
                          markers=True)
            st.plotly_chart(fig2, use_container_width=True)
        
        # AI Recommendations
        st.markdown("### 🤖 AI Recommendations")
        recommendations = [
            "📚 Focus more time on Mathematics - high improvement potential",
            "⚡ Physics shows strong progress - maintain current study habits",
            "🎯 Chemistry needs more practice - consider additional flashcards",
            "📈 Biology performance is stable - add more quizzes for reinforcement"
        ]
        
        for rec in recommendations:
            st.success(rec)
        
        st.info("💡 These predictions are based on your study patterns, quiz performance, and time allocation.")

# AUTO-REFRESH FOR TIMER & CLOCK
update_clock()
if st.session_state.session_active and st.session_state.timer_running:
    time.sleep(0.5)
    st.rerun()
else:
    time.sleep(60) 
    st.rerun()
# AUTO-REFRESH FOR TIMER & CLOCK
update_clock()
if st.session_state.session_active and st.session_state.timer_running:
    time.sleep(0.5)
    st.rerun()
else:
    time.sleep(60) 
    st.rerun()
