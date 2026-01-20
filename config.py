"""
Configuration file for the Advanced Mood-Based Book Recommender.
Stores paths, API settings, and constants.
"""

import os
from pathlib import Path

# ==================== Paths ====================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
CSV_PATH = (PROJECT_ROOT / "data" / "books.csv").resolve()
#CSV_PATH = DATA_DIR / "books.csv"

# ==================== API Configuration ====================
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_KEY = "AIzaSyCzoM5Z8JAP5RdGqxTIcOordZqUUCT_MqU"
GEMINI_MODEL = "gemini-2.5-flash"

# ==================== Recommendation Settings ====================
TOP_N_RECOMMENDATIONS = 5
MAX_TOKENS_EXPLANATION = 300
MAX_TOKENS_CHATBOT = 250

# ==================== Text Processing ====================
MAX_DESCRIPTION_LENGTH = 500  # Characters to send to AI API
MIN_DESCRIPTION_LENGTH = 50   # Minimum description length to use

# ==================== Caching ====================
CACHE_TTL = 3600  # Cache time-to-live in seconds (1 hour)

# ==================== UI Settings ====================
PAGE_TITLE = "📚 Advanced Mood-Based Book Recommender"
PAGE_ICON = "📚"
LAYOUT = "wide"
THEME = "light"

# ==================== Mood Analysis ====================
EMOTION_KEYWORDS = {
    "joy": ["happy", "cheerful", "joyful", "delighted", "excited"],
    "sadness": ["sad", "melancholic", "sorrowful", "blue", "down"],
    "anger": ["angry", "furious", "irritated", "enraged"],
    "fear": ["scared", "afraid", "anxious", "terrified"],
    "trust": ["confident", "trusting", "optimistic"],
    "anticipation": ["excited", "anticipating", "looking forward"],
    "surprise": ["surprised", "amazed", "shocked"],
    "disgust": ["disgusted", "repulsed"],
}

# ==================== Styling Constants ====================
CUSTOM_CSS = """
    <style>
    .book-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .book-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
        border-color: #6366f1;
    }
    .mood-input {
        background-color: #f0f4ff;
        border: 2px solid #6366f1;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
    }
    .success-box {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    .error-box {
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    </style>
"""
