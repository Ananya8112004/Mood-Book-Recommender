"""
Recommender utilities: mood analysis, TF-IDF vectorization, and book recommendation.
Uses RAG approach with vector similarity search.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import streamlit as st
from config import (
    CSV_PATH, MAX_DESCRIPTION_LENGTH, MIN_DESCRIPTION_LENGTH,
    TOP_N_RECOMMENDATIONS, CACHE_TTL
)

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

MOOD_KEYWORDS = {
    "anger": "revenge dark gritty brutal war violent aggressive harsh conflict rage fury intense battle",
    "joy": "celebration happy cheerful magical wonderful delightful inspiring uplifting triumph festive victory",
    "sadness": "melancholy tragic heartbreak loss grief sorrow despair lonely devastating emotional",
    "calm": "peaceful serene tranquil quiet meditation wisdom gentle philosophical introspection",
    "fear": "horror terror suspense thriller danger haunting eerie spooky supernatural mysterious ominous",
    "love": "romance passionate intimate relationship devotion affection tender connection family loyalty",
    "adventure": "quest expedition exploration journey discovery daring brave exotic thrilling action",
    "nostalgia": "memory childhood past reminiscent heritage tradition vintage retro historical"
}

MOOD_TO_BOOKS = {
    "anger": [6, 15, 20, 5, 10],  # 1984, The Shining, The Handmaid's Tale, Frankenstein, Da Vinci Code
    "joy": [1, 11, 18, 13, 9],     # Harry Potter, American Gods, The Alchemist, Foundation, Great Gatsby
    "sadness": [4, 8, 3, 5, 20],  # Jane Eyre, To Kill a Mockingbird, Pride and Prejudice, Frankenstein, Handmaid's Tale
    "calm": [18, 12, 14, 7, 2],   # The Alchemist, Left Hand of Darkness, 2001, Fahrenheit 451, Fellowship
    "fear": [15, 10, 6, 5, 16],   # The Shining, Da Vinci Code, 1984, Frankenstein, Murder on the Orient Express
    "love": [3, 4, 9, 11, 18],    # Pride and Prejudice, Jane Eyre, Great Gatsby, American Gods, Alchemist
    "adventure": [2, 19, 11, 14, 7],  # Fellowship, Life of Pi, American Gods, 2001, Fahrenheit 451
    "nostalgia": [3, 4, 8, 18, 1],  # Pride and Prejudice, Jane Eyre, To Kill a Mockingbird, Alchemist, Harry Potter
}

@st.cache_data(ttl=CACHE_TTL)
def load_books_data():
    """
    Load and preprocess book dataset.
    Returns: pd.DataFrame with cleaned book data
    """
    try:
        df = pd.read_csv(CSV_PATH, quotechar='"', quoting=1, encoding='utf-8', low_memory=False)
        
        # Data cleaning
        df = df.dropna(subset=['description'])
        df = df[df['description'].str.len() > MIN_DESCRIPTION_LENGTH]
        df['description'] = df['description'].fillna("")
        df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')
        df['original_publication_year'] = pd.to_numeric(
            df['original_publication_year'], errors='coerce'
        )
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    except FileNotFoundError:
        st.error(f"Dataset not found at {CSV_PATH}. Please add books.csv to the data folder.")
        return pd.DataFrame()
    except pd.errors.ParserError as e:
        st.error(f"Error parsing CSV file: {str(e)}. Please check the CSV format.")
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL)
def build_tfidf_vectorizer(df):
    """
    Build TF-IDF vectorizer for book descriptions.
    Returns: (vectorizer object, TF-IDF matrix)
    """
    if df.empty:
        return None, None
    
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9
    )
    
    # Combine title and description for better context
    df_copy = df.copy()
    df_copy['combined_text'] = df_copy['title'] + " " + df_copy['description']
    
    tfidf_matrix = vectorizer.fit_transform(df_copy['combined_text'])
    
    return vectorizer, tfidf_matrix

def enrich_mood_input(mood_input):
    """
    Enhance mood input with related keywords to improve matching.
    
    Args:
        mood_input (str): User's mood description
        
    Returns:
        str: Enhanced mood text with related keywords
    """
    mood_lower = mood_input.lower()
    enriched = mood_input
    
    # Check for mood keywords and add related themes
    for mood, keywords in MOOD_KEYWORDS.items():
        if mood in mood_lower:
            enriched += " " + keywords
    
    return enriched

def analyze_mood(mood_input):
    """
    Analyze user's mood using TextBlob and VADER sentiment analysis.
    
    Args:
        mood_input (str): User's mood description
        
    Returns:
        dict: Contains sentiment polarity, subjectivity, and VADER scores
    """
    # TextBlob for polarity and subjectivity
    blob = TextBlob(mood_input)
    polarity = blob.sentiment.polarity  # -1 to 1
    subjectivity = blob.sentiment.subjectivity  # 0 to 1
    
    # VADER for detailed sentiment breakdown
    vader = SentimentIntensityAnalyzer()
    vader_scores = vader.polarity_scores(mood_input)
    
    # Map VADER scores to emotional categories for display
    mood_category = "neutral"
    if vader_scores['compound'] > 0.5:
        mood_category = "joy"
    elif vader_scores['compound'] > 0.1:
        mood_category = "anticipation"
    elif vader_scores['compound'] < -0.5:
        mood_category = "sadness"
    elif vader_scores['compound'] < -0.1:
        mood_category = "anger"
    
    mood_analysis = {
        "input": mood_input,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "emotions": [(mood_category, abs(vader_scores['compound']))],  # Top emotion based on VADER
        "vader_scores": {
            "negative": vader_scores['neg'],
            "neutral": vader_scores['neu'],
            "positive": vader_scores['pos'],
            "compound": vader_scores['compound']
        }
    }
    
    return mood_analysis

def get_recommendations(mood_input, df, vectorizer, tfidf_matrix, top_n=TOP_N_RECOMMENDATIONS):
    """
    Get book recommendations based on user's mood using TF-IDF and cosine similarity.
    Prioritizes mood-specific book selections for better differentiation.
    """
    if df.empty or vectorizer is None:
        return pd.DataFrame()
    
    enriched_mood = enrich_mood_input(mood_input)
    
    try:
        mood_vector = vectorizer.transform([enriched_mood])
    except Exception as e:
        st.error(f"Error vectorizing mood: {str(e)}")
        return pd.DataFrame()
    
    similarities = cosine_similarity(mood_vector, tfidf_matrix).flatten()
    
    detected_mood = None
    mood_lower = mood_input.lower()
    for mood_key in MOOD_KEYWORDS.keys():
        if mood_key in mood_lower:
            detected_mood = mood_key
            break
    
    if detected_mood and detected_mood in MOOD_TO_BOOKS:
        mood_book_indices = MOOD_TO_BOOKS[detected_mood]
        # Boost scores for mood-specific books
        for book_idx in mood_book_indices:
            if 0 <= book_idx < len(similarities):
                similarities[book_idx] += 0.3  # Boost score by 0.3
    
    # Get top candidates
    top_indices_all = np.argsort(similarities)[::-1][:top_n * 2]
    
    selected_indices = []
    for idx in top_indices_all:
        if len(selected_indices) < top_n:
            selected_indices.append(idx)
    
    import random
    if len(selected_indices) > 3:
        # Keep top 1-2, randomize rest
        top_books = selected_indices[:2]
        other_books = selected_indices[2:]
        random.shuffle(other_books)
        selected_indices = top_books + other_books[:top_n-2]
    
    # Calculate normalized similarity scores
    similarities_selected = similarities[selected_indices]
    min_sim = similarities_selected.min()
    max_sim = similarities_selected.max()
    
    if max_sim > min_sim:
        normalized_similarities = 100 * (similarities_selected - min_sim) / (max_sim - min_sim)
    else:
        normalized_similarities = 100 * np.ones(len(selected_indices))
    
    # Create recommendations dataframe
    recommendations = df.iloc[selected_indices].copy()
    recommendations['similarity_score'] = normalized_similarities
    recommendations = recommendations.sort_values('similarity_score', ascending=False).reset_index(drop=True)
    
    return recommendations

def get_book_recommendations_with_scores(mood_input):
    """
    Complete recommendation pipeline: load data, build vectors, get recommendations.
    
    Args:
        mood_input (str): User's mood description
        
    Returns:
        tuple: (recommendations DataFrame, mood_analysis dict)
    """
    # Load data
    df = load_books_data()
    if df.empty:
        return pd.DataFrame(), None
    
    # Build TF-IDF
    vectorizer, tfidf_matrix = build_tfidf_vectorizer(df)
    if vectorizer is None:
        return pd.DataFrame(), None
    
    # Get recommendations
    recommendations = get_recommendations(mood_input, df, vectorizer, tfidf_matrix)
    
    # Analyze mood
    mood_analysis = analyze_mood(mood_input)
    
    return recommendations, mood_analysis

def truncate_description(description, max_length=MAX_DESCRIPTION_LENGTH):
    """
    Truncate description to max length for API calls.
    
    Args:
        description (str): Full description
        max_length (int): Maximum length
        
    Returns:
        str: Truncated description with ellipsis if needed
    """
    if len(description) > max_length:
        return description[:max_length] + "..."
    return description

def get_book_details(book_df_row):
    """
    Extract and format book details for display.
    
    Args:
        book_df_row: Single row from book DataFrame
        
    Returns:
        dict: Formatted book details
    """
    return {
        "title": book_df_row.get('title', 'Unknown'),
        "author": book_df_row.get('authors', 'Unknown'),
        "rating": round(book_df_row.get('average_rating', 0), 2),
        "year": int(book_df_row.get('original_publication_year', 0)),
        "description": book_df_row.get('description', 'No description available'),
        "image_url": book_df_row.get('image_url', ''),
        "language": book_df_row.get('language_code', 'Unknown'),
        "book_id": book_df_row.get('book_id', 'Unknown'),
    }
















import google.generativeai as genai

genai.configure(api_key="AIzaSyCzoM5Z8JAP5RdGqxTIcOordZqUUCT_MqU")

def creative_chatbot(prompt, mood="neutral", mode="story"):
    """
    Generates creative content safely using Gemini 2.5
    """
    try:
        # Safety: ensure no None or empty strings
        if not prompt:
            prompt = "Create something creative."
        if not mood:
            mood = "neutral"

        task_instruction = {
            "story": "Write a short, emotional story based on their mood.",
            "poem": "Write a deep, expressive poem.",
            "what_if": "Create a fun 'what if' alternate reality scenario.",
        }.get(mode, "Write something creative.")

        final_prompt = f"""
        User mood: {mood}
        Task: {task_instruction}
        User request: {prompt}

        Respond in 6–12 lines. Make it creative, emotional, and unique.
        """

        model = genai.GenerativeModel("gemini-2.5-flash")

        response = model.generate_content(final_prompt)

        # Safety: prevent list-index-out-of-range
        if hasattr(response, "text") and response.text:
            return response.text
        
        return "I tried to write something, but I couldn’t generate a response."

    except Exception as e:
        return f"Error: {str(e)}"










# import google.generativeai as genai
# from config import GEMINI_API_KEY, GEMINI_MODEL, MAX_TOKENS_CHATBOT

# # Configure Gemini
# genai.configure(api_key=GEMINI_API_KEY)

# def creative_chatbot(user_input, mood=None, mode="story"):
#     """
#     Creative AI chatbot using Google Gemini 2.5 models.
#     Supports: story, poem, what_if modes.
#     """

#     # Build prompt based on mode
#     if mode == "story":
#         prompt = f"Write a vivid, engaging short story inspired by this mood.\nMood: {mood}\nPrompt: {user_input}\n\nStory:"
#     elif mode == "poem":
#         prompt = f"Write a short, beautiful poem.\nMood: {mood}\nPrompt: {user_input}\n\nPoem:"
#     elif mode == "what_if":
#         prompt = f"Create a fun and imaginative 'what if' scenario.\nMood: {mood}\nPrompt: {user_input}\n\nScenario:"
#     else:
#         prompt = f"Create something creative based on this:\nMood: {mood}\nPrompt: {user_input}"

#     try:
#         model = genai.GenerativeModel(GEMINI_MODEL)

#         response = model.generate_content(
#             prompt,
#             generation_config={
#                 "temperature": 0.9,
#                 "max_output_tokens": MAX_TOKENS_CHATBOT,
#                 "top_p": 0.9
#             }
#         )

#         return response.text.strip()

#     except Exception as e:
#         return f"Error: {e}"
