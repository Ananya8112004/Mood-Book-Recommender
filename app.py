"""
Advanced Mood-Based Book Recommender with RAG
Main Streamlit application
"""
from recommender_utils import creative_chatbot

import streamlit as st
import pandas as pd
from datetime import datetime
from io import StringIO
import matplotlib.pyplot as plt
import urllib.parse
import requests
from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT, CUSTOM_CSS, TOP_N_RECOMMENDATIONS
)
from recommender_utils import (
    get_book_recommendations_with_scores, get_book_details, truncate_description
)
# from chatbot_utils import (
#     initialize_ai_client, generate_book_explanation, chat_with_book_expert
# )

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# ================== CUSTOM SIDEBAR CSS ==================
st.markdown(
    """
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #f0f4ff;  /* Your preferred color */
    }

    /* Sidebar text color */
    [data-testid="stSidebar"] .css-1d391kg {
        color: #1f2937;  /* Text color */
    }

    /* Sidebar header style */
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #4f46e5;  /* Header color */
    }

    /* Radio buttons and other inputs in sidebar */
    [data-testid="stSidebar"] .stRadio {
        background-color: #e0e7ff;
        border-radius: 8px;
        padding: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Background gradient with light pastel colors */
    .stApp {
        background: linear-gradient(135deg, #FFD1DC, #D1C4FF, #B3E5FC); /* Light pink, purple, blue */
        background-size: cover;
        background-attachment: fixed;
    }

    /* General text styling */
    .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        font-family: 'Times New Roman', 'Georgia', serif;
        color: #1a1a1a; /* Dark text for contrast */
        font-weight: bold;
    }

    /* Header-specific styling */
    h1, h2, h3, h4 {
        font-family: 'Georgia', 'Times New Roman', serif;
        font-weight: bolder;
        color: #2c2c2c; /* Slightly darker for headers */
    }

    /* Optional: add shadow for readability */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p {
        text-shadow: 1px 1px 2px rgba(255,255,255,0.6);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- Add background image CSS here ---
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://img.freepik.com/premium-photo/cute-watercolor-floral-background_664601-16622.jpg?w=2000");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# --- End background image CSS ---

# Custom CSS injection
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# # Initialize AI client
# @st.cache_resource
# def get_ai_client():
#     return initialize_ai_client()

# ai_client = get_ai_client()

# Session state initialization
if "page" not in st.session_state:
    st.session_state.page = "home"
if "selected_book" not in st.session_state:
    st.session_state.selected_book = None
if "mood_input" not in st.session_state:
    st.session_state.mood_input = ""
if "year_range" not in st.session_state:
    st.session_state.year_range = (1800, 2025)

def page_home():
    """Home page: mood input, timeline filter, and recommendation trigger"""
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; margin-bottom: 30px;'>
        <h1>📚 Advanced Mood-Based Book Recommender</h1>
        <p style='font-size: 18px; color: #666;'>
        Discover books that perfectly match your emotional state using AI-powered recommendations
        </p>
        </div>
        """, unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    st.markdown("### 📅 Select Publication Timeline")
    col1, col2 = st.columns(2)
    
    with col1:
        min_year = st.slider("From Year:", min_value=1800, max_value=2025, value=1800, key="min_year")
    
    with col2:
        max_year = st.slider("To Year:", min_value=1800, max_value=2025, value=2025, key="max_year")
    
    if min_year > max_year:
        st.error("Start year cannot be after end year!")
        return
    
    st.session_state.year_range = (min_year, max_year)
    st.info(f"📚 Showing books published between {min_year} and {max_year}")
    
    st.markdown("---")
    
    # Mood input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        mood_text = st.text_area(
            "What's your current mood or emotional state?",
            placeholder="e.g., 'Nostalgic and adventurous', 'Calm and reflective', 'Excited and optimistic'",
            height=100,
            key="mood_input_area"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        recommend_button = st.button("🔍 Get Recommendations", use_container_width=True, type="primary")
    
    st.markdown("---")
    
    # Example moods
    st.markdown("### 💡 Try These Moods:")
    example_moods = [
        "Melancholic yet hopeful",
        "Adventure-seeking and curious",
        "Cozy and nostalgic",
        "Intense and thrilling",
        "Peaceful and introspective"
    ]
    
    cols = st.columns(len(example_moods))
    for i, mood in enumerate(example_moods):
        with cols[i]:
            if st.button(mood, use_container_width=True, key=f"example_{i}"):
                st.session_state.mood_input = mood
                recommend_button = True
    
    # Process recommendation
    if recommend_button and mood_text:
        with st.spinner("✨ Analyzing your mood and finding perfect books..."):
            recommendations, mood_analysis = get_book_recommendations_with_scores(mood_text)
            
            if recommendations.empty:
                st.error("No books found. Please try a different mood description.")
                return
            
            recommendations = recommendations[
                (recommendations['original_publication_year'] >= min_year) &
                (recommendations['original_publication_year'] <= max_year)
            ]
            
            if recommendations.empty:
                st.warning(f"No books found in the {min_year}-{max_year} timeline. Try a different range!")
                return
            
            st.info(f"📝 Analyzing mood: **{mood_text}**")
            
            # Display mood analysis
            st.success("✅ Mood Analysis Complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                polarity = mood_analysis.get("polarity", 0)
                polarity_label = "Positive 😊" if polarity > 0.2 else "Neutral 😐" if polarity > -0.2 else "Negative 😔"
                st.metric("Emotional Tone", polarity_label, f"{polarity:.2f}")
            
            with col2:
                emotions = mood_analysis.get("emotions", [])
                top_emotion = "neutral"
                if emotions and isinstance(emotions, list) and len(emotions) > 0:
                    first_emotion = emotions[0]
                    if isinstance(first_emotion, (tuple, list)) and len(first_emotion) >= 1:
                        top_emotion = first_emotion[0]  # Get first element of tuple
                
                st.metric("Primary Emotion", str(top_emotion).capitalize() if top_emotion else "Neutral")
            
            with col3:
                subjectivity = mood_analysis.get("subjectivity", 0)
                st.metric("Subjectivity", f"{subjectivity:.2%}")
            
            if mood_analysis.get("vader_scores"):
                st.markdown("### 🎭 Sentiment Breakdown (VADER Analysis)")
                
                vader_scores = mood_analysis.get("vader_scores", {})
                if vader_scores and all(k in vader_scores for k in ["negative", "neutral", "positive"]):
                    sentiment_data = {
                        "Sentiment": ["Negative", "Neutral", "Positive"],
                        "Score": [
                            vader_scores.get("negative", 0),
                            vader_scores.get("neutral", 0),
                            vader_scores.get("positive", 0)
                        ]
                    }
                    sentiment_df = pd.DataFrame(sentiment_data)
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    colors = ["#ef4444", "#9ca3af", "#22c55e"]
                    ax.barh(sentiment_df["Sentiment"], sentiment_df["Score"], color=colors)
                    ax.set_xlabel("Score")
                    ax.set_title("Sentiment Analysis of Your Mood (VADER)")
                    ax.set_xlim(0, 1)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.warning("Unable to display sentiment breakdown")

            # # Display book explanation
            # st.markdown("---")
            # st.markdown("### 💭 Why These Books?")
            
            # with st.spinner("Generating personalized explanation..."):
            #     book_titles = recommendations['title'].tolist()[:TOP_N_RECOMMENDATIONS]
            #     mood_emotions = mood_analysis.get("emotions", [])
            #     if not mood_emotions or not isinstance(mood_emotions, list) or len(mood_emotions) == 0:
            #         mood_emotions = [("neutral", 1.0)]
                
            #     if book_titles and len(book_titles) > 0:
            #         explanation = generate_book_explanation(
            #             book_titles,
            #             mood_text,
            #             mood_emotions,
            #             ai_client
            #         )
            #         st.info(explanation)
            #     else:
            #         st.warning("No books available for this mood in the selected timeline.")
            
            st.markdown("---")
            st.markdown("### 📖 Your Personalized Recommendations")
            
            books_per_row = 3
            for idx, (_, book) in enumerate(recommendations.iterrows()):
                if idx % books_per_row == 0:
                    cols = st.columns(books_per_row)
                
                with cols[idx % books_per_row]:
                    book_details = get_book_details(book)
                    
                    st.markdown('<div class="book-card">', unsafe_allow_html=True)
                    
                    if pd.notna(book.get('image_url')):
                        try:
                            book_title = book_details['title']
                            author_name = book_details['author']
                            
                            # First try: search by ISBN or exact title+author
                            search_url = f"https://openlibrary.org/search.json?title={urllib.parse.quote(book_title)}&author={urllib.parse.quote(author_name)}&limit=1"
                            response = requests.get(search_url, timeout=5)
                            
                            cover_found = False
                            if response.status_code == 200:
                                search_results = response.json()
                                if search_results.get('docs') and len(search_results['docs']) > 0:
                                    doc = search_results['docs'][0]
                                    if doc.get('cover_i'):
                                        cover_url = f"https://covers.openlibrary.org/b/id/{doc['cover_i']}-M.jpg"
                                        st.image(cover_url, width=150)
                                        cover_found = True
                            
                            # Fallback: try generic cover placeholder if not found
                            if not cover_found:
                                # Use a generic book cover placeholder
                                placeholder_url = "https://covers.openlibrary.org/b/id/7725679-M.jpg"
                                try:
                                    st.image(placeholder_url, width=150)
                                except:
                                    st.info("📷 Cover unavailable")
                        except Exception as e:
                            st.info("📷 Cover not available")
                    else:
                        st.info("📷 Cover not available")
                    
                    # Title and Author
                    st.markdown(f"**{book_details['title']}**")
                    st.caption(f"by {book_details['author']}")
                    
                    # Rating and Year
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Rating", f"⭐ {book_details['rating']}")
                    with col_b:
                        st.metric("Year", book_details['year'])
                    
                    # Truncated description
                    truncated = truncate_description(book_details['description'], 150)
                    st.markdown(f"*{truncated}*")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Export recommendations
            st.markdown("---")
            st.markdown("### 📥 Export Recommendations")
            
            export_df = recommendations[[
                'title', 'authors', 'average_rating', 'original_publication_year',
                'language_code'
            ]].copy()
            export_df.columns = ['Title', 'Author', 'Rating', 'Year', 'Language']
            export_df['Export Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    elif recommend_button:
        st.warning("Please enter your mood before requesting recommendations.")

def page_book_details():
    """Book details and chatbot page"""
    
    if st.session_state.selected_book is None:
        st.error("No book selected. Please go back to home.")
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.rerun()
        return
    
    book = st.session_state.selected_book
    
    # Back button
    if st.button("← Back to Recommendations"):
        st.session_state.page = "home"
        st.rerun()
    
    st.markdown("---")
    
    st.markdown(f"## {book['title']}")
    st.markdown(f"**Author:** {book['author']}")
    st.markdown(f"**Published:** {book['year']}")
    st.markdown(f"**Language:** {book['language'].upper()}")
    st.metric("Rating", f"⭐ {book['rating']}")
    
    st.markdown("---")
    
    # Full description
    with st.expander("📄 Full Description", expanded=True):
        st.write(book['description'])
    
    st.markdown("---")
    
    # # Chatbot section
    # st.markdown("## 💬 Ask the Book Expert")
    # st.markdown("Get detailed answers about this book from our AI expert.")
    
    # # Chat input only - no history display
    # user_question = st.text_input(
    #     "Ask a question about this book:",
    #     placeholder="e.g., What are the main themes? Is it suitable for beginners?",
    #     key="chatbot_input"
    # )
    
    # col1, col2 = st.columns([4, 1])
    
    # with col1:
    #     pass  # For alignment
    
    # with col2:
    #     if st.button("Send", use_container_width=True, type="primary"):
    #         if user_question:
    #             with st.spinner("Expert is thinking..."):
    #                 response = chat_with_book_expert(
    #                     book['title'],
    #                     book['description'],
    #                     user_question,
    #                     ai_client
    #                 )
                
    #             st.success(f"**Expert:** {response}")
    #         else:
    #             st.warning("Please enter a question.")

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🗂️ Navigation")
    
    page_selection = st.radio(
        "Select Page:",
        ["Home", "About", "Moody Chat"],
        key="page_nav"
    )
    
    # st.markdown("---")
    # st.markdown("### ⚙️ Settings")
    
    # if st.checkbox("Show API Status"):
    #     if ai_client:
    #         st.success("✅ AI Service: Connected")
    #     else:
    #         st.error("❌ AI Service: Disconnected")
    
    # st.markdown("---")
    # st.markdown("### 📝 About")
    # st.markdown(
    #     """
    #     **Advanced Mood-Based Book Recommender**
        
    #     This app uses:
    #     - **TF-IDF + Cosine Similarity** for retrieval
    #     - **Sentiment Analysis** (TextBlob + VADER)
    #     - **AI Generation** (Gemini 1.5 Flash)
        
    #     **Version:** 1.0.0
    #     """
    # )

from recommender_utils import creative_chatbot

def page_moody_chat():
    st.title("🎭 Moody Chat — AI Mood Storyteller")
    st.write("Tell me your mood, and I will create stories, poems, or fun what-if worlds!")

    mood = st.text_input("How are you feeling?")

    user_prompt = st.text_area("What should I create for you? (story, poem, what-if...)")

    mode = st.radio("Choose a creative mode:", ["story", "poem", "what_if"])

    if st.button("✨ Generate"):
        if not user_prompt.strip():
            st.warning("Please enter something for the chatbot.")
            return
        
        with st.spinner("Creating your mood-based content..."):
            output = creative_chatbot(user_prompt, mood=mood, mode=mode)

        st.markdown("### ✨ Your Creative Result")
        st.success(output)


# Main page routing
if page_selection == "Home":
    page_home()
elif page_selection == "About":
    st.markdown("# About This Application")
    st.markdown(
        """
        ## Advanced Mood-Based Book Recommender 
        
        ### 🎯 Purpose
        This application helps you discover books that match your emotional state using 
        advanced AI techniques.
        
        ### 🔧 How It Works
        
        1. **Mood Analysis**: Your input is analyzed for:
           - Sentiment polarity (positive/negative/neutral)
           - Sentiment breakdown (VADER: negative, neutral, positive scores)
           - Subjectivity level
        
        2. **Book Retrieval**: We use TF-IDF vectorization and cosine similarity to find books
           whose descriptions best match your mood.
        
        3. **Timeline Filter**: Select the publication year range to find books from your preferred era.
        
        ### 📊 Technologies Used
        - **Frontend**: Streamlit
        - **NLP**: TextBlob, VADER, Scikit-learn
        - **Retrieval**: TF-IDF, Cosine Similarity
        - **Data Processing**: Pandas, NumPy, Matplotlib
        
        ### 💾 Dataset
        Contains classic and modern books with:
        - Title, Author, Publication Year
        - Average Rating
        - Description
        - Language Code
        
        ### 🔒 Privacy
        - No personal data is stored permanently
        - All data is processed server-side
        
        ### 📧 Contact & Support
        For issues or suggestions, please visit the project repository.
        """
    )
elif page_selection == "Moody Chat":
    page_moody_chat()