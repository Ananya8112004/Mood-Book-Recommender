# Advanced Mood-Based Book Recommender with RAG

A sophisticated Streamlit application that recommends books based on your emotional state using Retrieval-Augmented Generation (RAG) techniques powered by Google's Gemini API.

## 🌟 Features

### Core Functionality
- **Mood-Based Recommendations**: Enter your emotional state and get personalized book suggestions
- **Emotion Detection**: Analyzes sentiment, emotions (joy, sadness, anger, etc.), and subjectivity
- **RAG Implementation**: 
  - TF-IDF vectorization and cosine similarity for book retrieval
  - Gemini API-powered explanations for why books match your mood
  - Multi-turn chatbot for book-specific questions
- **Favorites/Bookmarks**: Save your favorite books for later
- **Export Features**: Download recommendations as CSV

### Technical Highlights
- **Sentiment Analysis**: TextBlob for polarity and subjectivity; VADER (Valence Aware Dictionary and sEntiment Reasoner) for detailed sentiment breakdown
- **Emotion Detection**: NRCLex for emotion classification
- **Retrieval**: Scikit-learn TF-IDF + cosine similarity (no Torch required)
- **AI Generation**: Google Gemini API integration
- **Caching**: Streamlit caching for optimal performance
- **Error Handling**: Comprehensive error handling and user feedback

### UI/UX Enhancements
- Responsive card-based layout for books
- Interactive emotion visualizations
- Expandable sections for detailed information
- Loading spinners for API calls
- Custom CSS styling for better aesthetics
- Sidebar navigation and settings

## 📋 Requirements

### System Requirements
- Python 3.9+
- pip package manager

### Dependencies
See `requirements.txt` for full list:
- streamlit
- pandas, numpy
- scikit-learn (TF-IDF, cosine similarity)
- textblob, nltk (sentiment analysis with TextBlob and VADER)
- matplotlib (visualizations)
- google-generativeai (Gemini API)

## 🚀 Setup & Installation

### Step 1: Get Gemini API Key

Gemini API is free to use
1. Visit https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key

### Step 2: Clone or Download
\`\`\`bash
git clone https://github.com/yourusername/mood-book-recommender.git
cd mood-book-recommender
\`\`\`

### Step 3: Create Virtual Environment (Recommended)
\`\`\`bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
\`\`\`

### Step 4: Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Step 5: Download TextBlob Corpora and VADER Lexicon
\`\`\`bash
python -m textblob.download_corpora
python -m nltk.downloader vader_lexicon
\`\`\`

### Step 6: Configure Gemini API Key

Create `.streamlit/secrets.toml` in your project root:
\`\`\`toml
GEMINI_API_KEY = "your-actual-gemini-api-key-here"
\`\`\`

Or use environment variables:
\`\`\`bash
export GEMINI_API_KEY="your-actual-gemini-api-key-here"
\`\`\`

### Step 7: Prepare Dataset
1. Ensure `data/books.csv` exists in the project root (sample provided)
2. CSV should have columns: `book_id, authors, original_publication_year, title, language_code, average_rating, image_url, description`
3. Clean data (remove rows with null descriptions) - app will do this automatically

### Step 8: Run the Application
\`\`\`bash
streamlit run app.py
\`\`\`

The app will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### Main Page
1. **Enter Your Mood**: Describe your emotional state (e.g., "nostalgic and adventurous")
2. **Click "Get Recommendations"**: The app analyzes your mood and retrieves matching books
3. **View Results**: 
   - See emotion analysis with visualizations
   - Read Gemini-generated explanation for recommendations
   - Browse recommended books in card format

### Book Details
1. **Click "Details"** on any book card
2. **View Full Information**:
   - Full description
   - Cover image, author, rating, publication year
3. **Ask Questions**: Use the chatbot to query Gemini about the book
4. **Add to Favorites**: Bookmark books for later

### Favorites Page
- View all your bookmarked books
- Clear all favorites with one click

### Export Recommendations
- Download your recommendations as CSV file
- Useful for sharing or archiving

## 🔧 Configuration

Edit `config.py` to customize:
- `TOP_N_RECOMMENDATIONS`: Number of books to recommend (default: 5)
- `MAX_TOKENS_EXPLANATION`: Gemini token limit for explanations (default: 300)
- `MAX_TOKENS_CHATBOT`: Gemini token limit for chatbot (default: 250)
- `CACHE_TTL`: Cache time-to-live in seconds (default: 3600)

## 🔐 Security Considerations

- **API Keys**: Always use `.streamlit/secrets.toml` for your Gemini key
- **Never commit** `.streamlit/secrets.toml` to version control
- Add `.streamlit/secrets.toml` to `.gitignore`:
\`\`\`
.streamlit/secrets.toml
data/
\`\`\`
- Use HTTPS for all API calls
- Streamlit's built-in caching improves privacy and performance

## 📊 File Structure

\`\`\`
mood-book-recommender/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration and constants
├── recommender_utils.py      # Recommendation logic & mood analysis
├── chatbot_utils.py          # Multi-turn chatbot utilities
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore file
├── .streamlit/
│   └── secrets.toml         # Gemini API key (DO NOT COMMIT)
└── data/
    └── books.csv            # Book dataset
\`\`\`

## 🎯 How the RAG Implementation Works

### 1. **Data Retrieval Phase**
- Load book dataset from CSV
- Vectorize book descriptions using TF-IDF
- Store vectors in memory for fast access

### 2. **User Query Processing**
- Accept user's mood description
- Perform sentiment analysis (TextBlob)
- Perform emotion detection (NRCLex)
- Vectorize mood input using same TF-IDF vectorizer

### 3. **Similarity Matching**
- Compute cosine similarity between mood vector and book vectors
- Rank books by similarity score
- Retrieve top N most relevant books

### 4. **Augmentation & Generation**
- Pass top books + mood analysis to Gemini API
- Gemini generates personalized explanation
- Cache results for performance

### 5. **Interactive Enhancement**
- Enable multi-turn chatbot per book via Gemini
- Maintain conversation history in session state
- Provide contextual answers using book description

## 🚀 Deployment

### Deploy to Streamlit Cloud (Recommended)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add `GEMINI_API_KEY` via Streamlit Cloud UI Secrets
5. Deploy!

### Deploy to Heroku
1. Create `Procfile`:
\`\`\`
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
\`\`\`
2. Push to Heroku and add `GEMINI_API_KEY` environment variable
3. Deploy!

### Deploy to Your Own Server
1. Set up Python environment
2. Run with gunicorn or similar ASGI server
3. Configure reverse proxy (nginx)
4. Set `GEMINI_API_KEY` environment variable

## 📈 Performance Optimizations

- **Caching**: TF-IDF matrix and dataset cached with `@st.cache_data`
- **Description Truncation**: Shortened descriptions reduce API tokens
- **Token Limits**: Optimized max_tokens to balance quality and cost
- **Lazy Loading**: Images loaded only when needed
- **Session State**: Minimize redundant API calls

## 🐛 Troubleshooting

### Gemini API Issues
- Verify API key is correct: https://makersuite.google.com/app/apikey
- Check you have quota remaining
- Ensure internet connection is active
- Restart Streamlit: `streamlit run app.py`

### Dataset Not Loading
- Ensure `data/books.csv` exists
- Check CSV has required columns
- Verify CSV encoding is UTF-8

### Sentiment Analysis Issues
- Run: `python -m textblob.download_corpora`
- Restart Streamlit app

### Slow Performance
- Increase `CACHE_TTL` in `config.py`
- Use fewer recommendations (`TOP_N_RECOMMENDATIONS`)
- Deploy on faster hardware

## 📚 Dataset Format

Expected CSV columns:
\`\`\`
book_id          | Integer, unique identifier
authors          | String, author name(s)
original_publication_year | Integer, publication year
title            | String, book title
language_code    | String, language code (e.g., 'eng')
average_rating   | Float, book rating (0-5)
image_url        | String, URL to book cover image
description      | String, book description (required)
\`\`\`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Streamlit for the amazing web app framework
- Scikit-learn for machine learning utilities
- TextBlob and NLTK for NLP capabilities (including VADER sentiment analysis)
- Google for Gemini API
- All book data providers and authors

## 📞 Support

For issues, questions, or suggestions:
1. Check existing GitHub issues
2. Create a new issue with detailed description
3. Include relevant error messages and logs
4. Specify Python version and OS
5. Verify Gemini API key is configured correctly

---

**Happy Reading! 📚✨**

Built with ❤️ using Streamlit, RAG, and Gemini API
