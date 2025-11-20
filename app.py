import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)

download_nltk_data()

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
domain_stop_words = {'course', 'class', 'unit', 'would', 'could', 'one', 'like', 'also', 
                     'please', 'think', 'feel', 'make', 'try', 'way', 'thing', 'get', 'go'}
stop_words.update(domain_stop_words)
sia = SentimentIntensityAnalyzer()

# Load models
@st.cache_resource
def load_models():
    lda_model = joblib.load('model/topic_model_lda.pkl')
    count_vectorizer = joblib.load('model/topic_vectorizer.pkl')
    sentiment_classifier = joblib.load('model/sentiment_classifier.pkl')
    tfidf_vectorizer = joblib.load('model/sentiment_vectorizer_tfidf.pkl')
    
    with open('model/topic_labels.json', 'r') as f:
        topic_labels = json.load(f)
    
    return lda_model, count_vectorizer, sentiment_classifier, tfidf_vectorizer, topic_labels

try:
    lda_model, count_vectorizer, sentiment_classifier, tfidf_vectorizer, topic_labels = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    error_msg = str(e)

# Text preprocessing function
def preprocess_text(text):
    """Clean and preprocess text"""
    if not text:
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens 
              if token not in stop_words and len(token) > 2]
    return ' '.join(tokens)

# Predict topic
def predict_topic(text):
    """Predict dominant topic for input text"""
    processed = preprocess_text(text)
    
    if not processed:
        return None, 0, "No valid text to analyze"
    
    try:
        # Transform text using the vectorizer
        doc_term = count_vectorizer.transform([processed])
        # Get topic distribution
        topic_dist = lda_model.transform(doc_term)[0]
        dominant_topic = np.argmax(topic_dist)
        confidence = topic_dist[dominant_topic]
        
        return dominant_topic, confidence, None
    except Exception as e:
        return None, 0, f"Error predicting topic: {str(e)}"

# Predict sentiment
def predict_sentiment(text):
    """Predict sentiment for input text"""
    try:
        # VADER sentiment
        vader_scores = sia.polarity_scores(text)
        compound = vader_scores['compound']
        
        # Classify
        if compound >= 0.05:
            sentiment = 'POSITIVE'
            emoji = '😊'
        elif compound <= -0.05:
            sentiment = 'NEGATIVE'
            emoji = '😞'
        else:
            sentiment = 'NEUTRAL'
            emoji = '😐'
        
        return sentiment, compound, emoji, vader_scores
    except Exception as e:
        return None, 0, '❓', None

# Streamlit UI
st.set_page_config(
    page_title="BBT 4206: Course Evaluation NLP",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #1f77b4;
        text-align: center;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .topic-box {
        background-color: #e8f4f8;
        border-left-color: #0066cc;
    }
    .sentiment-box {
        background-color: #f0f8e8;
        border-left-color: #00cc00;
    }
    .error-box {
        background-color: #ffe8e8;
        border-left-color: #cc0000;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 📊 Business Intelligence Course Evaluation Analysis")
st.markdown("**Topic Modeling & Sentiment Analysis System**")
st.markdown("---")

# Check if models are loaded
if not models_loaded:
    st.error(f"❌ Error loading models: {error_msg}")
    st.info("Make sure you have run the complete_nlp_analysis.ipynb notebook first and all models are saved in the `/model` folder.")
    st.stop()

# Sidebar
st.sidebar.markdown("## About")
st.sidebar.info("""
This application analyzes student course evaluations using Natural Language Processing (NLP) to:

1. **Identify Topics** - Discover the main themes discussed in student feedback
2. **Analyze Sentiment** - Determine the sentiment (positive/negative/neutral) of each evaluation

The system was trained on 130+ student evaluations for BBT 4106 and BBT 4206.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("## Model Information")
st.sidebar.metric("Topics Discovered", "5")
st.sidebar.metric("Training Evaluations", "130+")
st.sidebar.metric("Vocabulary Size", "200+")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter Student Evaluation")
    user_input = st.text_area(
        "Paste or type a student's course evaluation feedback:",
        placeholder="Example: 'The labs were engaging but the slides are too long. More practical examples would help with understanding.'",
        height=150,
        label_visibility="collapsed"
    )

with col2:
    st.subheader("Instructions")
    st.markdown("""
    1. Enter or paste a student evaluation
    2. Click "Analyze" button
    3. View topic and sentiment results
    
    **Note**: The more text you provide, the better the analysis.
    """)

st.markdown("---")

# Analyze button
if st.button("🔍 Analyze Evaluation", key="analyze_btn", use_container_width=True):
    if not user_input.strip():
        st.error("❌ Please enter some text to analyze")
    else:
        with st.spinner("Analyzing evaluation..."):
            # Get topic
            topic_id, topic_conf, topic_error = predict_topic(user_input)
            
            # Get sentiment
            sentiment, sentiment_score, emoji, vader_details = predict_sentiment(user_input)
            
            # Display results
            st.markdown("### 📊 Analysis Results")
            
            col1, col2 = st.columns(2)
            
            # Topic result
            with col1:
                if topic_error:
                    st.markdown(f'<div class="result-box error-box"><h4>❌ Topic Error</h4><p>{topic_error}</p></div>', 
                              unsafe_allow_html=True)
                else:
                    topic_name = f"Topic {topic_id}"
                    topic_words = topic_labels.get(topic_name, [])
                    
                    st.markdown(f"""
                    <div class="result-box topic-box">
                    <h3>🎯 Dominant Topic</h3>
                    <h4 style="color: #0066cc; margin: 0.5rem 0;">{topic_name}</h4>
                    <p><b>Confidence:</b> {topic_conf:.1%}</p>
                    <p><b>Key Terms:</b> {', '.join(topic_words)}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Sentiment result
            with col2:
                if sentiment:
                    sentiment_colors = {
                        'POSITIVE': '#00cc00',
                        'NEGATIVE': '#cc0000',
                        'NEUTRAL': '#999999'
                    }
                    color = sentiment_colors.get(sentiment, '#000000')
                    
                    st.markdown(f"""
                    <div class="result-box sentiment-box">
                    <h3>💭 Sentiment Analysis</h3>
                    <h4 style="color: {color}; margin: 0.5rem 0;">{emoji} {sentiment}</h4>
                    <p><b>Score:</b> {sentiment_score:.3f}</p>
                    <p style="font-size: 0.9rem; color: #666;">
                    Score range: -1.0 (very negative) to +1.0 (very positive)
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Detailed sentiment breakdown
            st.markdown("### 📈 Sentiment Components (VADER)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Positive", f"{vader_details['pos']:.1%}")
            with col2:
                st.metric("Neutral", f"{vader_details['neu']:.1%}")
            with col3:
                st.metric("Negative", f"{vader_details['neg']:.1%}")
            
            # Text statistics
            st.markdown("### 📋 Text Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Character Count", len(user_input))
            with col2:
                st.metric("Word Count", len(user_input.split()))
            with col3:
                processed = preprocess_text(user_input)
                st.metric("Tokens (after processing)", len(processed.split()) if processed else 0)

# Information section
st.markdown("---")
st.markdown("## 📚 About This Analysis")

st.markdown("""
### What are Topics?
Topics are discovered using **Latent Dirichlet Allocation (LDA)**, an unsupervised machine learning technique that identifies the main themes discussed in the evaluations. Each evaluation is assigned to the topic it discusses most prominently.

### What is Sentiment Analysis?
Sentiment analysis uses the **VADER (Valence Aware Dictionary and sEntiment Reasoner)** tool to determine whether text expresses positive, negative, or neutral sentiment. This helps identify whether students' feedback is encouraging or critical.

### How accurate is this?
- **Topics**: Confidence score shows how dominant a topic is for the given text (higher is better)
- **Sentiment**: Works best with longer text and clear emotional language
- The system was trained on 130+ real course evaluations

### Limitations
- Very short text may produce unreliable results
- The system works best with English text
- Topics are discovered from the training data and may not apply to completely novel topics
""")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #666; font-size: 0.9rem;'>BBT 4206 - Natural Language Processing Lab | Strathmore University</p>", 
           unsafe_allow_html=True)