import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import json
import os

# Custom CSS for better styling
st.set_page_config(layout="wide", page_title="Movie Recommendations")
st.markdown("""
<style>
    /* Main container */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* Headers */
    h1 {
        color: #2c3e50;
        text-align: center;
        font-size: 3em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 30px;
        background: linear-gradient(45deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Movie cards */
    .movie-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 15px 0;
        transition: transform 0.3s ease;
        border-left: 5px solid #3498db;
    }

    .movie-card:hover {
        transform: translateY(-5px);
    }

    .movie-title {
        color: #2c3e50;
        font-size: 1.5em;
        margin-bottom: 10px;
    }

    .movie-theme {
        color: #7f8c8d;
        font-size: 1.1em;
    }

    .similarity-score {
        color: #e74c3c;
        font-weight: bold;
    }

    /* Input fields */
    .stTextInput input {
        border-radius: 10px;
        border: 2px solid #3498db;
        padding: 10px;
        transition: all 0.3s ease;
    }

    .stTextInput input:focus {
        border-color: #2ecc71;
        box-shadow: 0 0 10px rgba(46,204,113,0.2);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #3498db, #2ecc71);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    /* Feedback section */
    .feedback-section {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 30px;
    }

    /* Slider */
    .stSlider {
        padding: 10px 0;
    }

    /* Error messages */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize feedback file
FEEDBACK_FILE = 'feedback.json'
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump([], f)

@st.cache_resource
def initialize_nltk():
    for resource in ['punkt', 'stopwords', 'wordnet']:
        nltk.download(resource)
    return WordNetLemmatizer(), set(stopwords.words('english'))

def save_feedback(feedback_data):
    try:
        with open(FEEDBACK_FILE, 'r') as f:
            existing_feedback = json.load(f)
        existing_feedback.append(feedback_data)
        with open(FEEDBACK_FILE, 'w') as f:
            json.dump(existing_feedback, f)
        return True
    except Exception as e:
        st.error(f"Error saving feedback: {str(e)}")
        return False

lemmatizer, stop_words = initialize_nltk()

def process_text(text):
    """
    Tokenizes, removes stopwords, lemmatizes, and expands text using WordNet synsets.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    # Tokenize and preprocess
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.lower() not in stop_words and word.isalnum()]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Expand using WordNet synsets
    expanded_tokens = set(lemmatized_tokens)  # To avoid duplicates
    for token in lemmatized_tokens:
        synsets = wordnet.synsets(token)
        for synset in synsets:
            expanded_tokens.update(synset.lemma_names())  # Add synonyms

    return list(expanded_tokens)

@st.cache_data
def load_and_preprocess_data():
    data = pd.read_excel('rottentomatoes800.xlsx')
    data.dropna(subset=['movie_info', 'theme'], inplace=True)
    data['theme'] = data['theme'].astype(str)
    data['genre'] = data['genre'].astype(str) if 'genre' in data.columns else ''
    data['emotions'] = data['emotions'].astype(str) if 'emotions' in data.columns else ''
    data['processed_theme'] = data['theme'].apply(lambda x: ' '.join(process_text(x)))
    data['processed_genre'] = data['genre'].apply(lambda x: ' '.join(process_text(x)))
    data['processed_emotions'] = data['emotions'].apply(lambda x: ' '.join(process_text(x)))
    data['combined_features'] = data[['processed_theme', 'processed_genre', 'processed_emotions']].apply(' '.join, axis=1)
    return data

@st.cache_resource
def create_tfidf_matrix(combined_features):
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(combined_features)
    return vectorizer, tfidf_matrix

def get_recommendations(input_theme, vectorizer, tfidf_matrix, data, top_n=10):
    input_vector = vectorizer.transform([input_theme])
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()
    positive_indices = [i for i, score in enumerate(similarity_scores) if score > 0]

    if not positive_indices:
        return None

    sorted_indices = sorted(positive_indices, key=lambda i: similarity_scores[i], reverse=True)[:top_n]
    recommendations = data.iloc[sorted_indices][['movie_title', 'theme', 'genre', 'emotions']]
    recommendations['similarity_score'] = similarity_scores[sorted_indices]
    return recommendations

def main():
    st.title("🎬 Movie Magic: Theme, Genre, and Emotion-Based Recommendations")

    try:
        with st.container():
            st.markdown("""
                <div style='text-align: center; color: #7f8c8d; margin-bottom: 30px;'>
                    Discover movies that match your interests! Enter a theme, genre, or emotion keywords below.
                </div>
            """, unsafe_allow_html=True)

            data = load_and_preprocess_data()
            vectorizer, tfidf_matrix = create_tfidf_matrix(data['combined_features'])

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                user_theme = st.text_input("",
                                           placeholder="Enter theme or keywords (e.g., 'space adventure', 'romantic comedy')")
                search_button = st.button("🔍 Find Movies")

            if search_button:
                if user_theme:
                    user_theme_expanded = ' '.join(process_text(user_theme))
                    recommendations = get_recommendations(user_theme_expanded, vectorizer, tfidf_matrix, data)

                    if recommendations is None:
                        st.error("🎬 No movies found matching your input. Please try different keywords!")
                    else:
                        st.session_state['current_recommendations'] = recommendations

                        st.markdown("<h3 style='text-align: center; color: #2c3e50;'>🌟 Recommended Movies</h3>",
                                    unsafe_allow_html=True)

                        for _, movie in recommendations.iterrows():
                            st.markdown(f"""
                                <div class='movie-card'>
                                    <div class='movie-title'>{movie['movie_title']}</div>
                                    <div class='movie-theme'>Theme: {movie['theme']}</div>
                                    <div class='movie-genre'>Genre: {movie['genre']}</div>
                                    <div class='movie-emotions'>Emotions: {movie['emotions']}</div>
                                    <div class='similarity-score'>Match Score: {movie['similarity_score']:.2f}</div>
                                </div>
                            """, unsafe_allow_html=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                "📥 Download Recommendations",
                                recommendations.to_csv(index=False),
                                "movie_recommendations.csv",
                                "text/csv"
                            )

                        st.markdown("<div class='feedback-section'>", unsafe_allow_html=True)
                        st.markdown("<h3 style='color: #2c3e50; text-align: center;'>📝 Your Feedback</h3>",
                                    unsafe_allow_html=True)

                        with st.form(key='feedback_form'):
                            col1, col2 = st.columns(2)
                            with col1:
                                rating = st.slider("⭐ Rate these recommendations", 1, 5, 3)
                            with col2:
                                relevance = st.select_slider(
                                    "📊 Relevance of recommendations",
                                    options=["Not relevant", "Somewhat relevant", "Very relevant"]
                                )

                            comments = st.text_area("💭 Additional comments")
                            submit_feedback = st.form_submit_button("Submit Feedback")

                            if submit_feedback:
                                feedback_data = {
                                    'timestamp': datetime.datetime.now().isoformat(),
                                    'search_theme': user_theme,
                                    'rating': rating,
                                    'relevance': relevance,
                                    'comments': comments,
                                    'recommended_movies': recommendations['movie_title'].tolist()
                                }

                                if save_feedback(feedback_data):
                                    st.success("🎉 Thank you for your feedback!")

                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error("Please enter a theme or keywords to get recommendations!")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.error("Please check if your data file exists and contains the required columns.")

    st.markdown("""
        <div style='text-align: center; color: #7f8c8d; padding: 20px; margin-top: 50px;'>
            Made with ❤️ for movie lovers | © 2024
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
