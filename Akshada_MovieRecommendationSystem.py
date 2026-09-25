import os
import ast
import difflib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: #05080c;
    color: #f5f5f5;
}
.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
}
.navbar {
    padding: 14px 0 22px 0;
    border-bottom: 1px solid #20262d;
    margin-bottom: 30px;
}
.brand {
    font-size: 25px;
    font-weight: 800;
}
.brand span {
    color: #ff2638;
}
.hero {
    padding: 38px 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #0c1118, #111821);
    border: 1px solid #202832;
    margin-bottom: 28px;
}
.hero h1 {
    font-size: 48px;
    margin: 0 0 8px 0;
}
.hero h1 span {
    color: #ff2638;
}
.hero p {
    color: #aeb7c2;
    font-size: 17px;
}
.movie-card {
    background: #0b1118;
    border: 1px solid #202832;
    border-radius: 16px;
    overflow: hidden;
    min-height: 510px;
    padding-bottom: 14px;
}
.movie-card:hover {
    border-color: #ff2638;
}
.movie-card img {
    width: 100%;
    height: 320px;
    object-fit: cover;
}
.movie-card-content {
    padding: 14px 16px;
}
.movie-title {
    font-size: 19px;
    font-weight: 700;
    margin-bottom: 7px;
}
.movie-meta {
    color: #b7c0ca;
    font-size: 14px;
}
.rating {
    color: #ffd166;
    font-weight: 700;
}
.detail-box {
    background: #0b1118;
    border: 1px solid #202832;
    border-radius: 18px;
    padding: 22px;
    margin: 18px 0 28px 0;
}
.section-title {
    font-size: 28px;
    font-weight: 800;
    margin: 25px 0 5px 0;
}
.muted {
    color: #9da8b4;
}
.footer {
    text-align: center;
    color: #77818d;
    padding: 35px 0 15px 0;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Paths and constants
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MOVIES_CSV = DATASET_DIR / "movies.csv.csv"
CREDITS_CSV = DATASET_DIR / "credits.csv.csv"
MOVIES_PKL = BASE_DIR / "movies.pkl"
SIMILARITY_PKL = BASE_DIR / "similarity.pkl"

OMDB_API_KEY = os.getenv("OMDB_API_KEY", "").strip()


# -----------------------------
# Helpers for dataset parsing
# -----------------------------
def parse_names(value):
    """Return names from a TMDB-style stringified list of dictionaries."""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        data = value
    else:
        try:
            data = ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            return []

    if not isinstance(data, list):
        return []

    result = []
    for item in data:
        if isinstance(item, dict):
            name = item.get("name")
            if name:
                result.append(str(name))
    return result


def parse_cast(value, limit=3):
    names = parse_names(value)
    return names[:limit]


def get_director(value):
    """Extract the director from a TMDB-style crew field."""
    if pd.isna(value):
        return ""
    try:
        data = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return ""

    if not isinstance(data, list):
        return ""

    for item in data:
        if isinstance(item, dict) and item.get("job") == "Director":
            return str(item.get("name", ""))
    return ""


# -----------------------------
# Build model if pickle files
# are not available
# -----------------------------
@st.cache_data(show_spinner=False)
def build_model_from_csv():
    if not MOVIES_CSV.exists() or not CREDITS_CSV.exists():
        return None, None

    movies = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)

    required_movie_cols = [
        "movie_id", "title", "overview", "genres", "keywords"
    ]
    required_credit_cols = [
        "movie_id", "title", "cast", "crew"
    ]

    if not all(c in movies.columns for c in required_movie_cols):
        return None, None
    if not all(c in credits.columns for c in required_credit_cols):
        return None, None

    movies = movies[required_movie_cols].copy()
    credits = credits[required_credit_cols].copy()

    merged = movies.merge(
        credits[["movie_id", "cast", "crew"]],
        on="movie_id",
        how="inner",
    )

    merged["overview"] = merged["overview"].fillna("")
    merged["genres"] = merged["genres"].fillna("[]")
    merged["keywords"] = merged["keywords"].fillna("[]")
    merged["cast"] = merged["cast"].fillna("[]")
    merged["crew"] = merged["crew"].fillna("[]")

    merged["genres_list"] = merged["genres"].apply(parse_names)
    merged["keywords_list"] = merged["keywords"].apply(parse_names)
    merged["cast_list"] = merged["cast"].apply(parse_cast)
    merged["director"] = merged["crew"].apply(get_director)

    merged["tags"] = (
        merged["genres_list"].apply(lambda x: " ".join(x)) + " " +
        merged["keywords_list"].apply(lambda x: " ".join(x)) + " " +
        merged["cast_list"].apply(lambda x: " ".join(x)) + " " +
        merged["director"].fillna("") + " " +
        merged["overview"].fillna("")
    ).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

    merged = merged.drop_duplicates(subset=["title"]).reset_index(drop=True)

    vectorizer = CountVectorizer(
        max_features=5000,
        stop_words="english",
    )
    vectors = vectorizer.fit_transform(merged["tags"])
    similarity = cosine_similarity(vectors)

    return merged, similarity


@st.cache_resource(show_spinner=False)
def load_project_model():
    # Prefer the already-generated project pickle files.
    if MOVIES_PKL.exists() and SIMILARITY_PKL.exists():
        try:
            with open(MOVIES_PKL, "rb") as f:
                movies = pickle.load(f)
            with open(SIMILARITY_PKL, "rb") as f:
                similarity = pickle.load(f)

            if isinstance(movies, pd.DataFrame):
                movies = movies.reset_index(drop=True)
                return movies, np.asarray(similarity)
        except Exception:
            pass

    # If pickle files are unavailable, rebuild from the CSV dataset.
    return build_model_from_csv()


movies, similarity = load_project_model()

if movies is None or similarity is None:
    st.error(
        "Project data was not found. Keep movies.pkl and similarity.pkl in "
        "the same folder as this Python file, or keep the CSV files inside "
        "the dataset folder."
    )
    st.stop()


if "title" not in movies.columns:
    st.error("The movie data must contain a 'title' column.")
    st.stop()

movies["title"] = movies["title"].astype(str).str.strip()
movie_list = movies["title"].tolist()


# -----------------------------
# Movie search
# -----------------------------
def find_movie(search_text):
    query = str(search_text).strip().lower()
    if not query:
        return None

    normalized = {title.lower(): title for title in movie_list}

    if query in normalized:
        return normalized[query]

    partial = [title for title in movie_list if query in title.lower()]
    if partial:
        return partial[0]

    matches = difflib.get_close_matches(
        query,
        [title.lower() for title in movie_list],
        n=1,
        cutoff=0.55,
    )
    if matches:
        return normalized.get(matches[0])

    return None


# -----------------------------
# OMDb API - optional
# -----------------------------
def fetch_omdb_details(movie_title):
    """Get poster/rating/year/genre/plot when an OMDb key is available."""
    if not OMDB_API_KEY:
        return {}

    try:
        response = requests.get(
            "https://www.omdbapi.com/",
            params={"apikey": OMDB_API_KEY, "t": movie_title},
            timeout=8,
        )
        data = response.json()
        if data.get("Response") == "True":
            return data
    except requests.RequestException:
        pass

    return {}


def local_details(movie_title):
    """Fallback details from the project dataset when OMDb is unavailable."""
    row = movies[movies["title"].str.lower() == movie_title.lower()]
    if row.empty:
        return {}

    row = row.iloc[0]
    genres = ""

    if "genres" in row.index:
        parsed = parse_names(row["genres"])
        genres = ", ".join(parsed)

    return {
        "Title": movie_title,
        "Year": "",
        "Genre": genres,
        "Plot": str(row.get("overview", "") or ""),
        "Poster": "",
        "imdbRating": "N/A",
    }


def fetch_movie_details(movie_title):
    details = fetch_omdb_details(movie_title)
    if details:
        return details
    return local_details(movie_title)


# -----------------------------
# Recommendation function
# -----------------------------
def recommend(movie_title, n=5):
    matches = movies.index[
        movies["title"].str.lower() == movie_title.lower()
    ].tolist()

    if not matches:
        return []

    index = matches[0]

    if index >= len(similarity):
        return []

    distances = list(enumerate(similarity[index]))
    distances = sorted(
        distances,
        key=lambda x: x[1],
        reverse=True,
    )

    recommendations = []
    for movie_index, score in distances[1:n + 1]:
        if movie_index < len(movies):
            recommendations.append(
                (movies.iloc[movie_index]["title"], float(score))
            )

    return recommendations


# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="navbar">
    <div class="brand">🎬 Movie<span>Recommender</span></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>Find Your Next <span>Favorite Movie</span></h1>
    <p>Discover movies similar to the ones you already love using
    content-based machine learning.</p>
</div>
""", unsafe_allow_html=True)

search_movie = st.text_input(
    "Search for a movie",
    placeholder="Example: Avatar, Inception, Titanic...",
    label_visibility="collapsed",
)

recommend_button = st.button(
    "🎬 Get Recommendations",
    type="primary",
    use_container_width=False,
)


# -----------------------------
# Search and results
# -----------------------------
if recommend_button:
    if not search_movie.strip():
        st.warning("Please enter a movie name.")
    else:
        selected_movie = find_movie(search_movie)

        if selected_movie is None:
            st.error("Movie not found. Try another movie title.")
        else:
            selected_details = fetch_movie_details(selected_movie)

            st.markdown(
                '<div class="section-title">Selected Movie</div>',
                unsafe_allow_html=True,
            )

            poster = selected_details.get("Poster", "")
            title = selected_details.get("Title", selected_movie)
            year = selected_details.get("Year", "")
            rating = selected_details.get("imdbRating", "N/A")
            genre = selected_details.get("Genre", "")
            plot = selected_details.get("Plot", "")

            left, right = st.columns([1, 2])

            with left:
                if poster and poster != "N/A":
                    st.image(poster, use_container_width=True)
                else:
                    st.info("Poster unavailable without an OMDb API key.")

            with right:
                st.markdown(
                    f"""
                    <div class="detail-box">
                        <h2>{title}</h2>
                        <p class="movie-meta">
                            <span class="rating">⭐ {rating}</span>
                            &nbsp;&nbsp; {year}
                        </p>
                        <p><b>Genre:</b> {genre or "Not available"}</p>
                        <p class="muted">{plot or "Movie description is not available in the local dataset."}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            recommendations = recommend(selected_movie, 5)

            st.markdown(
                '<div class="section-title">Recommended Movies</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="muted">Top 5 movies selected using content similarity.</p>',
                unsafe_allow_html=True,
            )

            if recommendations:
                cols = st.columns(5)

                for col, (rec_title, score) in zip(cols, recommendations):
                    details = fetch_movie_details(rec_title)

                    with col:
                        rec_poster = details.get("Poster", "")
                        rec_year = details.get("Year", "")
                        rec_rating = details.get("imdbRating", "N/A")
                        rec_genre = details.get("Genre", "")

                        if rec_poster and rec_poster != "N/A":
                            st.image(rec_poster, use_container_width=True)
                        else:
                            st.info("No poster")

                        st.markdown(
                            f"""
                            <div class="movie-card-content">
                                <div class="movie-title">{rec_title}</div>
                                <div class="movie-meta">
                                    {rec_year} &nbsp; | &nbsp;
                                    <span class="rating">⭐ {rec_rating}</span>
                                </div>
                                <div class="movie-meta">{rec_genre}</div>
                                <div class="movie-meta">
                                    Similarity: {score:.2f}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            else:
                st.error("Could not generate recommendations.")

st.markdown("""
<div class="footer">
    🎬 Movie Recommendation System<br>
    <span>Content-based movie recommendation using Python and machine learning.</span>
</div>
""", unsafe_allow_html=True)
