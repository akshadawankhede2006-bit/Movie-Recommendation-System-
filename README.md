# Movie Recommendation System

## Project Description

The Movie Recommendation System is a content-based recommendation project developed using Python, Machine Learning and Streamlit.

The system allows a user to search for a movie and receive the top 5 movies with similar content. Movie information such as poster, IMDb rating, release year, genre and plot can be displayed using the OMDb API.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- CountVectorizer
- Cosine Similarity
- Streamlit
- Requests
- OMDb API
- Jupyter Notebook

## Main Features

- Movie title search
- Content-based movie recommendation
- Top 5 similar movie recommendations
- CountVectorizer for text feature extraction
- Cosine similarity for finding similar movies
- Streamlit web interface
- Movie poster display
- IMDb rating display
- Release year and genre display
- Movie plot display
- Fallback movie information when an API key is not available

## Dataset

The project uses the TMDB 5000 Movies Dataset.

Dataset files:

- `dataset/movies.csv.csv`
- `dataset/credits.csv.csv`

The dataset is processed to create movie tags using genres, keywords, cast and director information.

## Machine Learning Method

This project uses content-based filtering.

1. Movie information is combined into a `tags` feature.
2. CountVectorizer converts the text into numerical vectors.
3. Cosine similarity calculates similarity between movies.
4. The searched movie is located in the dataset.
5. The five movies with the highest similarity scores are recommended.

## Project Structure

```text
Movie Recommendation System Project
│
├── Akshada_MovieRecommendationSystem.ipynb
├── README.md
├── movies.pkl
├── similarity.pkl
│
└── dataset
    ├── movies.csv.csv
    └── credits.csv.csv
```

## How to Run

Install the required libraries:

```bash
pip install streamlit pandas numpy scikit-learn requests
```

The complete project code is provided in:

`Akshada_MovieRecommendationSystem.ipynb`

The notebook contains both the recommendation-system backend and the Streamlit frontend.

For the Streamlit website, export the notebook code as:

```text
Akshada_MovieRecommendationSystem.py
```

Then run:

```bash
streamlit run Akshada_MovieRecommendationSystem.py
```

## OMDb API Key

The API key is not stored directly in the project code.

On Windows PowerShell, set:

```powershell
$env:OMDB_API_KEY="YOUR_OMDB_API_KEY"
```

Then run the application.

If the API key is not provided, the application can still run using the available local movie information.

## Output

The application provides:

- Movie search interface
- Selected movie details
- Movie poster
- IMDb rating
- Release year
- Genre
- Plot
- Top 5 movie recommendations

## Project Type

**Machine Learning / Data Analytics / Recommendation System**

## Author

**Akshada**

B.Sc. Computer Science
