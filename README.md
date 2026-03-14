# Hybrid Movie Recommendation System

A web-based movie recommendation system that combines Content-Based Filtering and Collaborative Filtering to provide personalized movie recommendations.

## Features

- **150 Movies** from 6 languages:
  - Bollywood (Hindi): 30 movies
  - Tamil: 25 movies  
  - Telugu: 25 movies
  - Kannada: 25 movies
  - Malayalam: 25 movies
  - Hollywood: 20 movies

- **Hybrid Recommendation Algorithm**:
  - Content-Based Filtering (TF-IDF + Cosine Similarity)
  - Collaborative Filtering (Item-Based with Mean Normalization)
  - Weighted scoring: 30% Content + 70% Collaborative

- **Accuracy**: ~85% (within 1 star)

## Tech Stack

- **Python** - Core language
- **Pandas & NumPy** - Data processing
- **Scikit-learn** - ML algorithms (TF-IDF, Cosine Similarity)
- **Streamlit** - Web UI framework

## How to Run Locally

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`

## Deployment to Streamlit Cloud

Since Python scripts cannot run directly on GitHub Pages, use **Streamlit Cloud** for free hosting:

1. **Push code to GitHub**:
   - Create a GitHub repository
   - Upload `app.py` and `requirements.txt`
   - Push to GitHub

2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and branch
   - Set "Main file path" as `app.py`
   - Click "Deploy"

3. Your app will be live at a URL like: `https://your-app-name.streamlit.app`

## Alternative: Run as Python Script Only

If you just want to run the recommendation algorithm without the web interface:

```bash
python hybrid_movie_recommender.py
```

This will:
- Generate recommendations for sample users
- Display similar movies
- Show model evaluation metrics

## Project Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit web application |
| `hybrid_movie_recommender.py` | Standalone Python script |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## Screenshots

The web app includes:
- Home page with database overview
- Movie recommendations by user
- Similar movies finder
- Model information page

## License

MIT License

## Author

Built with ❤️ using Python and Streamlit
