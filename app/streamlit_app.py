import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import os
from pathlib import Path

st.set_page_config(
    page_title="CineMatch — Film Öneri Sistemi",
    page_icon="🎬",
    layout="wide"
)

# Project root (one level above the app/ folder)
BASE = Path(__file__).resolve().parent.parent

# TMDB API key: read from .streamlit/secrets.toml or the TMDB_API_KEY environment variable
try:
    TMDB_KEY = st.secrets["TMDB_API_KEY"]
except Exception:
    TMDB_KEY = os.getenv("TMDB_API_KEY", "")

def get_poster(movie_title, year=None):
    if not TMDB_KEY:
        return None
    try:
        title = movie_title.split(" (")[0].strip()
        params = {"api_key": TMDB_KEY, "query": title, "language": "tr-TR"}
        if year:
            params["year"] = year
        r = requests.get("https://api.themoviedb.org/3/search/movie",
                         params=params, timeout=5)
        data = r.json()
        if data.get("results"):
            path = data["results"][0].get("poster_path")
            if path:
                return f"https://image.tmdb.org/t/p/w300{path}"
    except:
        pass
    return None

def extract_year(title):
    try:
        return int(title.strip()[-5:-1])
    except:
        return None

@st.cache_data
def load_data():
    base = BASE
    ratings = pd.read_csv(os.path.join(base, "data", "rating.csv"))
    movies  = pd.read_csv(os.path.join(base, "data", "movie.csv"))
    return ratings, movies

@st.cache_resource
def load_models():
    base = BASE
    models = {}
    svd_path = os.path.join(base, "svdpp_best_model_100.pkl")
    if os.path.exists(svd_path):
        with open(svd_path, "rb") as f:
            models["SVD++"] = {"type": "surprise", "model": pickle.load(f)}
    nmf_path = os.path.join(base, "nmf_best_model_100.pkl")
    if os.path.exists(nmf_path):
        with open(nmf_path, "rb") as f:
            models["NMF"] = {"type": "surprise", "model": pickle.load(f)}
    return models

@st.cache_data
def load_als_recommendations():
    base = BASE
    als_path = os.path.join(base, "als_recommendations")
    if os.path.exists(als_path):
        # Parquet dosyalarını oku
        dfs = []
        for f in os.listdir(als_path):
            if f.endswith(".parquet"):
                dfs.append(pd.read_parquet(os.path.join(als_path, f)))
        if dfs:
            return pd.concat(dfs, ignore_index=True)
    return None

ratings, movies = load_data()
models = load_models()
als_recs = load_als_recommendations()

if als_recs is not None:
    models["ALS"] = {"type": "precomputed", "data": als_recs}

METRICS = {
    "SVD++": {"Precision@10": 0.8659, "NDCG@10": 0.8902, "MAP": 0.8257},
    "NMF":   {"Precision@10": 0.8102, "NDCG@10": 0.8286, "MAP": 0.7384},
    "ALS":   {"Precision@10": 0.8608, "NDCG@10": 0.8865, "MAP": 0.8197},
}

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #0F172A !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label {
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="input"] *,
[data-testid="stSidebar"] input {
    color: #0F172A !important;
    background-color: #F8FAFC !important;
}
[data-testid="stSidebar"] button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    font-weight: bold !important;
    font-size: 15px !important;
}
[data-testid="stSidebar"] button:hover {
    background-color: #1D4ED8 !important;
}
.metric-card {
    background: #1E293B;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 5px 0;
    border-left: 4px solid #2563EB;
}
.metric-label { font-size: 11px; color: #94A3B8 !important; margin-bottom: 2px; }
.metric-value { font-size: 20px; font-weight: bold; color: #60A5FA !important; }
.film-title {
    font-size: 12px !important;
    font-weight: bold !important;
    color: #1E293B !important;
    margin: 6px 0 3px 0;
    line-height: 1.4;
    display: block;
}
.film-score {
    font-size: 14px !important;
    font-weight: bold !important;
    color: #2563EB !important;
    margin: 3px 0;
    display: block;
}
.film-rating {
    font-size: 13px !important;
    color: #D97706 !important;
    margin: 3px 0;
    display: block;
}
.film-genre {
    font-size: 10px !important;
    color: #475569 !important;
    line-height: 1.4;
    display: block;
    margin-top: 2px;
}
.section-title {
    font-size: 17px;
    font-weight: bold;
    color: #0F172A;
    margin: 18px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 3px solid #2563EB;
}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.markdown("---")

    st.markdown("### ⚙️ Algoritma")
    algo_name = st.selectbox(
        "Algoritma seç",
        options=list(models.keys()),
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 📊 Model Metrikleri")
    m = METRICS.get(algo_name, {})
    for label, value in m.items():
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 👤 Kullanıcı ID")
    user_id = st.number_input(
        "Kullanıcı ID gir",
        min_value=1,
        max_value=int(ratings['userId'].max()),
        value=133254,
        label_visibility="collapsed"
    )

    st.markdown("### 🎯 Öneri Sayısı")
    n_rec = st.slider(
        "Öneri sayısı",
        min_value=5, max_value=20, value=10,
        label_visibility="collapsed"
    )

    st.markdown("---")
    recommend_btn = st.button("🎯 Öneri Al", use_container_width=True)

# ── ANA ALAN ─────────────────────────────────────────────────
st.markdown(f"## 🎬 CineMatch — {algo_name} Önerileri")

if not recommend_btn:
    st.info("👈 Sol panelden kullanıcı ID ve algoritmayı seç, **Öneri Al** butonuna tıkla.")
    st.stop()

# Kullanıcı kontrolü
user_ratings = ratings[ratings['userId'] == user_id]
if user_ratings.empty:
    st.error(f"Kullanıcı {user_id} bulunamadı!")
    st.stop()

# Film geçmişi
user_history = (
    user_ratings
    .merge(movies, on='movieId')
    .sort_values('rating', ascending=False)
)
rated_movie_ids = set(user_ratings['movieId'].tolist())

# ── TAHMİN YAP ───────────────────────────────────────────────
algo_info = models[algo_name]

if algo_info["type"] == "precomputed":
    # ALS: Önceden hesaplanmış önerilerden al
    user_recs = algo_info["data"][
        algo_info["data"]["userId"] == user_id
    ].sort_values("predicted_rating", ascending=False)

    # İzlenmeyenleri filtrele
    user_recs = user_recs[~user_recs["movieId"].isin(rated_movie_ids)]
    top_preds = list(zip(
        user_recs["movieId"].head(n_rec).tolist(),
        user_recs["predicted_rating"].head(n_rec).tolist()
    ))

    if not top_preds:
        st.error(f"Kullanıcı {user_id} için ALS önerisi bulunamadı!")
        st.stop()

else:
    # SVD++ / NMF: Canlı tahmin
    algo = algo_info["model"]
    movie_counts     = ratings.groupby('movieId')['rating'].count()
    popular_movies   = movie_counts[movie_counts >= 50].index.tolist()
    candidate_movies = [mid for mid in popular_movies if mid not in rated_movie_ids]

    with st.spinner(f"🤖 {algo_name} ile tahminler hesaplanıyor..."):
        preds = []
        for mid in candidate_movies:
            try:
                est = algo.predict(uid=user_id, iid=mid).est
                preds.append((mid, round(est, 2)))
            except:
                pass
        preds.sort(key=lambda x: x[1], reverse=True)
        top_preds = preds[:n_rec]

# ── ÖNERİLER ─────────────────────────────────────────────────
st.markdown(
    f'<div class="section-title">🎯 {algo_name} — Top {n_rec} Öneri</div>',
    unsafe_allow_html=True
)

rec_scores    = {mid: score for mid, score in top_preds}
rec_movies_df = movies[movies['movieId'].isin(rec_scores.keys())].copy()
rec_movies_df['score'] = rec_movies_df['movieId'].map(rec_scores)
rec_movies_df = rec_movies_df.sort_values('score', ascending=False)

cols_per_row = 5
for i in range(0, len(rec_movies_df), cols_per_row):
    row_df = rec_movies_df.iloc[i:i+cols_per_row]
    cols   = st.columns(cols_per_row)
    for col, (_, film) in zip(cols, row_df.iterrows()):
        with col:
            year   = extract_year(film['title'])
            poster = get_poster(film['title'], year)
            score  = film['score']
            genres = film['genres'].replace("|", " · ") if pd.notna(film['genres']) else "—"
            title  = film['title'][:38] + "..." if len(film['title']) > 38 else film['title']

            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.markdown("🎬")

            st.markdown(f"""
            <span class="film-title">{title}</span>
            <span class="film-score">⭐ Tahmin: {score:.2f}</span>
            <span class="film-genre">{genres}</span>
            """, unsafe_allow_html=True)

# ── KULLANICI GEÇMİŞİ ────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<div class="section-title">📽 Kullanıcı {user_id} — İzleme Geçmişi (En Çok Beğendikleri)</div>',
    unsafe_allow_html=True
)

top_history = user_history.head(n_rec)

for i in range(0, len(top_history), cols_per_row):
    row_df = top_history.iloc[i:i+cols_per_row]
    cols   = st.columns(cols_per_row)
    for col, (_, film) in zip(cols, row_df.iterrows()):
        with col:
            year   = extract_year(film['title'])
            poster = get_poster(film['title'], year)
            rating = film['rating']
            genres = film['genres'].replace("|", " · ") if pd.notna(film['genres']) else "—"
            title  = film['title'][:38] + "..." if len(film['title']) > 38 else film['title']

            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.markdown("🎬")

            st.markdown(f"""
            <span class="film-title">{title}</span>
            <span class="film-rating">⭐ Verilen Puan: {rating}</span>
            <span class="film-genre">{genres}</span>
            """, unsafe_allow_html=True)

st.markdown("---")
st.caption(
    f"Kullanıcı {user_id} · Toplam {len(user_history)} film izlendi · "
    f"Ort. rating: {user_history['rating'].mean():.2f} · Model: {algo_name}"
)
