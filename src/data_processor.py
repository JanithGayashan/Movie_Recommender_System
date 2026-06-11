import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler

def build_content_profile(row):
    """Builds the text profile repeating genres 3x, adding decade, and clean title."""
    parts = []
    if isinstance(row['genre_list'], list):
        genres_str = ' '.join(row['genre_list'])
        parts.append((genres_str + ' ') * 3) 
        
    if pd.notna(row.get('year')) and row['year'] > 0:
        decade = f"decade_{int(row['year'] // 10 * 10)}s"
        parts.append(decade)
        
    if pd.notna(row.get('clean_title')):
        title_words = re.sub(r'[^a-zA-Z0-9\s]', '', row['clean_title'].lower())
        parts.append(title_words)
        
    return ' '.join(parts).strip()

def process_movies(movies_df):
    """Cleans movie titles and creates the TF-IDF content profiles."""
    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)
    movies_df['clean_title'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()
    movies_df['genre_list'] = movies_df['genres'].str.split('|')
    movies_df['content_profile'] = movies_df.apply(build_content_profile, axis=1)
    return movies_df

def compute_tfidf_sim(movies_df):
    """Computes the Cosine Similarity matrix using TF-IDF."""
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['content_profile'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    movie_id_to_idx = pd.Series(movies_df.index, index=movies_df['movie_id'])
    idx_to_movie_id = pd.Series(movies_df['movie_id'].values, index=movies_df.index)
    
    return cosine_sim, movie_id_to_idx, idx_to_movie_id

def compute_popularity(train_df, movies_df):
    """Computes Bayesian weighted average ratings for the popularity baseline."""
    movie_stats = train_df.groupby('movie_id').agg(avg_rating=('rating', 'mean'), vote_count=('rating', 'count')).reset_index()
    
    C = movie_stats['avg_rating'].mean()
    m = movie_stats['vote_count'].quantile(0.60)
    
    def bayesian_rating(row, C, m):
        v = row['vote_count']
        R = row['avg_rating']
        return (v / (v + m)) * R + (m / (v + m)) * C
        
    movie_stats['popularity_score'] = movie_stats.apply(bayesian_rating, axis=1, args=(C, m))
    
    scaler = MinMaxScaler()
    movie_stats['popularity_score_norm'] = scaler.fit_transform(movie_stats[['popularity_score']])
    
    movies_df = movies_df.merge(
        movie_stats[['movie_id','avg_rating','vote_count','popularity_score','popularity_score_norm']],
        on='movie_id', how='left'
    ).fillna({'popularity_score': C, 'popularity_score_norm': 0.5, 'avg_rating': C, 'vote_count': 0})
    
    return movies_df