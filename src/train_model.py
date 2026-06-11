import os
import pandas as pd
import pickle
from surprise import Reader, Dataset, SVD
from data_processor import process_movies, compute_tfidf_sim, compute_popularity

def train_and_save():
    print("Loading data...")
    # FIX: Removed the ../ from the paths below
    ratings_df = pd.read_csv('data/ml-1m/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python', encoding='latin-1')
    movies_df = pd.read_csv('data/ml-1m/movies.dat', sep='::', names=['movie_id', 'title', 'genres'], engine='python', encoding='latin-1')
    
    # Clean & Filter constraints
    ratings_df = ratings_df.drop_duplicates(subset=['user_id', 'movie_id'], keep='last')
    ratings_df = ratings_df[ratings_df['rating'].between(1, 5)]
    
    active_users = ratings_df['user_id'].value_counts()[ratings_df['user_id'].value_counts() >= 20].index
    ratings_df = ratings_df[ratings_df['user_id'].isin(active_users)]
    
    popular_movies = ratings_df['movie_id'].value_counts()[ratings_df['movie_id'].value_counts() >= 10].index
    ratings_df = ratings_df[ratings_df['movie_id'].isin(popular_movies)]
    
    movies_df = movies_df[movies_df['movie_id'].isin(ratings_df['movie_id'])].reset_index(drop=True)

    print("Processing content features & popularity...")
    movies_df = process_movies(movies_df)
    cosine_sim, movie_id_to_idx, idx_to_movie_id = compute_tfidf_sim(movies_df)
    movies_df = compute_popularity(ratings_df, movies_df)
    
    # FIX: Removed the ../ from the model paths below
    os.makedirs('models', exist_ok=True)

    movies_df.to_pickle('models/movies_processed.pkl')
    ratings_df[['user_id', 'movie_id', 'rating']].to_pickle('models/train_ratings.pkl')
    with open('models/cosine_sim.pkl', 'wb') as f:
        pickle.dump((cosine_sim, movie_id_to_idx, idx_to_movie_id), f)
        
    print("Training SVD model (this may take a minute)...")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    svd_model = SVD(n_factors=100, n_epochs=30, lr_all=0.01, reg_all=0.1, random_state=42)
    svd_model.fit(trainset)
    
    print("Saving SVD model...")
    with open('models/svd_model.pkl', 'wb') as f:
        pickle.dump(svd_model, f)
        
    print("✅ All models trained and saved successfully!")

if __name__ == "__main__":
    train_and_save()