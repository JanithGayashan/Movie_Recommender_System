import pandas as pd
import numpy as np
import pickle

# Load models into memory once
def load_artifacts():
    with open('models/svd_model.pkl', 'rb') as f:
        svd_model = pickle.load(f)
    with open('models/cosine_sim.pkl', 'rb') as f:
        cosine_sim, movie_id_to_idx, idx_to_movie_id = pickle.load(f)
    movies_df = pd.read_pickle('models/movies_processed.pkl')
    train_df = pd.read_pickle('models/train_ratings.pkl')
    return svd_model, cosine_sim, movie_id_to_idx, idx_to_movie_id, movies_df, train_df

# Global variables so the API doesn't crash on every reload
SVD_MODEL, COSINE_SIM, MOVIE_ID_TO_IDX, IDX_TO_MOVIE_ID, MOVIES_DF, TRAIN_DF = load_artifacts()
POP_SCORES_DICT = dict(zip(MOVIES_DF['movie_id'], MOVIES_DF['popularity_score_norm']))

def hybrid_recommend(user_id, alpha=0.5, beta=0.3, gamma=0.2, n=10):
    """Generates Top-N recommendations using the Hybrid weights from tuning experiment."""
    
    # 1. Identify seen vs unseen movies
    user_ratings = TRAIN_DF[TRAIN_DF['user_id'] == int(user_id)]
    rated_ids = set(user_ratings['movie_id'])
    unseen_movies = set(MOVIES_DF['movie_id']) - rated_ids
    
    if not unseen_movies:
        return {"error": "User has rated all movies."}

    # 2. Collaborative (SVD) Norm Scores
    svd_raw = {mid: SVD_MODEL.predict(str(user_id), str(mid)).est for mid in unseen_movies}
    if svd_raw:
        svd_vals = np.array(list(svd_raw.values()))
        svd_min, svd_max = svd_vals.min(), svd_vals.max()
        svd_norm = {mid: (v - svd_min) / (svd_max - svd_min) if svd_max > svd_min else 0.5 for mid, v in svd_raw.items()}
    else:
        svd_norm = {}

    # 3. Content-Based (CBF) Norm Scores (Handles Cold-Start)
    liked = user_ratings[user_ratings['rating'] >= 4]
    if liked.empty:
        liked = user_ratings # fallback to any rating if they have no 4+ ratings
    seed_movies = liked.sort_values('rating', ascending=False).head(5)

    cbf_raw = {}
    for _, row in seed_movies.iterrows():
        seed_mid = row['movie_id']
        if seed_mid in MOVIE_ID_TO_IDX:
            idx = MOVIE_ID_TO_IDX[seed_mid]
            for j, sim in enumerate(COSINE_SIM[idx]):
                cand_mid = IDX_TO_MOVIE_ID.values[j]
                if cand_mid not in rated_ids:
                    cbf_raw[cand_mid] = cbf_raw.get(cand_mid, 0) + sim * (row['rating'] / 5.0)

    if cbf_raw:
        cbf_vals = np.array(list(cbf_raw.values()))
        cbf_min, cbf_max = cbf_vals.min(), cbf_vals.max()
        cbf_norm = {mid: (v - cbf_min) / (cbf_max - cbf_min) if cbf_max > cbf_min else 0.5 for mid, v in cbf_raw.items()}
    else:
        cbf_norm = {}

    # 4. Compute Final Hybrid Score
    hybrid_scores = []
    for mid in unseen_movies:
        svd_s = svd_norm.get(mid, 0.0)
        cbf_s = cbf_norm.get(mid, 0.0)
        pop_s = POP_SCORES_DICT.get(mid, 0.0)
        
        hybrid_s = (alpha * svd_s) + (beta * cbf_s) + (gamma * pop_s)
        hybrid_scores.append({'movie_id': mid, 'hybrid_score': round(hybrid_s, 4)})

    # 5. Sort & Return JSON-ready Output
    result_df = pd.DataFrame(hybrid_scores)
    result_df = result_df.sort_values('hybrid_score', ascending=False).head(n)
    result_df = result_df.merge(MOVIES_DF[['movie_id', 'clean_title', 'genres', 'year']], on='movie_id')
    
    return result_df[['clean_title', 'genres', 'year', 'hybrid_score']].to_dict(orient='records')