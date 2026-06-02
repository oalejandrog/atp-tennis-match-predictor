import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

MATCH_FEATURES = [
    'surface', 'tourney_level', 'best_of',
    'winner_hand', 'winner_ht', 'winner_age', 'winner_rank', 'winner_rank_points',
    'loser_hand', 'loser_ht', 'loser_age', 'loser_rank', 'loser_rank_points'
]

def load_atp_matches(data_path='data/raw/tennis_atp/atp_matches_20*.csv'):
    files = glob.glob(data_path)
    modern_files = [f for f in files if any(str(year) in f for year in range(2018, 2025))]
    matches = pd.concat([pd.read_csv(f) for f in modern_files], ignore_index=True)
    return matches[MATCH_FEATURES].copy()

def clean_match_data(df):
    df = df.dropna(subset=['winner_rank', 'loser_rank']).copy()
    
    median_height = df['winner_ht'].median()
    df.loc[:, 'winner_ht'] = df['winner_ht'].fillna(median_height)
    df.loc[:, 'loser_ht'] = df['loser_ht'].fillna(median_height)
    
    df.loc[:, 'winner_hand'] = df['winner_hand'].fillna('U')
    df.loc[:, 'loser_hand'] = df['loser_hand'].fillna('U')
    return df

def create_balanced_dataset(df):
    np.random.seed(42)
    balanced_samples = []

    for _, row in df.iterrows():
        player_a_is_winner = np.random.rand() > 0.5
        
        if player_a_is_winner:
            sample = {
                'surface': row['surface'], 'tourney_level': row['tourney_level'], 'best_of': row['best_of'],
                'p1_hand': row['winner_hand'], 'p1_ht': row['winner_ht'], 'p1_age': row['winner_age'],
                'p1_rank': row['winner_rank'], 'p1_rank_pts': row['winner_rank_points'],
                'p2_hand': row['loser_hand'], 'p2_ht': row['loser_ht'], 'p2_age': row['loser_age'],
                'p2_rank': row['loser_rank'], 'p2_rank_pts': row['loser_rank_points'],
                'label': 1
            }
        else:
            sample = {
                'surface': row['surface'], 'tourney_level': row['tourney_level'], 'best_of': row['best_of'],
                'p1_hand': row['loser_hand'], 'p1_ht': row['loser_ht'], 'p1_age': row['loser_age'],
                'p1_rank': row['loser_rank'], 'p1_rank_pts': row['loser_rank_points'],
                'p2_hand': row['winner_hand'], 'p2_ht': row['winner_ht'], 'p2_age': row['winner_age'],
                'p2_rank': row['winner_rank'], 'p2_rank_pts': row['winner_rank_points'],
                'label': 0
            }
        balanced_samples.append(sample)
    
    return pd.DataFrame(balanced_samples)

def encode_features(df):
    categorical_columns = ['surface', 'tourney_level', 'p1_hand', 'p2_hand']
    for col in categorical_columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

def preprocess_pipeline():
    raw_matches = load_atp_matches()
    cleaned_matches = clean_match_data(raw_matches)
    balanced_df = create_balanced_dataset(cleaned_matches)
    final_df = encode_features(balanced_df)
    
    X = final_df.drop('label', axis=1)
    y = final_df['label']
    
    X_scaled = StandardScaler().fit_transform(X)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    os.makedirs('data/processed', exist_ok=True)
    np.savez('data/processed/tennis_data.npz', 
             X_train=X_train, y_train=y_train, 
             X_val=X_val, y_val=y_val, 
             X_test=X_test, y_test=y_test)
    
    print(f"Dataset processed. Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

if __name__ == "__main__":
    preprocess_pipeline()
