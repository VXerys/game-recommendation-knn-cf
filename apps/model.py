"""
model.py - Machine Learning Logic for Game Recommendation System
This module contains all ML-related functions for the KNN-based game recommendation system.
K is fixed at 30 as per requirements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

FIXED_K = 30

class GameRecommendationModel:
    """
    Game Recommendation Model using Item-Based Collaborative Filtering
    with fixed K=30 neighbors
    """
    
    def __init__(self):
        self.similarity_df = None
        self.game_means = None
        self.user_item_matrix = None
        self.games_df = None
        self.filtered_ratings = None
        self.is_trained = False
        
    def load_and_process_data(self):
        """Load and preprocess data with robust error handling"""
        try:
            # Load data files
            ratings_df = pd.read_csv('data/game_ratings.csv')
            games_df = pd.read_csv('data/games_metadata_5k.csv')
            
            # Validate required columns
            required_rating_cols = ['user_id', 'game_id', 'rating']
            required_game_cols = ['game_id', 'name']
            
            if not all(col in ratings_df.columns for col in required_rating_cols):
                raise ValueError(f"Missing required columns in ratings: {required_rating_cols}")
                
            if not all(col in games_df.columns for col in required_game_cols):
                raise ValueError(f"Missing required columns in games: {required_game_cols}")
            
            # Filter data sparse - minimal 5 ratings
            MIN_RATINGS = 5
            user_counts = ratings_df.groupby('user_id').size()
            game_counts = ratings_df.groupby('game_id').size()
            
            valid_users = user_counts[user_counts >= MIN_RATINGS].index
            valid_games = game_counts[game_counts >= MIN_RATINGS].index
            
            filtered_ratings = ratings_df[
                (ratings_df['user_id'].isin(valid_users)) &
                (ratings_df['game_id'].isin(valid_games))
            ].copy()
            
            if len(filtered_ratings) == 0:
                raise ValueError("No data remaining after filtering!")
            
            # Filter games_df to only include valid games
            games_df = games_df[games_df['game_id'].isin(valid_games)].copy()
            
            # Store processed data
            self.filtered_ratings = filtered_ratings
            self.games_df = games_df
            
            return {
                'n_ratings': len(filtered_ratings),
                'n_users': filtered_ratings['user_id'].nunique(),
                'n_games': filtered_ratings['game_id'].nunique(),
                'sparsity': 1 - (len(filtered_ratings) / (filtered_ratings['user_id'].nunique() * filtered_ratings['game_id'].nunique()))
            }
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def build_similarity_matrix(self):
        """Build similarity matrix using cosine similarity with mean-centering"""
        try:
            if self.filtered_ratings is None:
                raise ValueError("Data not loaded. Call load_and_process_data() first.")
            
            # Split data for training (80%)
            np.random.seed(42)
            train_indices = np.random.choice(
                self.filtered_ratings.index, 
                size=int(len(self.filtered_ratings) * 0.8), 
                replace=False
            )
            train_set = self.filtered_ratings.loc[train_indices].copy()
            
            # Calculate mean rating per game
            self.game_means = train_set.groupby('game_id')['rating'].mean().to_dict()
            
            # Create user-item matrix
            self.user_item_matrix = pd.pivot_table(
                train_set, 
                values='rating', 
                index='user_id', 
                columns='game_id', 
                fill_value=0
            )
            
            # Mean-centering for similarity calculation
            centered_ratings = train_set.copy()
            centered_ratings['rating_centered'] = centered_ratings.apply(
                lambda row: row['rating'] - self.game_means.get(row['game_id'], 3.0), 
                axis=1
            )
            
            # Create item-user matrix (games as rows)
            item_user_matrix = pd.pivot_table(
                centered_ratings, 
                values='rating_centered', 
                index='game_id', 
                columns='user_id', 
                fill_value=0
            )
            
            # Calculate cosine similarity between games
            if len(item_user_matrix) > 1:
                similarity_matrix = cosine_similarity(item_user_matrix.values)
                self.similarity_df = pd.DataFrame(
                    similarity_matrix, 
                    index=item_user_matrix.index, 
                    columns=item_user_matrix.index
                )
                # Remove self-similarity
                np.fill_diagonal(self.similarity_df.values, 0)
            else:
                self.similarity_df = pd.DataFrame()
            
            self.is_trained = True
            return True
            
        except Exception as e:
            raise Exception(f"Error building similarity matrix: {str(e)}")
    
    def predict_rating(self, user_id, game_id):
        """Predict rating using Item-Based Collaborative Filtering with K=30"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call build_similarity_matrix() first.")
        
        # If user/game not found, return average game rating
        if user_id not in self.user_item_matrix.index or game_id not in self.similarity_df.index:
            return self.game_means.get(game_id, 3.0)
        
        # Get games rated by user
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_games = user_ratings[user_ratings > 0]
        
        if len(rated_games) == 0:
            return self.game_means.get(game_id, 3.0)
        
        # Get K=30 nearest neighbors based on similarity
        similarities = self.similarity_df.loc[game_id, rated_games.index].nlargest(FIXED_K)
        
        if len(similarities) == 0 or similarities.sum() == 0:
            return self.game_means.get(game_id, 3.0)
        
        # Calculate weighted average: base rating + weighted sum of deviations
        base_rating = self.game_means.get(game_id, 3.0)
        weighted_sum = sum(sim * (rating - self.game_means.get(gid, 3.0)) 
                          for sim, (gid, rating) in zip(similarities, rated_games[similarities.index].items()))
        
        prediction = base_rating + weighted_sum / similarities.abs().sum()
        return max(1.0, min(5.0, prediction))  # Constrain rating to 1-5
    
    def get_game_recommendations(self, selected_game_id, n_recommendations=5, min_rating=3.5, min_similarity=0.1):
        """
        Generate game recommendations based on similarity to selected game
        Uses K=30 for similarity calculations
        """
        try:
            if not self.is_trained:
                raise ValueError("Model not trained. Call build_similarity_matrix() first.")
            
            if selected_game_id not in self.similarity_df.index:
                return []
            
            # Get similarity scores for selected game
            game_similarities = self.similarity_df.loc[selected_game_id].copy()
            
            # Remove the selected game itself
            game_similarities = game_similarities.drop(selected_game_id, errors='ignore')
            
            # Filter by minimum similarity
            game_similarities = game_similarities[game_similarities >= min_similarity]
            
            if len(game_similarities) == 0:
                return []
            
            # Get top similar games (more than needed for filtering)
            top_similar_games = game_similarities.nlargest(n_recommendations * 3)
            
            recommendations = []
            for game_id, similarity_score in top_similar_games.items():
                # Get game information
                game_info = self.games_df[self.games_df['game_id'] == game_id]
                
                if not game_info.empty:
                    game_data = game_info.iloc[0]
                    avg_rating = self.game_means.get(game_id, 3.0)
                    
                    # Filter by minimum rating
                    if avg_rating >= min_rating:
                        rec = {
                            'game_id': game_id,
                            'name': game_data.get('name', f'Game {game_id}'),
                            'genre': game_data.get('genre', 'Unknown'),
                            'developer': game_data.get('developer', 'Unknown'),
                            'avg_rating': avg_rating,
                            'similarity_score': similarity_score
                        }
                        recommendations.append(rec)
            
            # Sort by combined score (rating + similarity)
            recommendations.sort(key=lambda x: (x['avg_rating'] * 0.3 + x['similarity_score'] * 0.7), reverse=True)
            
            return recommendations[:n_recommendations]
            
        except Exception as e:
            raise Exception(f"Error generating recommendations: {str(e)}")
    
    def get_recommendation_summary(self, selected_game_id, recommendations):
        """
        Generate a summary explaining why games were recommended
        """
        try:
            if not recommendations:
                return {
                    'explanation': "No recommendations found with current criteria.",
                    'genre_analysis': {},
                    'rating_analysis': {},
                    'similarity_analysis': {}
                }
            
            # Get selected game info
            selected_game_info = self.games_df[self.games_df['game_id'] == selected_game_id]
            if selected_game_info.empty:
                selected_genre = "Unknown"
                selected_rating = self.game_means.get(selected_game_id, 3.0)
            else:
                selected_genre = selected_game_info.iloc[0].get('genre', 'Unknown')
                selected_rating = self.game_means.get(selected_game_id, 3.0)
            
            # Analyze recommendations
            genres = [rec['genre'] for rec in recommendations]
            ratings = [rec['avg_rating'] for rec in recommendations]
            similarities = [rec['similarity_score'] for rec in recommendations]
            
            # Genre analysis
            genre_counts = pd.Series(genres).value_counts()
            same_genre_count = sum(1 for genre in genres if genre == selected_genre)
            
            # Rating analysis
            avg_rec_rating = np.mean(ratings)
            high_rating_count = sum(1 for rating in ratings if rating >= 4.0)
            
            # Similarity analysis
            avg_similarity = np.mean(similarities)
            high_similarity_count = sum(1 for sim in similarities if sim >= 0.3)
            
            # Generate explanation
            explanation_parts = []
            
            if same_genre_count > 0:
                explanation_parts.append(f"{same_genre_count} games share the same genre ({selected_genre})")
            
            if high_rating_count > 0:
                explanation_parts.append(f"{high_rating_count} games have high ratings (4.0+)")
            
            if high_similarity_count > 0:
                explanation_parts.append(f"{high_similarity_count} games have strong similarity (30%+)")
            
            explanation = "These games were recommended because: " + ", ".join(explanation_parts) + "."
            
            return {
                'explanation': explanation,
                'genre_analysis': {
                    'selected_genre': selected_genre,
                    'same_genre_count': same_genre_count,
                    'genre_distribution': genre_counts.to_dict()
                },
                'rating_analysis': {
                    'selected_rating': selected_rating,
                    'avg_recommended_rating': avg_rec_rating,
                    'high_rating_count': high_rating_count
                },
                'similarity_analysis': {
                    'avg_similarity': avg_similarity,
                    'high_similarity_count': high_similarity_count,
                    'similarity_range': f"{min(similarities):.2f} - {max(similarities):.2f}"
                }
            }
            
        except Exception as e:
            return {
                'explanation': f"Error generating summary: {str(e)}",
                'genre_analysis': {},
                'rating_analysis': {},
                'similarity_analysis': {}
            }
    
    def get_available_games(self):
        """Get list of available games for selection"""
        if self.games_df is None:
            return []
        
        # Filter games that are in similarity matrix
        if self.similarity_df is not None:
            available_games = self.games_df[self.games_df['game_id'].isin(self.similarity_df.index)].copy()
        else:
            available_games = self.games_df.copy()
        
        # Add average rating to each game
        game_list = []
        for _, row in available_games.iterrows():
            game_id = row['game_id']
            avg_rating = self.game_means.get(game_id, 3.0) if self.game_means else 3.0
            
            game_list.append({
                'game_id': game_id,
                'name': row['name'],
                'genre': row.get('genre', 'Unknown'),
                'developer': row.get('developer', 'Unknown'),
                'avg_rating': avg_rating
            })
        
        # Sort by rating descending
        game_list.sort(key=lambda x: x['avg_rating'], reverse=True)
        return game_list