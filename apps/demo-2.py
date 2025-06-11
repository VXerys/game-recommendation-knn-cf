"""
app.py - Improved Streamlit Game Recommendation Application
Features: Beginner-friendly UI, modular ML integration, recommendation summaries
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from model import GameRecommendationModel
import time

# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="🎮 Game Finder - Discover Your Next Favorite Game",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS FOR BETTER UI =====
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #566573;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .step-header {
        font-size: 1.5rem;
        color: #2E86C1;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2E86C1;
        padding-bottom: 0.5rem;
    }
    .game-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #2E86C1;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        color: #2c3e50;
    }
    .game-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .metric-card {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .rating-badge {
        background-color: #27AE60;
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
        font-size: 0.9rem;
    }
    .rating-badge.medium {
        background-color: #F39C12;
        color: white;
    }
    .rating-badge.low {
        background-color: #E74C3C;
        color: white;
    }
    .similarity-badge {
        background-color: #3498DB;
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
        font-size: 0.9rem;
    }
    .summary-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        border-left: 5px solid #e17055;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #2c3e50;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
        color: #1565c0;
    }
    .success-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        color: #2e7d32;
    }
    .stButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2980b9 0%, #1f618d 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ===== HELPER FUNCTIONS =====
@st.cache_data
def load_model_data():
    """Load and initialize the recommendation model"""
    model = GameRecommendationModel()
    try:
        stats = model.load_and_process_data()
        model.build_similarity_matrix()
        return model, stats, None
    except Exception as e:
        return None, None, str(e)

def display_game_card(rank, rec):
    """Display a game recommendation card with improved styling"""
    # Determine rating badge class
    if rec['avg_rating'] >= 4.5:
        rating_class = "rating-badge"
    elif rec['avg_rating'] >= 3.5:
        rating_class = "rating-badge medium"
    else:
        rating_class = "rating-badge low"
    
    # Create the game card
    with st.container():
        st.markdown(f"""
        <div class="game-card">
            <h3>🏆 #{rank} {rec['name']}</h3>
            <div style="margin: 1rem 0;">
                <strong>🎮 Genre:</strong> {rec['genre']}<br>
                <strong>🏢 Developer:</strong> {rec['developer']}
            </div>
            <div style="display: flex; gap: 10px; align-items: center; margin-top: 1rem;">
                <span class="{rating_class}">⭐ {rec['avg_rating']:.1f}/5.0</span>
                <span class="similarity-badge">🔗 {rec['similarity_score']*100:.1f}% Match</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

def create_summary_visualization(summary_data):
    """Create visualizations for the recommendation summary"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Genre distribution pie chart
        if summary_data['genre_analysis']['genre_distribution']:
            genre_data = summary_data['genre_analysis']['genre_distribution']
            fig_genre = px.pie(
                values=list(genre_data.values()),
                names=list(genre_data.keys()),
                title="📊 Recommended Games by Genre",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_genre.update_layout(
                font=dict(size=12),
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig_genre, use_container_width=True)
    
    with col2:
        # Rating and similarity metrics
        rating_data = summary_data['rating_analysis']
        similarity_data = summary_data['similarity_analysis']
        
        fig_metrics = go.Figure()
        
        # Add rating bar
        fig_metrics.add_trace(go.Bar(
            x=['Selected Game', 'Recommended Games'],
            y=[rating_data['selected_rating'], rating_data['avg_recommended_rating']],
            name='Average Rating',
            marker_color=['#3498db', '#2ecc71'],
            text=[f"{rating_data['selected_rating']:.1f}", f"{rating_data['avg_recommended_rating']:.1f}"],
            textposition='auto'
        ))
        
        fig_metrics.update_layout(
            title="📈 Rating Comparison",
            yaxis_title="Rating (1-5)",
            showlegend=False,
            height=400,
            font=dict(size=12)
        )
        
        st.plotly_chart(fig_metrics, use_container_width=True)

# ===== MAIN APPLICATION =====
def main():
    # Header
    st.markdown('<h1 class="main-header">🎮 Game Finder</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover your next favorite game using AI-powered recommendations</p>', unsafe_allow_html=True)
    
    # Initialize model with loading indicator
    with st.spinner('🔄 Loading game database and training AI model...'):
        model, stats, error = load_model_data()
    
    if error:
        st.error(f"❌ **Error loading data:** {error}")
        st.info("📂 **Required files:** Please ensure `data/game_ratings.csv` and `data/games_metadata_5k.csv` are available.")
        return
    
    if model is None:
        st.error("❌ **Failed to initialize the recommendation model.**")
        return
    
    # Success message
    st.markdown(f"""
    <div class="success-box">
        ✅ <strong>Model loaded successfully!</strong><br>
        📊 Database contains <strong>{stats['n_ratings']:,}</strong> ratings from <strong>{stats['n_users']:,}</strong> users for <strong>{stats['n_games']:,}</strong> games
    </div>
    """, unsafe_allow_html=True)
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'<div class="metric-card"><h3>{stats["n_ratings"]:,}</h3><p>Total Ratings</p></div>', 
                   unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>{stats["n_users"]:,}</h3><p>Active Users</p></div>', 
                   unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h3>{stats["n_games"]:,}</h3><p>Games Available</p></div>', 
                   unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h3>{stats["sparsity"]:.1%}</h3><p>Data Coverage</p></div>', 
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Step 1: Game Selection
    st.markdown('<h2 class="step-header">🎯 Step 1: Choose a Game You Like</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        💡 <strong>How it works:</strong> Select a game you enjoyed, and our AI will find similar games based on what other players with similar tastes liked. 
        The system uses advanced machine learning with K=30 neighbors for accurate recommendations.
    </div>
    """, unsafe_allow_html=True)
    
    # Get available games
    available_games = model.get_available_games()
    
    if not available_games:
        st.error("❌ No games available for recommendation.")
        return
    
    # Create game selection interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Prepare game options for dropdown
        game_options = {}
        for game in available_games:
            display_name = f"{game['name']} ({game['genre']}) - ⭐{game['avg_rating']:.1f}"
            game_options[display_name] = game['game_id']
        
        selected_game_display = st.selectbox(
            "🎮 **Select a game you enjoyed:**",
            options=list(game_options.keys()),
            help="Choose from our database of highly-rated games. The rating shown is the average from all users."
        )
        
        selected_game_id = game_options[selected_game_display]
    
    with col2:
        # Display selected game info
        selected_game = next(game for game in available_games if game['game_id'] == selected_game_id)
        st.markdown("### 📋 Selected Game")
        st.markdown(f"**Name:** {selected_game['name']}")
        st.markdown(f"**Genre:** {selected_game['genre']}")
        st.markdown(f"**Developer:** {selected_game['developer']}")
        st.markdown(f"**Rating:** ⭐{selected_game['avg_rating']:.1f}/5.0")
    
    # Step 2: Recommendation Settings
    st.markdown('<h2 class="step-header">⚙️ Step 2: Customize Your Recommendations</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_rating = st.slider(
            "⭐ **Minimum Rating**",
            min_value=3.0,
            max_value=5.0,
            value=4.0,
            step=0.1,
            help="Only show games with at least this rating"
        )
    
    with col2:
        min_similarity = st.slider(
            "🔗 **Minimum Similarity**",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Only show games with at least this similarity percentage"
        )
    
    with col3:
        n_recommendations = st.slider(
            "📊 **Number of Recommendations**",
            min_value=3,
            max_value=10,
            value=5,
            help="How many game recommendations to show"
        )
    
    # Step 3: Generate Recommendations
    st.markdown('<h2 class="step-header">🚀 Step 3: Get Your Recommendations</h2>', unsafe_allow_html=True)
    
    if st.button("🎯 **Find Similar Games**", type="primary"):
        with st.spinner('🤖 AI is analyzing thousands of games to find the best matches for you...'):
            # Simulate processing time for better UX
            time.sleep(2)
            
            # Generate recommendations
            recommendations = model.get_game_recommendations(
                selected_game_id=selected_game_id,
                n_recommendations=n_recommendations,
                min_rating=min_rating,
                min_similarity=min_similarity
            )
            
            if not recommendations:
                st.warning("⚠️ **No games found** matching your criteria. Try lowering the minimum rating or similarity threshold.")
                return
            
            # Display recommendations
            st.markdown('<h2 class="step-header">🎮 Your Personalized Game Recommendations</h2>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="success-box">
                🎉 <strong>Found {len(recommendations)} great games for you!</strong><br>
                These recommendations are based on similarity to <strong>{selected_game['name']}</strong> and preferences of users with similar tastes.
            </div>
            """, unsafe_allow_html=True)
            
            # Display each recommendation
            for i, rec in enumerate(recommendations, 1):
                display_game_card(i, rec)
            
            # Generate and display summary
            st.markdown('<h2 class="step-header">📊 Why These Games Were Recommended</h2>', unsafe_allow_html=True)
            
            summary_data = model.get_recommendation_summary(selected_game_id, recommendations)
            
            # Main explanation - REVISED SECTION
            st.markdown(f"""
            <div class="summary-box">
                <h3>🧠 AI Analysis Summary</h3>
                <p style="font-size: 1.1rem; margin-bottom: 1rem;">{summary_data['explanation']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Replace the grid HTML with Streamlit columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div style="background-color: #72ACD5FF; padding: 1rem; border-radius: 10px; text-align: center;">
                    <strong>🎮 Genre Match</strong><br>
                    {summary_data['genre_analysis']['same_genre_count']} games share the same genre
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style="background-color: #85B485FF; padding: 1rem; border-radius: 10px; text-align: center;">
                    <strong>⭐ High Quality</strong><br>
                    {summary_data['rating_analysis']['high_rating_count']} games have 4.0+ ratings
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div style="background-color: #C4AD88FF; padding: 1rem; border-radius: 10px; text-align: center;">
                    <strong>🔗 Strong Similarity</strong><br>
                    Average match: {summary_data['similarity_analysis']['avg_similarity']*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed analysis with visualizations
            st.markdown("### 📈 Detailed Analysis")
            create_summary_visualization(summary_data)
            
            # Top 5 Most Similar Games section
            if len(recommendations) >= 3:
                st.markdown("### 🏆 Top 5 Most Similar Games")
                top_similar = sorted(recommendations, key=lambda x: x['similarity_score'], reverse=True)[:5]
                
                for i, game in enumerate(top_similar, 1):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**{i}. {game['name']}**")
                    with col2:
                        st.write(f"⭐ {game['avg_rating']:.1f}")
                    with col3:
                        st.write(f"🔗 {game['similarity_score']*100:.1f}%")
            
            # Because You Liked section
            st.markdown(f"### 💝 Because You Liked '{selected_game['name']}'")
            st.markdown(f"""
            Players who enjoyed **{selected_game['name']}** ({selected_game['genre']}) also loved these games. 
            Our AI found these matches by analyzing the gaming patterns of thousands of users with similar preferences.
            """)
            
            # Additional insights
            with st.expander("🔍 **Learn More About Our Recommendation System**"):
                st.markdown(f"""
                **How it works:**
                - 🤖 **Algorithm:** Item-Based Collaborative Filtering with K=30 neighbors
                - 📊 **Data:** Analyzed {stats['n_ratings']:,} ratings from {stats['n_users']:,} users
                - 🎯 **Similarity:** Uses cosine similarity with mean-centering for accuracy
                - ⚡ **Performance:** Optimized for both accuracy and recommendation diversity
                
                **Your settings:**
                - Minimum rating: {min_rating}/5.0
                - Minimum similarity: {min_similarity*100:.0f}%
                - Number of recommendations: {n_recommendations}
                """)

if __name__ == "__main__":
    main()