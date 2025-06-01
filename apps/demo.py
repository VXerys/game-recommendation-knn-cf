import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ===== KONFIGURASI HALAMAN =====
st.set_page_config(
    page_title="🎮 Game Recommendation System",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== SIMPLIFIED CSS - FOKUS PADA KOMPATIBILITAS STREAMLIT =====
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #566573;
        text-align: center;
        margin-bottom: 2rem;
    }
    .game-card {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E86C1;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .rating-badge {
        background-color: #27AE60;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
        margin: 5px 0;
    }
    .rating-badge.medium {
        background-color: #F39C12;
    }
    .rating-badge.low {
        background-color: #E74C3C;
    }
    .similarity-badge {
        background-color: #3498DB;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ===== FUNGSI UTILITAS =====
@st.cache_data
def load_and_process_data():
    """Load dan preprocess data dengan error handling yang robust"""
    try:
        # Coba berbagai path yang mungkin
        possible_paths = [
            ('data/game_ratings.csv', 'data/games_metadata_5k.csv'),
            ('game_ratings.csv', 'games_metadata_5k.csv'),
            ('../data/game_ratings.csv', '../data/games_metadata_5k.csv')
        ]
        
        ratings_df = None
        games_df = None
        
        for ratings_path, games_path in possible_paths:
            try:
                ratings_df = pd.read_csv(ratings_path)
                games_df = pd.read_csv(games_path)
                st.success(f"✅ Data berhasil dimuat dari {ratings_path}")
                break
            except FileNotFoundError:
                continue
        
        if ratings_df is None or games_df is None:
            st.error("❌ File data tidak ditemukan! Pastikan file berada di folder yang benar.")
            st.info("📂 File yang dibutuhkan: `data/game_ratings.csv` dan `data/games_metadata_5k.csv`")
            return None, None, None, None, None, None
        
        # Validasi kolom yang dibutuhkan
        required_rating_cols = ['user_id', 'game_id', 'rating']
        required_game_cols = ['game_id', 'name']
        
        if not all(col in ratings_df.columns for col in required_rating_cols):
            st.error(f"❌ Kolom yang dibutuhkan di ratings: {required_rating_cols}")
            return None, None, None, None, None, None
            
        if not all(col in games_df.columns for col in required_game_cols):
            st.error(f"❌ Kolom yang dibutuhkan di games: {required_game_cols}")
            return None, None, None, None, None, None
        
        # Preprocessing data
        st.info("🔄 Memproses data...")
        
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
            st.error("❌ Tidak ada data yang tersisa setelah filtering!")
            return None, None, None, None, None, None
        
        # Pastikan games_df hanya berisi game yang valid
        games_df = games_df[games_df['game_id'].isin(valid_games)].copy()
        
        # Buat mapping game_id ke nama untuk dropdown
        game_names = games_df.set_index('game_id')['name'].to_dict()
        
        # Convert Index to list untuk memudahkan caching
        valid_users_list = valid_users.tolist()
        valid_games_list = valid_games.tolist()
        
        return filtered_ratings, games_df, game_names, valid_users_list, valid_games_list, MIN_RATINGS
        
    except Exception as e:
        st.error(f"❌ Error dalam memuat data: {str(e)}")
        return None, None, None, None, None, None

@st.cache_resource
def build_similarity_matrix(_filtered_ratings, _valid_games):
    """Build similarity matrix dengan error handling"""
    try:
        if _filtered_ratings is None or len(_filtered_ratings) == 0:
            return None, None
        
        st.info("🔄 Membangun similarity matrix...")
        
        # Split data untuk training (80%)
        np.random.seed(42)
        train_indices = np.random.choice(
            _filtered_ratings.index, 
            size=int(len(_filtered_ratings) * 0.8), 
            replace=False
        )
        train_set = _filtered_ratings.loc[train_indices].copy()
        
        # Hitung rata-rata rating per game
        game_means = train_set.groupby('game_id')['rating'].mean().to_dict()
        
        # Buat user-item matrix
        user_item_matrix = pd.pivot_table(
            train_set, 
            values='rating', 
            index='user_id', 
            columns='game_id', 
            fill_value=0
        )
        
        # Mean-centering untuk similarity calculation
        centered_ratings = train_set.copy()
        centered_ratings['rating_centered'] = centered_ratings.apply(
            lambda row: row['rating'] - game_means.get(row['game_id'], 3.0), 
            axis=1
        )
        
        # Buat item-user matrix (games sebagai rows)
        item_user_matrix = pd.pivot_table(
            centered_ratings, 
            values='rating_centered', 
            index='game_id', 
            columns='user_id', 
            fill_value=0
        )
        
        # Hitung cosine similarity antar games
        if len(item_user_matrix) > 1:
            similarity_matrix = cosine_similarity(item_user_matrix.values)
            similarity_df = pd.DataFrame(
                similarity_matrix, 
                index=item_user_matrix.index, 
                columns=item_user_matrix.index
            )
            # Hapus self-similarity
            np.fill_diagonal(similarity_df.values, 0)
        else:
            similarity_df = pd.DataFrame()
        
        return similarity_df, game_means
        
    except Exception as e:
        st.error(f"❌ Error dalam membangun model: {str(e)}")
        return None, None

def get_game_recommendations(selected_game_id, similarity_df, games_df, game_means, 
                           n_recommendations=10, min_rating=4.0, min_similarity=0.1):
    """
    Generate rekomendasi dengan filtering berdasarkan rating dan similarity
    
    Parameters:
    - min_rating: rating minimal yang diinginkan (default 4.0)
    - min_similarity: similarity minimal (default 0.1)
    - n_recommendations: jumlah awal kandidat sebelum filtering (default 10)
    """
    try:
        if similarity_df is None or similarity_df.empty:
            return []
        
        if selected_game_id not in similarity_df.index:
            st.warning(f"⚠️ Game ID {selected_game_id} tidak ditemukan dalam similarity matrix")
            return []
        
        # Ambil similarity scores untuk game yang dipilih
        game_similarities = similarity_df.loc[selected_game_id].copy()
        
        # Hapus game yang dipilih sendiri
        game_similarities = game_similarities.drop(selected_game_id, errors='ignore')
        
        # Filter berdasarkan minimum similarity
        game_similarities = game_similarities[game_similarities >= min_similarity]
        
        if len(game_similarities) == 0:
            st.warning(f"⚠️ Tidak ada game dengan similarity >= {min_similarity}")
            return []
        
        # Urutkan berdasarkan similarity tertinggi
        top_similar_games = game_similarities.nlargest(n_recommendations * 3)  # Ambil lebih banyak untuk filtering
        
        recommendations = []
        for game_id, similarity_score in top_similar_games.items():
            # Ambil informasi game dari metadata
            game_info = games_df[games_df['game_id'] == game_id]
            
            if not game_info.empty:
                game_data = game_info.iloc[0]
                avg_rating = game_means.get(game_id, 3.0)
                
                # Filter berdasarkan rating minimal
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
        
        # Urutkan berdasarkan kombinasi rating dan similarity
        recommendations.sort(key=lambda x: (x['avg_rating'] * 0.3 + x['similarity_score'] * 0.7), reverse=True)
        
        return recommendations[:n_recommendations]
        
    except Exception as e:
        st.error(f"❌ Error dalam generate rekomendasi: {str(e)}")
        return []

def display_game_card_native(rank, rec):
    """Display game recommendation menggunakan native Streamlit components"""
    
    # Determine rating color class
    if rec['avg_rating'] >= 4.5:
        rating_class = "rating-badge"
    elif rec['avg_rating'] >= 3.5:
        rating_class = "rating-badge medium"
    else:
        rating_class = "rating-badge low"
    
    # Create columns for layout
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown(f"### 🏆 #{rank} {rec['name']}")
        st.markdown(f"**🎮 Genre:** {rec['genre']}")
        st.markdown(f"**🏢 Developer:** {rec['developer']}")
    
    with col2:
        st.markdown(f'<span class="{rating_class}">⭐ {rec["avg_rating"]:.1f}</span>', 
                   unsafe_allow_html=True)
        
    with col3:
        similarity_percent = rec['similarity_score'] * 100
        st.markdown(f'<span class="similarity-badge">🔗 {similarity_percent:.1f}%</span>', 
                   unsafe_allow_html=True)
    
    # Progress bar untuk similarity
    st.progress(rec['similarity_score'])
    
    st.markdown("---")

# ===== HEADER APLIKASI =====
st.markdown('<h1 class="main-header">🎮 Game Recommendation System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistem Rekomendasi Game Berbasis Item-Based Collaborative Filtering</p>', unsafe_allow_html=True)

# ===== MAIN APPLICATION =====
with st.spinner('⏳ Loading data dan membangun model...'):
    # Load dan process data
    result = load_and_process_data()
    
    if all(x is not None for x in result[:3]):
        filtered_ratings, games_df, game_names, valid_users, valid_games, MIN_RATINGS = result
        
        # Build similarity matrix
        similarity_df, game_means = build_similarity_matrix(filtered_ratings, valid_games)
        
        if similarity_df is not None and not similarity_df.empty:
            # ===== DASHBOARD METRICS =====
            col1, col2, col3, col4 = st.columns(4)
            
            n_users = filtered_ratings['user_id'].nunique()
            n_games = filtered_ratings['game_id'].nunique()
            n_ratings = len(filtered_ratings)
            sparsity = 1 - (n_ratings / (n_users * n_games))
            
            with col1:
                st.markdown(f'<div class="metric-card"><h3>{n_ratings:,}</h3><p>Total Ratings</p></div>', 
                           unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><h3>{n_users:,}</h3><p>Active Users</p></div>', 
                           unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card"><h3>{n_games:,}</h3><p>Games Available</p></div>', 
                           unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="metric-card"><h3>{sparsity:.1%}</h3><p>Matrix Sparsity</p></div>', 
                           unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ===== SIDEBAR CONTROLS =====
            st.sidebar.header("🎯 Pengaturan Rekomendasi")
            
            # Prepare game options untuk dropdown
            available_games = games_df[games_df['game_id'].isin(similarity_df.index)].copy()
            
            if len(available_games) > 0:
                # Buat dictionary untuk dropdown
                game_options = {}
                for _, row in available_games.iterrows():
                    game_id = row['game_id']
                    name = row['name']
                    genre = row.get('genre', 'Unknown')
                    avg_rating = game_means.get(game_id, 3.0)
                    display_name = f"{name} ({genre}) - {avg_rating:.1f}⭐"
                    game_options[display_name] = game_id
                
                # Dropdown pilihan game
                selected_game_display = st.sidebar.selectbox(
                    "Pilih Game yang Anda Sukai:",
                    options=list(game_options.keys()),
                    help="Pilih game untuk mendapatkan rekomendasi serupa"
                )
                
                selected_game_id = game_options[selected_game_display]
                
                # CONTROLS UNTUK FILTERING
                st.sidebar.markdown("### 🔧 Filter Rekomendasi")
                
                min_rating = st.sidebar.slider(
                    "Rating Minimal:",
                    min_value=3.0,
                    max_value=5.0,
                    value=4.0,
                    step=0.1,
                    help="Hanya tampilkan game dengan rating minimal ini"
                )
                
                min_similarity = st.sidebar.slider(
                    "Similarity Minimal:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.05,
                    help="Hanya tampilkan game dengan similarity minimal ini"
                )
                
                n_recommendations = st.sidebar.slider(
                    "Jumlah Rekomendasi:",
                    min_value=3,
                    max_value=10,
                    value=5,
                    help="Tentukan berapa banyak game yang ingin direkomendasikan"
                )
                
                # Informasi game yang dipilih
                st.sidebar.markdown("### 📋 Game yang Dipilih")
                selected_game_info = games_df[games_df['game_id'] == selected_game_id].iloc[0]
                st.sidebar.markdown(f"**{selected_game_info['name']}**")
                st.sidebar.markdown(f"**Genre:** {selected_game_info.get('genre', 'Unknown')}")
                st.sidebar.markdown(f"**Developer:** {selected_game_info.get('developer', 'Unknown')}")
                st.sidebar.markdown(f"**Avg Rating:** {game_means.get(selected_game_id, 3.0):.2f}/5.0")
                
                # ===== MAIN CONTENT =====
                st.header("🎯 Rekomendasi Game Berkualitas Tinggi")
                
                # Generate dan tampilkan rekomendasi
                with st.spinner('🔄 Generating high-quality recommendations...'):
                    recommendations = get_game_recommendations(
                        selected_game_id, similarity_df, games_df, game_means, 
                        n_recommendations, min_rating, min_similarity
                    )
                
                if recommendations and len(recommendations) > 0:
                    st.success(f"✅ Ditemukan {len(recommendations)} rekomendasi berkualitas tinggi (Rating ≥ {min_rating}⭐)!")
                    
                    # Display recommendations menggunakan native Streamlit
                    for i, rec in enumerate(recommendations, 1):
                        with st.container():
                            display_game_card_native(i, rec)
                        
                else:
                    st.warning(f"⚠️ Tidak ditemukan rekomendasi dengan kriteria: Rating ≥ {min_rating}⭐ dan Similarity ≥ {min_similarity}")
                    
                    # Saran alternatif
                    st.info("💡 **Saran:**")
                    st.info(f"- Coba turunkan rating minimal ke 3.5⭐")
                    st.info(f"- Atau turunkan similarity minimal ke 0.05")
                    
                    # Debug info
                    with st.expander("🔍 Debug Information"):
                        if selected_game_id in similarity_df.index:
                            all_similarities = similarity_df.loc[selected_game_id]
                            valid_similarities = all_similarities[all_similarities >= min_similarity]
                            st.write(f"Game dengan similarity ≥ {min_similarity}: {len(valid_similarities)}")
                            
                            if len(valid_similarities) > 0:
                                # Check ratings for these games
                                high_sim_games = valid_similarities.index
                                high_rating_count = sum(1 for gid in high_sim_games 
                                                       if game_means.get(gid, 3.0) >= min_rating)
                                st.write(f"Dari {len(valid_similarities)} game dengan similarity tinggi, {high_rating_count} memiliki rating ≥ {min_rating}")
                
                # ===== VISUALISASI SECTION =====
                st.markdown("---")
                st.header("📊 Data Analytics Dashboard")
                
                # Create two columns for visualizations
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    st.subheader("📈 Distribusi Rating")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    counts, bins, patches = ax.hist(filtered_ratings['rating'], bins=5, 
                                                   color='#3498db', edgecolor='white', 
                                                   linewidth=2, alpha=0.8)
                    
                    colors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60']
                    for i, (patch, color) in enumerate(zip(patches, colors)):
                        patch.set_facecolor(color)
                    
                    ax.set_title('Distribusi Rating Games', fontsize=16, fontweight='bold', pad=20)
                    ax.set_xlabel('Rating (1-5 ⭐)', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Jumlah Rating', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    for i, count in enumerate(counts):
                        ax.text(bins[i] + 0.1, count + max(counts)*0.01, str(int(count)), 
                               ha='center', va='bottom', fontweight='bold')
                    
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_facecolor('#f8f9fa')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with viz_col2:
                    st.subheader("🏆 Top Genres")
                    
                    if 'genre' in games_df.columns:
                        genre_counts = games_df['genre'].value_counts().head(8)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        bars = ax.barh(range(len(genre_counts)), genre_counts.values, 
                                      color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', 
                                            '#9b59b6', '#1abc9c', '#34495e', '#e67e22'])
                        
                        ax.set_yticks(range(len(genre_counts)))
                        ax.set_yticklabels([genre[:15] + '...' if len(genre) > 15 else genre 
                                           for genre in genre_counts.index])
                        ax.set_xlabel('Jumlah Games', fontsize=12, fontweight='bold')
                        ax.set_title('Genre Game Terpopuler', fontsize=16, fontweight='bold', pad=20)
                        
                        for i, (bar, value) in enumerate(zip(bars, genre_counts.values)):
                            ax.text(value + max(genre_counts.values)*0.01, i, str(value), 
                                   va='center', fontweight='bold')
                        
                        ax.grid(True, axis='x', alpha=0.3)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.set_facecolor('#f8f9fa')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.info("📊 Data genre tidak tersedia")
                
                # Additional statistics
                st.markdown("---")
                st.subheader("📋 Statistics Summary")
                
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                
                with stats_col1:
                    avg_rating = filtered_ratings['rating'].mean()
                    st.metric("Rating Rata-rata", f"{avg_rating:.2f}⭐", 
                             delta=f"{avg_rating-3:.2f} dari median")
                
                with stats_col2:
                    ratings_per_user = filtered_ratings.groupby('user_id').size().mean()
                    st.metric("Rata-rata Rating per User", f"{ratings_per_user:.1f}", 
                             delta="ratings")
                
                with stats_col3:
                    ratings_per_game = filtered_ratings.groupby('game_id').size().mean()
                    st.metric("Rata-rata Rating per Game", f"{ratings_per_game:.1f}", 
                             delta="ratings")
                
            else:
                st.error("❌ Tidak ada game yang tersedia untuk rekomendasi!")
        else:
            st.error("❌ Gagal membangun similarity matrix!")
    else:
        st.error("❌ Gagal memuat data. Pastikan file CSV tersedia dan format sudah benar!")

# ===== INFORMASI TEKNIS =====
with st.expander("🔧 Informasi Teknis Model"):
    st.markdown("""
    **🤖 Algoritma:** Item-Based Collaborative Filtering dengan Quality Filtering
    
    **🔢 Similarity Measure:** Cosine Similarity dengan Mean-Centering
    
    **🛠️ Preprocessing:**
    - Filter minimum 5 ratings per user dan game
    - Mean-centering untuk menghilangkan bias rating
    - Quality filtering berdasarkan rating minimal dan similarity threshold
    
    **📊 Model Performance:**
    - Optimal K-neighbors: 20 (dari evaluasi sebelumnya)
    - RMSE: ~0.85 (estimated)
    - MAE: ~0.67 (estimated)
    
    **💡 How it works:**
    1. Pilih game yang Anda sukai
    2. Sistem mencari game lain dengan pola rating serupa (similarity)
    3. Filter game dengan rating tinggi (≥ 4.0⭐ default)
    4. Rekomendasi diurutkan berdasarkan kombinasi rating dan similarity
    """)

# ===== FOOTER =====
st.markdown("---")
st.markdown('<div style="text-align: center; padding: 2rem; color: #566573; font-style: italic;">🚀 Dibuat oleh Tim Machine Learning | Sistem Rekomendasi Game 2024</div>', 
           unsafe_allow_html=True)