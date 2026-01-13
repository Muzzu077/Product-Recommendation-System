import streamlit as st
import pandas as pd
import time
import os
from src.data.loader import load_data
from src.recommenders.rule_based import RuleBasedRecommender
from src.recommenders.collaborative import CollaborativeRecommender
from src.recommenders.deep_learning import DeepLearningRecommender
from src.evaluation import calculate_rmse
import tensorflow as tf

# Page Config
st.set_page_config(
    page_title="Product Recommendation System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for High-End UI
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #d93d3d;
        transform: translateY(-2px);
    }
    
    /* Product Card Style */
    .product-card {
        background-color: #262730;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #363945;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .product-card:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border-color: #ff4b4b;
    }
    .product-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 4px;
        color: #e0e0e0;
    }
    .product-category {
        font-size: 0.8rem;
        color: #aaaaaa;
        margin-bottom: 8px;
        font-style: italic;
    }
    .product-score {
        font-size: 0.9rem;
        color: #ff4b4b;
    }
    .badge {
        background-color: #333;
        color: #fff;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #16181e;
        border-right: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# Load Data & Models (Cached)
# --------------------------------------------------------------------------------
@st.cache_data
def get_data():
    ratings, items = load_data('.') # Assumes files in root or fix path
    return ratings, items

@st.cache_resource
def get_rule_based_model(_ratings, _items):
    return RuleBasedRecommender(_ratings, _items)

@st.cache_resource
def get_collaborative_model(_ratings, _items):
    return CollaborativeRecommender(_ratings, _items)

@st.cache_resource
def get_dl_model(_ratings, _items):
    # This might take a few seconds
    model = DeepLearningRecommender(_ratings, _items)
    with st.spinner('Initializing Neural Network...'):
        model.train(epochs=3) # Short training for demo speed
    return model

try:
    ratings, items = get_data()
except FileNotFoundError:
    st.error("Dataset not found! Please ensure 'u.data' and 'u.item' are in the project root or data directory.")
    st.stop()

# Initialize Recommenders
rb_engine = get_rule_based_model(ratings, items)
cf_engine = get_collaborative_model(ratings, items)
dl_engine = get_dl_model(ratings, items)

# --------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------
with st.sidebar:
    st.title("🛍️ Config")
    st.markdown("---")
    
    user_ids = ratings['user_id'].unique()
    selected_user = st.selectbox("Select Customer ID", sorted(user_ids))
    
    st.markdown("### Recommendation Engine")
    method = st.radio(
        "Choose Algorithm:",
        ("Rule-Based (Popularity)", "Rule-Based (Top Rated)", "Collaborative Filtering", "Deep Learning","Hybrid (Conceptual)")
    )
    
    st.markdown("---")
    st.info("System Status: **Online** 🟢")
    st.markdown("**Dataset Info:**")
    st.text(f"Users: {len(user_ids)}")
    st.text(f"Products: {len(items)}")
    st.text(f"Interactions: {len(ratings)}")

# --------------------------------------------------------------------------------
# Main Content
# --------------------------------------------------------------------------------
st.title("🛒 Intelligent Product Recommendation System")
st.markdown("### Leveraging AI to personalize shopping experiences.")

# User Profile Section
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown(f"""
    <div style="background-color: #262730; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b;">
        <h4>Customer Profile</h4>
        <p><strong>ID:</strong> {selected_user}</p>
        <p><strong>Total Orders:</strong> {len(ratings[ratings['user_id'] == selected_user])}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    st.markdown("**Recent Interactions:**")
    user_history = ratings[ratings['user_id'] == selected_user].merge(items, on='product_id').head(5)
    for idx, row in user_history.iterrows():
        st.caption(f"⭐ {row['rating']} - {row['product_name']} ({row['category']})")

with col2:
    st.subheader(f"Recommendations using {method}")
    
    start_time = time.time()
    recs = pd.DataFrame()
    
    if method == "Rule-Based (Popularity)":
        recs = rb_engine.get_recommendations(method='popular')
    elif method == "Rule-Based (Top Rated)":
        recs = rb_engine.get_recommendations(method='top_rated')
    elif method == "Collaborative Filtering":
        recs = cf_engine.recommend(selected_user)
    elif method == "Deep Learning":
        recs = dl_engine.recommend(selected_user)
    elif method == "Hybrid (Conceptual)":
        st.warning("Hybrid method is conceptual in this demo. Showing Deep Learning results as proxy.")
        recs = dl_engine.recommend(selected_user)

    end_time = time.time()
    elapsed = end_time - start_time
    
    if not recs.empty:
        st.success(f"Generated recommendations in {elapsed:.4f} seconds")
        
        # Display Grid
        # Create 2 columns for grid layout
        grid_cols = st.columns(2)
        
        for i, row in recs.iterrows():
            col_idx = i % 2
            with grid_cols[col_idx]:
                 score_display = f"Score: {row['score']:.2f}" if 'score' in row else "Top Pick"
                 
                 st.markdown(f"""
                 <div class="product-card">
                    <div class="product-title">📦 {row['product_name']}</div>
                    <div class="product-category">{row['category']}</div>
                    <div class="product-score">{score_display}</div>
                 </div>
                 """, unsafe_allow_html=True)
    else:
        st.info("No recommendations found for this criteria.")
        
# --------------------------------------------------------------------------------
# Comparison Section
# --------------------------------------------------------------------------------
st.markdown("---")
with st.expander("📊 Algorithm Comparison & Insights"):
    st.markdown("""
    ### Performance Metrics
    
    | Method | Accuracy | Personalization | Cold Start Handling |
    |--------|----------|-----------------|---------------------|
    | Rule-Based | Low | ❌ No | ✅ Excellent |
    | Collaborative Filtering | Medium | ✅ Yes | ⚠️ Poor |
    | Deep Learning | High | ✅ High | ⚠️ Moderate |
    
    **Why Deep Learning?**
    It captures non-linear relationships and interactions better than linear matrix factorization methods.
    """)
    
    if method == "Deep Learning" or method == "Collaborative Filtering":
        st.bar_chart(recs.set_index('product_name')['score'].head(10))

