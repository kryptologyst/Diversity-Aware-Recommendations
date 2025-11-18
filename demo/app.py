"""Streamlit demo for diversity-aware recommendations."""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.loader import DataLoader
from src.evaluation.metrics import RecommendationMetrics
from src.utils.helpers import load_config, calculate_item_popularity

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Diversity-Aware Recommendations",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-item {
        background-color: #ffffff;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data_and_models(config_path: str = "configs/default.yaml"):
    """Load data and models with caching."""
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Load data
        data_loader = DataLoader(config)
        interactions = data_loader.load_interactions()
        items = data_loader.load_items()
        
        # Load models
        models = {}
        model_dir = Path("models")
        
        if model_dir.exists():
            for model_file in model_dir.glob("*_model.pkl"):
                model_name = model_file.stem.replace("_model", "")
                
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                    models[model_name] = model
        
        return config, interactions, items, models
    
    except Exception as e:
        st.error(f"Failed to load data and models: {e}")
        return None, None, None, {}


def display_recommendations(
    recommendations: List[Tuple[int, float]], 
    items: pd.DataFrame,
    model_name: str
) -> None:
    """Display recommendations in a nice format."""
    st.subheader(f"Recommendations from {model_name.replace('_', ' ').title()}")
    
    if not recommendations:
        st.warning("No recommendations available for this user.")
        return
    
    for i, (item_id, score) in enumerate(recommendations[:10], 1):
        # Get item information
        item_info = items[items["item_id"] == item_id]
        
        if not item_info.empty:
            title = item_info.iloc[0]["title"]
            tags = item_info.iloc[0].get("tags", "No tags")
            description = item_info.iloc[0].get("text", "No description")
        else:
            title = f"Item {item_id}"
            tags = "Unknown"
            description = "No description available"
        
        with st.container():
            st.markdown(f"""
            <div class="recommendation-item">
                <strong>{i}. {title}</strong><br>
                <small>Score: {score:.3f} | Tags: {tags}</small><br>
                <em>{description}</em>
            </div>
            """, unsafe_allow_html=True)


def plot_diversity_metrics(results: Dict[str, Dict[str, float]]) -> None:
    """Plot diversity metrics comparison."""
    if not results:
        return
    
    diversity_metrics = ["diversity", "novelty", "coverage", "popularity_bias"]
    
    # Filter metrics that exist in results
    available_metrics = []
    for metric in diversity_metrics:
        if any(metric in model_results for model_results in results.values()):
            available_metrics.append(metric)
    
    if not available_metrics:
        return
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[metric.replace('_', ' ').title() for metric in available_metrics],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, metric in enumerate(available_metrics):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        model_names = []
        metric_values = []
        
        for model_name, model_results in results.items():
            if metric in model_results:
                model_names.append(model_name.replace('_', ' ').title())
                metric_values.append(model_results[metric])
        
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=metric_values,
                name=metric,
                marker_color=colors[i % len(colors)],
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=600,
        title_text="Diversity Metrics Comparison",
        title_x=0.5
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_relevance_metrics(results: Dict[str, Dict[str, float]]) -> None:
    """Plot relevance metrics comparison."""
    if not results:
        return
    
    relevance_metrics = ["precision@5", "precision@10", "recall@5", "recall@10", "ndcg@5", "ndcg@10"]
    
    # Filter metrics that exist in results
    available_metrics = []
    for metric in relevance_metrics:
        if any(metric in model_results for model_results in results.values()):
            available_metrics.append(metric)
    
    if not available_metrics:
        return
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[metric.replace('@', '@').title() for metric in available_metrics],
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, metric in enumerate(available_metrics):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        model_names = []
        metric_values = []
        
        for model_name, model_results in results.items():
            if metric in model_results:
                model_names.append(model_name.replace('_', ' ').title())
                metric_values.append(model_results[metric])
        
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=metric_values,
                name=metric,
                marker_color=colors[i % len(colors)],
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=600,
        title_text="Relevance Metrics Comparison",
        title_x=0.5
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit application."""
    st.markdown('<h1 class="main-header">🎯 Diversity-Aware Recommendations</h1>', unsafe_allow_html=True)
    
    # Load data and models
    config, interactions, items, models = load_data_and_models()
    
    if config is None:
        st.error("Failed to load configuration. Please check the config file.")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Recommendations", "Model Comparison", "Data Analysis"]
    )
    
    if page == "Recommendations":
        st.header("🎯 Get Recommendations")
        
        if not models:
            st.error("No trained models found. Please run the training script first.")
            return
        
        # User selection
        available_users = sorted(interactions["user_id"].unique())
        selected_user = st.selectbox(
            "Select a user:",
            available_users,
            index=0
        )
        
        # Model selection
        selected_model = st.selectbox(
            "Select a model:",
            list(models.keys()),
            index=0
        )
        
        # Number of recommendations
        n_recs = st.slider("Number of recommendations:", 5, 50, 10)
        
        # Get recommendations
        if st.button("Get Recommendations"):
            try:
                model = models[selected_model]
                recommendations = model.recommend(selected_user, n_recs)
                
                # Display recommendations
                display_recommendations(recommendations, items, selected_model)
                
                # Show user's interaction history
                user_interactions = interactions[interactions["user_id"] == selected_user]
                st.subheader("User's Interaction History")
                
                if not user_interactions.empty:
                    user_items = user_interactions.merge(items, on="item_id")
                    st.dataframe(
                        user_items[["title", "weight", "tags"]].head(10),
                        use_container_width=True
                    )
                else:
                    st.info("No interaction history found for this user.")
                
            except Exception as e:
                st.error(f"Failed to generate recommendations: {e}")
    
    elif page == "Model Comparison":
        st.header("📊 Model Comparison")
        
        # Load evaluation results if available
        results_path = Path("results/evaluation_results.csv")
        if results_path.exists():
            results_df = pd.read_csv(results_path, index_col=0)
            results = results_df.to_dict('index')
            
            # Display results table
            st.subheader("Evaluation Results")
            st.dataframe(results_df, use_container_width=True)
            
            # Plot metrics
            st.subheader("Relevance Metrics")
            plot_relevance_metrics(results)
            
            st.subheader("Diversity Metrics")
            plot_diversity_metrics(results)
            
        else:
            st.warning("No evaluation results found. Please run the evaluation script first.")
    
    elif page == "Data Analysis":
        st.header("📈 Data Analysis")
        
        # Data overview
        st.subheader("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", len(interactions["user_id"].unique()))
        
        with col2:
            st.metric("Total Items", len(interactions["item_id"].unique()))
        
        with col3:
            st.metric("Total Interactions", len(interactions))
        
        with col4:
            sparsity = 1 - len(interactions) / (len(interactions["user_id"].unique()) * len(interactions["item_id"].unique()))
            st.metric("Sparsity", f"{sparsity:.4f}")
        
        # Interaction distribution
        st.subheader("Interaction Distribution")
        
        # User activity
        user_activity = interactions["user_id"].value_counts()
        
        fig1 = px.histogram(
            x=user_activity.values,
            nbins=50,
            title="Distribution of User Activity",
            labels={"x": "Number of Interactions", "y": "Number of Users"}
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Item popularity
        item_popularity = interactions["item_id"].value_counts()
        
        fig2 = px.histogram(
            x=item_popularity.values,
            nbins=50,
            title="Distribution of Item Popularity",
            labels={"x": "Number of Interactions", "y": "Number of Items"}
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Weight distribution
        if "weight" in interactions.columns:
            fig3 = px.histogram(
                interactions,
                x="weight",
                title="Distribution of Interaction Weights",
                labels={"x": "Weight", "y": "Count"}
            )
            st.plotly_chart(fig3, use_container_width=True)


if __name__ == "__main__":
    main()
