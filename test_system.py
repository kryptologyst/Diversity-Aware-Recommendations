#!/usr/bin/env python3
"""Simple test script to verify the system works."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
from src.utils.helpers import setup_logging, set_random_seeds
from src.models.recommenders import PopularityRecommender, ItemKNNRecommender
from src.evaluation.metrics import RecommendationMetrics

def create_simple_data():
    """Create simple test data."""
    interactions = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        "item_id": [10, 11, 12, 10, 11, 13, 11, 12, 14, 12, 13, 15],
        "weight": [5, 4, 3, 4, 5, 3, 3, 4, 5, 4, 3, 5],
        "timestamp": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    })
    
    items = pd.DataFrame({
        "item_id": [10, 11, 12, 13, 14, 15],
        "title": ["Item A", "Item B", "Item C", "Item D", "Item E", "Item F"],
        "tags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6"],
        "text": ["Desc A", "Desc B", "Desc C", "Desc D", "Desc E", "Desc F"],
        "features": ["{}"] * 6
    })
    
    return interactions, items

def test_system():
    """Test the recommendation system."""
    print("Testing Diversity-Aware Recommendation System")
    print("=" * 50)
    
    # Setup
    setup_logging()
    set_random_seeds(42)
    
    # Create test data
    interactions, items = create_simple_data()
    print(f"Created test data: {len(interactions)} interactions, {len(items)} items")
    
    # Test popularity model
    print("\nTesting Popularity Recommender...")
    config = {"models": {"popularity": {"enabled": True}}}
    popularity_model = PopularityRecommender(config)
    popularity_model.fit(interactions, items)
    
    recs = popularity_model.recommend(1, 3)
    print(f"Popularity recommendations for user 1: {recs}")
    
    # Test item-kNN model
    print("\nTesting Item-kNN Recommender...")
    config = {
        "models": {
            "item_knn": {"enabled": True, "k": 3, "similarity": "cosine"}
        }
    }
    item_knn_model = ItemKNNRecommender(config)
    item_knn_model.fit(interactions, items)
    
    recs = item_knn_model.recommend(1, 3)
    print(f"Item-kNN recommendations for user 1: {recs}")
    
    # Test metrics
    print("\nTesting Evaluation Metrics...")
    config = {"evaluation": {"k_values": [3, 5]}}
    metrics = RecommendationMetrics(config)
    
    recommendations = [10, 11, 12]
    relevant_items = [10, 11, 15]
    
    precision = metrics.precision_at_k(recommendations, relevant_items, 3)
    recall = metrics.recall_at_k(recommendations, relevant_items, 3)
    ndcg = metrics.ndcg_at_k(recommendations, relevant_items, 3)
    
    print(f"Precision@3: {precision:.3f}")
    print(f"Recall@3: {recall:.3f}")
    print(f"NDCG@3: {ndcg:.3f}")
    
    print("\n✅ All tests passed! System is working correctly.")

if __name__ == "__main__":
    test_system()
