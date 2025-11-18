"""Unit tests for diversity-aware recommendations."""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from src.data.loader import DataLoader
from src.models.recommenders import (
    PopularityRecommender,
    UserKNNRecommender,
    ItemKNNRecommender,
    DiversityAwareRecommender
)
from src.evaluation.metrics import RecommendationMetrics
from src.utils.helpers import (
    create_user_item_matrix,
    calculate_item_popularity,
    calculate_user_activity,
    filter_cold_start_users,
    filter_cold_start_items
)


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "data": {
                "interactions_file": "data/interactions.csv",
                "items_file": "data/items.csv",
                "users_file": "data/users.csv",
                "min_interactions_per_user": 5,
                "min_interactions_per_item": 5,
                "test_size": 0.2,
                "val_size": 0.1,
                "random_seed": 42
            }
        }
        
        self.sample_interactions = pd.DataFrame({
            "user_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "item_id": [10, 11, 10, 12, 11, 13, 12, 14, 13, 15],
            "weight": [5, 4, 3, 5, 4, 3, 5, 4, 3, 5],
            "timestamp": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        })
        
        self.sample_items = pd.DataFrame({
            "item_id": [10, 11, 12, 13, 14, 15],
            "title": ["Item A", "Item B", "Item C", "Item D", "Item E", "Item F"],
            "tags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6"],
            "text": ["Desc A", "Desc B", "Desc C", "Desc D", "Desc E", "Desc F"],
            "features": ["{}"] * 6
        })
    
    def test_data_loader_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader(self.config)
        self.assertEqual(loader.min_user_interactions, 5)
        self.assertEqual(loader.min_item_interactions, 5)
        self.assertEqual(loader.random_seed, 42)
    
    def test_filter_data(self):
        """Test data filtering functionality."""
        loader = DataLoader(self.config)
        filtered_data = loader.filter_data(self.sample_interactions)
        
        # Should filter out users/items with fewer than 5 interactions
        self.assertLessEqual(len(filtered_data), len(self.sample_interactions))
    
    def test_create_train_val_test_split(self):
        """Test train/validation/test split creation."""
        loader = DataLoader(self.config)
        train, val, test = loader.create_train_val_test_split(self.sample_interactions)
        
        # Check that splits are non-empty and don't overlap
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)
        self.assertEqual(len(train) + len(val) + len(test), len(self.sample_interactions))


class TestRecommenders(unittest.TestCase):
    """Test cases for recommendation models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "models": {
                "popularity": {"enabled": True},
                "user_knn": {"enabled": True, "k": 3, "similarity": "cosine"},
                "item_knn": {"enabled": True, "k": 3, "similarity": "cosine"},
                "diversity_aware": {
                    "enabled": True,
                    "diversity_factor": 0.5,
                    "mmr_lambda": 0.7,
                    "novelty_weight": 0.3
                }
            }
        }
        
        self.interactions = pd.DataFrame({
            "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "item_id": [10, 11, 12, 10, 11, 13, 11, 12, 14],
            "weight": [5, 4, 3, 4, 5, 3, 3, 4, 5],
            "timestamp": [100, 200, 300, 400, 500, 600, 700, 800, 900]
        })
        
        self.items = pd.DataFrame({
            "item_id": [10, 11, 12, 13, 14],
            "title": ["Item A", "Item B", "Item C", "Item D", "Item E"],
            "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
            "text": ["Desc A", "Desc B", "Desc C", "Desc D", "Desc E"],
            "features": ["{}"] * 5
        })
    
    def test_popularity_recommender(self):
        """Test PopularityRecommender."""
        model = PopularityRecommender(self.config)
        model.fit(self.interactions, self.items)
        
        recommendations = model.recommend(1, 3)
        
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 3)
        
        # Check that recommendations are tuples of (item_id, score)
        for rec in recommendations:
            self.assertIsInstance(rec, tuple)
            self.assertEqual(len(rec), 2)
            self.assertIsInstance(rec[0], int)  # item_id
            self.assertIsInstance(rec[1], (int, float))  # score
    
    def test_user_knn_recommender(self):
        """Test UserKNNRecommender."""
        model = UserKNNRecommender(self.config)
        model.fit(self.interactions, self.items)
        
        recommendations = model.recommend(1, 3)
        
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 3)
    
    def test_item_knn_recommender(self):
        """Test ItemKNNRecommender."""
        model = ItemKNNRecommender(self.config)
        model.fit(self.interactions, self.items)
        
        recommendations = model.recommend(1, 3)
        
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 3)
        
        # Test item similarity
        similarities = model.get_item_similarity(10, 3)
        self.assertIsInstance(similarities, list)
    
    def test_diversity_aware_recommender(self):
        """Test DiversityAwareRecommender."""
        model = DiversityAwareRecommender(self.config)
        model.fit(self.interactions, self.items)
        
        recommendations = model.recommend(1, 3)
        
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 3)


class TestMetrics(unittest.TestCase):
    """Test cases for evaluation metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "evaluation": {
                "k_values": [5, 10],
                "metrics": ["precision@5", "recall@5", "ndcg@5"]
            }
        }
        
        self.recommendations = [10, 11, 12, 13, 14]
        self.relevant_items = [10, 11, 15, 16, 17]
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        metrics = RecommendationMetrics(self.config)
        
        precision = metrics.precision_at_k(self.recommendations, self.relevant_items, 5)
        
        self.assertIsInstance(precision, float)
        self.assertGreaterEqual(precision, 0.0)
        self.assertLessEqual(precision, 1.0)
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        metrics = RecommendationMetrics(self.config)
        
        recall = metrics.recall_at_k(self.recommendations, self.relevant_items, 5)
        
        self.assertIsInstance(recall, float)
        self.assertGreaterEqual(recall, 0.0)
        self.assertLessEqual(recall, 1.0)
    
    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        metrics = RecommendationMetrics(self.config)
        
        ndcg = metrics.ndcg_at_k(self.recommendations, self.relevant_items, 5)
        
        self.assertIsInstance(ndcg, float)
        self.assertGreaterEqual(ndcg, 0.0)
        self.assertLessEqual(ndcg, 1.0)
    
    def test_hit_rate_at_k(self):
        """Test hit rate@k calculation."""
        metrics = RecommendationMetrics(self.config)
        
        hit_rate = metrics.hit_rate_at_k(self.recommendations, self.relevant_items, 5)
        
        self.assertIsInstance(hit_rate, float)
        self.assertIn(hit_rate, [0.0, 1.0])
    
    def test_intra_list_diversity(self):
        """Test intra-list diversity calculation."""
        metrics = RecommendationMetrics(self.config)
        
        diversity = metrics.intra_list_diversity(self.recommendations)
        
        self.assertIsInstance(diversity, float)
        self.assertGreaterEqual(diversity, 0.0)
        self.assertLessEqual(diversity, 1.0)
    
    def test_novelty(self):
        """Test novelty calculation."""
        metrics = RecommendationMetrics(self.config)
        
        item_popularity = pd.Series([0.8, 0.6, 0.4, 0.2, 0.1], index=[10, 11, 12, 13, 14])
        novelty = metrics.novelty(self.recommendations, item_popularity)
        
        self.assertIsInstance(novelty, float)
        self.assertGreaterEqual(novelty, 0.0)
        self.assertLessEqual(novelty, 1.0)


class TestHelpers(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.interactions = pd.DataFrame({
            "user_id": [1, 1, 2, 2, 3, 3],
            "item_id": [10, 11, 10, 12, 11, 13],
            "weight": [5, 4, 3, 5, 4, 3],
            "timestamp": [100, 200, 300, 400, 500, 600]
        })
    
    def test_create_user_item_matrix(self):
        """Test user-item matrix creation."""
        matrix = create_user_item_matrix(self.interactions)
        
        self.assertIsInstance(matrix, pd.DataFrame)
        self.assertEqual(matrix.shape[0], len(self.interactions["user_id"].unique()))
        self.assertEqual(matrix.shape[1], len(self.interactions["item_id"].unique()))
    
    def test_calculate_item_popularity(self):
        """Test item popularity calculation."""
        popularity = calculate_item_popularity(self.interactions)
        
        self.assertIsInstance(popularity, pd.Series)
        self.assertEqual(len(popularity), len(self.interactions["item_id"].unique()))
        
        # Check that popularity scores are normalized
        self.assertGreaterEqual(popularity.min(), 0.0)
        self.assertLessEqual(popularity.max(), 1.0)
    
    def test_calculate_user_activity(self):
        """Test user activity calculation."""
        activity = calculate_user_activity(self.interactions)
        
        self.assertIsInstance(activity, pd.Series)
        self.assertEqual(len(activity), len(self.interactions["user_id"].unique()))
        
        # Check that activity scores are normalized
        self.assertGreaterEqual(activity.min(), 0.0)
        self.assertLessEqual(activity.max(), 1.0)
    
    def test_filter_cold_start_users(self):
        """Test cold start user filtering."""
        filtered = filter_cold_start_users(self.interactions, min_interactions=2)
        
        self.assertIsInstance(filtered, pd.DataFrame)
        self.assertLessEqual(len(filtered), len(self.interactions))
    
    def test_filter_cold_start_items(self):
        """Test cold start item filtering."""
        filtered = filter_cold_start_items(self.interactions, min_interactions=2)
        
        self.assertIsInstance(filtered, pd.DataFrame)
        self.assertLessEqual(len(filtered), len(self.interactions))


if __name__ == "__main__":
    unittest.main()
