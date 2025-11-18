"""Evaluation metrics for diversity-aware recommendations."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class RecommendationMetrics:
    """Comprehensive evaluation metrics for recommendation systems."""
    
    def __init__(self, config: Dict) -> None:
        """Initialize metrics calculator.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.k_values = config["evaluation"]["k_values"]
    
    def precision_at_k(
        self, 
        recommendations: List[int], 
        relevant_items: List[int], 
        k: int
    ) -> float:
        """Calculate Precision@K.
        
        Args:
            recommendations: List of recommended item IDs.
            relevant_items: List of relevant item IDs.
            k: Number of top recommendations to consider.
            
        Returns:
            Precision@K score.
        """
        if k == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_count = len(set(top_k_recs) & set(relevant_items))
        return relevant_count / k
    
    def recall_at_k(
        self, 
        recommendations: List[int], 
        relevant_items: List[int], 
        k: int
    ) -> float:
        """Calculate Recall@K.
        
        Args:
            recommendations: List of recommended item IDs.
            relevant_items: List of relevant item IDs.
            k: Number of top recommendations to consider.
            
        Returns:
            Recall@K score.
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_count = len(set(top_k_recs) & set(relevant_items))
        return relevant_count / len(relevant_items)
    
    def ndcg_at_k(
        self, 
        recommendations: List[int], 
        relevant_items: List[int], 
        k: int
    ) -> float:
        """Calculate NDCG@K.
        
        Args:
            recommendations: List of recommended item IDs.
            relevant_items: List of relevant item IDs.
            k: Number of top recommendations to consider.
            
        Returns:
            NDCG@K score.
        """
        if k == 0 or len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        
        # Create relevance scores (1 for relevant, 0 for irrelevant)
        relevance_scores = [1 if item in relevant_items else 0 for item in top_k_recs]
        
        # Calculate DCG
        dcg = sum(score / np.log2(idx + 2) for idx, score in enumerate(relevance_scores))
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = [1] * min(len(relevant_items), k)
        idcg = sum(score / np.log2(idx + 2) for idx, score in enumerate(ideal_relevance))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def hit_rate_at_k(
        self, 
        recommendations: List[int], 
        relevant_items: List[int], 
        k: int
    ) -> float:
        """Calculate Hit Rate@K.
        
        Args:
            recommendations: List of recommended item IDs.
            relevant_items: List of relevant item IDs.
            k: Number of top recommendations to consider.
            
        Returns:
            Hit Rate@K score (0 or 1).
        """
        top_k_recs = recommendations[:k]
        return 1.0 if len(set(top_k_recs) & set(relevant_items)) > 0 else 0.0
    
    def map_at_k(
        self, 
        recommendations: List[int], 
        relevant_items: List[int], 
        k: int
    ) -> float:
        """Calculate MAP@K.
        
        Args:
            recommendations: List of recommended item IDs.
            relevant_items: List of relevant item IDs.
            k: Number of top recommendations to consider.
            
        Returns:
            MAP@K score.
        """
        if k == 0 or len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(top_k_recs):
            if item in relevant_items:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        return precision_sum / len(relevant_items)
    
    def intra_list_diversity(
        self, 
        recommendations: List[int], 
        item_features: Optional[pd.DataFrame] = None,
        item_similarity: Optional[np.ndarray] = None,
        item_to_idx: Optional[Dict[int, int]] = None
    ) -> float:
        """Calculate intra-list diversity (ILD).
        
        Args:
            recommendations: List of recommended item IDs.
            item_features: DataFrame with item features.
            item_similarity: Precomputed item similarity matrix.
            item_to_idx: Mapping from item_id to matrix index.
            
        Returns:
            Average pairwise dissimilarity in the recommendation list.
        """
        if len(recommendations) < 2:
            return 0.0
        
        if item_similarity is not None and item_to_idx is not None:
            # Use precomputed similarity matrix
            similarities = []
            for i in range(len(recommendations)):
                for j in range(i + 1, len(recommendations)):
                    item_i_idx = item_to_idx.get(recommendations[i])
                    item_j_idx = item_to_idx.get(recommendations[j])
                    
                    if item_i_idx is not None and item_j_idx is not None:
                        similarity = item_similarity[item_i_idx, item_j_idx]
                        similarities.append(similarity)
            
            if similarities:
                return 1.0 - np.mean(similarities)
        
        elif item_features is not None:
            # Compute similarity from features
            rec_features = item_features[item_features.index.isin(recommendations)]
            if len(rec_features) < 2:
                return 0.0
            
            similarity_matrix = cosine_similarity(rec_features.values)
            # Get upper triangle (excluding diagonal)
            upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            
            if len(upper_triangle) > 0:
                return 1.0 - np.mean(upper_triangle)
        
        # Fallback: assume items are dissimilar
        return 1.0
    
    def novelty(
        self, 
        recommendations: List[int], 
        item_popularity: pd.Series
    ) -> float:
        """Calculate novelty of recommendations.
        
        Args:
            recommendations: List of recommended item IDs.
            item_popularity: Series with item popularity scores.
            
        Returns:
            Average novelty score (higher = more novel).
        """
        if not recommendations:
            return 0.0
        
        novelty_scores = []
        for item_id in recommendations:
            if item_id in item_popularity.index:
                # Novelty is inverse of popularity
                novelty_scores.append(1.0 - item_popularity[item_id])
            else:
                # Unknown items are considered novel
                novelty_scores.append(1.0)
        
        return np.mean(novelty_scores)
    
    def coverage(
        self, 
        all_recommendations: List[List[int]], 
        total_items: int
    ) -> float:
        """Calculate catalog coverage.
        
        Args:
            all_recommendations: List of recommendation lists for all users.
            total_items: Total number of items in catalog.
            
        Returns:
            Coverage score (fraction of items recommended).
        """
        if total_items == 0:
            return 0.0
        
        recommended_items = set()
        for recs in all_recommendations:
            recommended_items.update(recs)
        
        return len(recommended_items) / total_items
    
    def popularity_bias(
        self, 
        recommendations: List[int], 
        item_popularity: pd.Series
    ) -> float:
        """Calculate popularity bias in recommendations.
        
        Args:
            recommendations: List of recommended item IDs.
            item_popularity: Series with item popularity scores.
            
        Returns:
            Average popularity score of recommendations.
        """
        if not recommendations:
            return 0.0
        
        popularity_scores = []
        for item_id in recommendations:
            if item_id in item_popularity.index:
                popularity_scores.append(item_popularity[item_id])
            else:
                popularity_scores.append(0.0)
        
        return np.mean(popularity_scores)
    
    def evaluate_user(
        self, 
        user_id: int,
        recommendations: List[int],
        relevant_items: List[int],
        item_popularity: pd.Series,
        item_features: Optional[pd.DataFrame] = None,
        item_similarity: Optional[np.ndarray] = None,
        item_to_idx: Optional[Dict[int, int]] = None
    ) -> Dict[str, float]:
        """Evaluate recommendations for a single user.
        
        Args:
            user_id: User ID.
            recommendations: List of recommended item IDs.
            relevant_items: List of relevant item IDs.
            item_popularity: Series with item popularity scores.
            item_features: DataFrame with item features.
            item_similarity: Precomputed item similarity matrix.
            item_to_idx: Mapping from item_id to matrix index.
            
        Returns:
            Dictionary of metric scores.
        """
        metrics = {}
        
        # Relevance metrics
        for k in self.k_values:
            metrics[f"precision@{k}"] = self.precision_at_k(recommendations, relevant_items, k)
            metrics[f"recall@{k}"] = self.recall_at_k(recommendations, relevant_items, k)
            metrics[f"ndcg@{k}"] = self.ndcg_at_k(recommendations, relevant_items, k)
            metrics[f"hit_rate@{k}"] = self.hit_rate_at_k(recommendations, relevant_items, k)
            metrics[f"map@{k}"] = self.map_at_k(recommendations, relevant_items, k)
        
        # Diversity metrics
        metrics["diversity"] = self.intra_list_diversity(
            recommendations, item_features, item_similarity, item_to_idx
        )
        metrics["novelty"] = self.novelty(recommendations, item_popularity)
        metrics["popularity_bias"] = self.popularity_bias(recommendations, item_popularity)
        
        return metrics
    
    def evaluate_model(
        self,
        model,
        test_interactions: pd.DataFrame,
        item_popularity: pd.Series,
        item_features: Optional[pd.DataFrame] = None,
        item_similarity: Optional[np.ndarray] = None,
        item_to_idx: Optional[Dict[int, int]] = None,
        n_recommendations: int = 10
    ) -> Dict[str, float]:
        """Evaluate a recommendation model on test data.
        
        Args:
            model: Trained recommendation model.
            test_interactions: Test interactions DataFrame.
            item_popularity: Series with item popularity scores.
            item_features: DataFrame with item features.
            item_similarity: Precomputed item similarity matrix.
            item_to_idx: Mapping from item_id to matrix index.
            n_recommendations: Number of recommendations per user.
            
        Returns:
            Dictionary of average metric scores.
        """
        all_metrics = []
        all_recommendations = []
        
        # Group test interactions by user
        user_items = test_interactions.groupby("user_id")["item_id"].apply(list).to_dict()
        
        for user_id, relevant_items in user_items.items():
            try:
                # Generate recommendations
                recs_with_scores = model.recommend(user_id, n_recommendations)
                recommendations = [item_id for item_id, _ in recs_with_scores]
                all_recommendations.append(recommendations)
                
                # Evaluate user
                user_metrics = self.evaluate_user(
                    user_id, recommendations, relevant_items, item_popularity,
                    item_features, item_similarity, item_to_idx
                )
                all_metrics.append(user_metrics)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate user {user_id}: {e}")
                continue
        
        if not all_metrics:
            logger.warning("No users could be evaluated")
            return {}
        
        # Calculate average metrics
        avg_metrics = {}
        for metric_name in all_metrics[0].keys():
            values = [metrics[metric_name] for metrics in all_metrics]
            avg_metrics[metric_name] = np.mean(values)
        
        # Calculate coverage
        total_items = len(item_popularity)
        avg_metrics["coverage"] = self.coverage(all_recommendations, total_items)
        
        return avg_metrics
