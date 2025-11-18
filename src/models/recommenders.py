"""Diversity-aware recommendation models."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class BaseRecommender(ABC):
    """Abstract base class for recommendation models."""
    
    def __init__(self, config: Dict) -> None:
        """Initialize the recommender with configuration.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, interactions: pd.DataFrame, items: pd.DataFrame) -> None:
        """Fit the recommendation model.
        
        Args:
            interactions: User-item interactions DataFrame.
            items: Item metadata DataFrame.
        """
        pass
    
    @abstractmethod
    def recommend(
        self, 
        user_id: int, 
        n_recommendations: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User ID to generate recommendations for.
            n_recommendations: Number of recommendations to generate.
            exclude_items: Items to exclude from recommendations.
            
        Returns:
            List of (item_id, score) tuples.
        """
        pass
    
    def get_item_similarity(self, item_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """Get similar items for a given item.
        
        Args:
            item_id: Item ID to find similar items for.
            n_similar: Number of similar items to return.
            
        Returns:
            List of (item_id, similarity_score) tuples.
        """
        raise NotImplementedError("Item similarity not implemented for this model")


class PopularityRecommender(BaseRecommender):
    """Popularity-based recommender."""
    
    def fit(self, interactions: pd.DataFrame, items: pd.DataFrame) -> None:
        """Fit the popularity model.
        
        Args:
            interactions: User-item interactions DataFrame.
            items: Item metadata DataFrame.
        """
        self.item_popularity = interactions["item_id"].value_counts()
        self.is_fitted = True
        logger.info("Fitted popularity recommender")
    
    def recommend(
        self, 
        user_id: int, 
        n_recommendations: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """Generate popularity-based recommendations.
        
        Args:
            user_id: User ID (not used for popularity).
            n_recommendations: Number of recommendations to generate.
            exclude_items: Items to exclude from recommendations.
            
        Returns:
            List of (item_id, popularity_score) tuples.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        recommendations = self.item_popularity.head(n_recommendations)
        
        if exclude_items:
            recommendations = recommendations[~recommendations.index.isin(exclude_items)]
        
        return [(item_id, score) for item_id, score in recommendations.items()]


class UserKNNRecommender(BaseRecommender):
    """User-based collaborative filtering with k-nearest neighbors."""
    
    def __init__(self, config: Dict) -> None:
        """Initialize user-kNN recommender.
        
        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)
        self.k = config["models"]["user_knn"]["k"]
        self.similarity = config["models"]["user_knn"]["similarity"]
        self.user_item_matrix = None
        self.user_similarity = None
        self.user_knn = None
    
    def fit(self, interactions: pd.DataFrame, items: pd.DataFrame) -> None:
        """Fit the user-kNN model.
        
        Args:
            interactions: User-item interactions DataFrame.
            items: Item metadata DataFrame.
        """
        # Create user-item matrix
        self.user_item_matrix = interactions.pivot_table(
            index="user_id", 
            columns="item_id", 
            values="weight", 
            fill_value=0
        )
        
        # Compute user similarity
        if self.similarity == "cosine":
            self.user_similarity = cosine_similarity(self.user_item_matrix.values)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity}")
        
        # Fit k-NN model
        self.user_knn = NearestNeighbors(
            n_neighbors=self.k + 1,  # +1 to exclude self
            metric="cosine"
        )
        self.user_knn.fit(self.user_item_matrix.values)
        
        self.is_fitted = True
        logger.info(f"Fitted user-kNN recommender with k={self.k}")
    
    def recommend(
        self, 
        user_id: int, 
        n_recommendations: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """Generate user-kNN recommendations.
        
        Args:
            user_id: User ID to generate recommendations for.
            n_recommendations: Number of recommendations to generate.
            exclude_items: Items to exclude from recommendations.
            
        Returns:
            List of (item_id, predicted_score) tuples.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id not in self.user_item_matrix.index:
            # Cold start: return popular items
            popularity_rec = PopularityRecommender(self.config)
            popularity_rec.fit(
                pd.DataFrame({"item_id": self.user_item_matrix.columns}), 
                pd.DataFrame()
            )
            return popularity_rec.recommend(user_id, n_recommendations, exclude_items)
        
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_vector = self.user_item_matrix.iloc[user_idx].values
        
        # Find similar users
        distances, indices = self.user_knn.kneighbors([user_vector])
        similar_users = indices[0][1:]  # Exclude self
        similarities = 1 - distances[0][1:]  # Convert distance to similarity
        
        # Predict ratings for unrated items
        predictions = {}
        for item_idx, item_id in enumerate(self.user_item_matrix.columns):
            if user_vector[item_idx] == 0:  # Unrated item
                # Weighted average of similar users' ratings
                weighted_sum = 0
                similarity_sum = 0
                
                for sim_user_idx, similarity in zip(similar_users, similarities):
                    sim_user_rating = self.user_item_matrix.iloc[sim_user_idx, item_idx]
                    if sim_user_rating > 0:
                        weighted_sum += similarity * sim_user_rating
                        similarity_sum += similarity
                
                if similarity_sum > 0:
                    predictions[item_id] = weighted_sum / similarity_sum
        
        # Sort by predicted score
        recommendations = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        if exclude_items:
            recommendations = [(item_id, score) for item_id, score in recommendations 
                             if item_id not in exclude_items]
        
        return recommendations[:n_recommendations]


class ItemKNNRecommender(BaseRecommender):
    """Item-based collaborative filtering with k-nearest neighbors."""
    
    def __init__(self, config: Dict) -> None:
        """Initialize item-kNN recommender.
        
        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)
        self.k = config["models"]["item_knn"]["k"]
        self.similarity = config["models"]["item_knn"]["similarity"]
        self.user_item_matrix = None
        self.item_similarity = None
        self.item_knn = None
    
    def fit(self, interactions: pd.DataFrame, items: pd.DataFrame) -> None:
        """Fit the item-kNN model.
        
        Args:
            interactions: User-item interactions DataFrame.
            items: Item metadata DataFrame.
        """
        # Create user-item matrix
        self.user_item_matrix = interactions.pivot_table(
            index="user_id", 
            columns="item_id", 
            values="weight", 
            fill_value=0
        )
        
        # Compute item similarity
        if self.similarity == "cosine":
            self.item_similarity = cosine_similarity(self.user_item_matrix.T.values)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity}")
        
        # Fit k-NN model
        self.item_knn = NearestNeighbors(
            n_neighbors=self.k + 1,  # +1 to exclude self
            metric="cosine"
        )
        self.item_knn.fit(self.user_item_matrix.T.values)
        
        self.is_fitted = True
        logger.info(f"Fitted item-kNN recommender with k={self.k}")
    
    def recommend(
        self, 
        user_id: int, 
        n_recommendations: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """Generate item-kNN recommendations.
        
        Args:
            user_id: User ID to generate recommendations for.
            n_recommendations: Number of recommendations to generate.
            exclude_items: Items to exclude from recommendations.
            
        Returns:
            List of (item_id, predicted_score) tuples.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id not in self.user_item_matrix.index:
            # Cold start: return popular items
            popularity_rec = PopularityRecommender(self.config)
            popularity_rec.fit(
                pd.DataFrame({"item_id": self.user_item_matrix.columns}), 
                pd.DataFrame()
            )
            return popularity_rec.recommend(user_id, n_recommendations, exclude_items)
        
        user_vector = self.user_item_matrix.loc[user_id].values
        user_items = np.where(user_vector > 0)[0]  # Items the user has rated
        
        if len(user_items) == 0:
            # No rated items: return popular items
            popularity_rec = PopularityRecommender(self.config)
            popularity_rec.fit(
                pd.DataFrame({"item_id": self.user_item_matrix.columns}), 
                pd.DataFrame()
            )
            return popularity_rec.recommend(user_id, n_recommendations, exclude_items)
        
        # Predict ratings for unrated items
        predictions = {}
        for item_idx, item_id in enumerate(self.user_item_matrix.columns):
            if user_vector[item_idx] == 0:  # Unrated item
                # Find similar items that user has rated
                item_vector = self.user_item_matrix.T.iloc[item_idx].values
                distances, indices = self.item_knn.kneighbors([item_vector])
                similar_items = indices[0][1:]  # Exclude self
                similarities = 1 - distances[0][1:]  # Convert distance to similarity
                
                # Weighted average of similar items' ratings
                weighted_sum = 0
                similarity_sum = 0
                
                for sim_item_idx, similarity in zip(similar_items, similarities):
                    if sim_item_idx in user_items:
                        user_rating = user_vector[sim_item_idx]
                        weighted_sum += similarity * user_rating
                        similarity_sum += similarity
                
                if similarity_sum > 0:
                    predictions[item_id] = weighted_sum / similarity_sum
        
        # Sort by predicted score
        recommendations = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        if exclude_items:
            recommendations = [(item_id, score) for item_id, score in recommendations 
                             if item_id not in exclude_items]
        
        return recommendations[:n_recommendations]
    
    def get_item_similarity(self, item_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """Get similar items for a given item.
        
        Args:
            item_id: Item ID to find similar items for.
            n_similar: Number of similar items to return.
            
        Returns:
            List of (item_id, similarity_score) tuples.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing similarities")
        
        if item_id not in self.user_item_matrix.columns:
            return []
        
        item_idx = self.user_item_matrix.columns.get_loc(item_id)
        similarities = self.item_similarity[item_idx]
        
        # Get top similar items (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
        similar_items = [(self.user_item_matrix.columns[idx], similarities[idx]) 
                        for idx in similar_indices]
        
        return similar_items


class DiversityAwareRecommender(BaseRecommender):
    """Diversity-aware recommender using MMR (Maximum Marginal Relevance)."""
    
    def __init__(self, config: Dict) -> None:
        """Initialize diversity-aware recommender.
        
        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)
        self.diversity_factor = config["models"]["diversity_aware"]["diversity_factor"]
        self.mmr_lambda = config["models"]["diversity_aware"]["mmr_lambda"]
        self.novelty_weight = config["models"]["diversity_aware"]["novelty_weight"]
        self.base_recommender = None
        self.item_popularity = None
        self.item_similarity = None
    
    def fit(self, interactions: pd.DataFrame, items: pd.DataFrame) -> None:
        """Fit the diversity-aware model.
        
        Args:
            interactions: User-item interactions DataFrame.
            items: Item metadata DataFrame.
        """
        # Use item-kNN as base recommender
        self.base_recommender = ItemKNNRecommender(self.config)
        self.base_recommender.fit(interactions, items)
        
        # Compute item popularity for novelty
        self.item_popularity = interactions["item_id"].value_counts()
        self.item_popularity = self.item_popularity / self.item_popularity.max()  # Normalize
        
        # Get item similarity matrix from base recommender
        self.item_similarity = self.base_recommender.item_similarity
        
        self.is_fitted = True
        logger.info("Fitted diversity-aware recommender")
    
    def recommend(
        self, 
        user_id: int, 
        n_recommendations: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """Generate diversity-aware recommendations using MMR.
        
        Args:
            user_id: User ID to generate recommendations for.
            n_recommendations: Number of recommendations to generate.
            exclude_items: Items to exclude from recommendations.
            
        Returns:
            List of (item_id, mmr_score) tuples.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get initial recommendations from base model
        initial_recs = self.base_recommender.recommend(
            user_id, 
            n_recommendations * 3,  # Get more candidates for diversity selection
            exclude_items
        )
        
        if not initial_recs:
            return []
        
        # Apply MMR re-ranking
        selected_items = []
        remaining_items = initial_recs.copy()
        
        # Select first item (highest relevance)
        if remaining_items:
            selected_items.append(remaining_items[0])
            remaining_items = remaining_items[1:]
        
        # Apply MMR for remaining items
        while len(selected_items) < n_recommendations and remaining_items:
            best_item = None
            best_score = -float('inf')
            best_idx = -1
            
            for idx, (item_id, relevance_score) in enumerate(remaining_items):
                # Compute diversity penalty
                diversity_penalty = 0
                if selected_items:
                    similarities = []
                    for selected_item_id, _ in selected_items:
                        if item_id in self.base_recommender.user_item_matrix.columns:
                            item_idx = self.base_recommender.user_item_matrix.columns.get_loc(item_id)
                            selected_idx = self.base_recommender.user_item_matrix.columns.get_loc(selected_item_id)
                            similarity = self.item_similarity[item_idx, selected_idx]
                            similarities.append(similarity)
                    
                    if similarities:
                        diversity_penalty = max(similarities)
                
                # Compute novelty bonus
                novelty_bonus = 0
                if item_id in self.item_popularity.index:
                    novelty_bonus = 1 - self.item_popularity[item_id]  # Lower popularity = higher novelty
                
                # MMR score
                mmr_score = (
                    self.mmr_lambda * relevance_score - 
                    (1 - self.mmr_lambda) * diversity_penalty +
                    self.novelty_weight * novelty_bonus
                )
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_item = (item_id, mmr_score)
                    best_idx = idx
            
            if best_item:
                selected_items.append(best_item)
                remaining_items.pop(best_idx)
            else:
                break
        
        return selected_items
    
    def get_item_similarity(self, item_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """Get similar items for a given item.
        
        Args:
            item_id: Item ID to find similar items for.
            n_similar: Number of similar items to return.
            
        Returns:
            List of (item_id, similarity_score) tuples.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing similarities")
        
        return self.base_recommender.get_item_similarity(item_id, n_similar)
