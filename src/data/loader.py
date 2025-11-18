"""Data loading and preprocessing utilities for diversity-aware recommendations."""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and preprocessing of recommendation data."""
    
    def __init__(self, config: Dict) -> None:
        """Initialize data loader with configuration.
        
        Args:
            config: Configuration dictionary containing data paths and parameters.
        """
        self.config = config
        self.data_dir = Path(config["data"]["interactions_file"]).parent
        self.min_user_interactions = config["data"]["min_interactions_per_user"]
        self.min_item_interactions = config["data"]["min_interactions_per_item"]
        self.random_seed = config["data"]["random_seed"]
        
        # Set random seeds for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
    
    def load_interactions(self) -> pd.DataFrame:
        """Load user-item interactions from CSV file.
        
        Returns:
            DataFrame with columns: user_id, item_id, timestamp, weight
        """
        file_path = self.data_dir / "interactions.csv"
        
        if not file_path.exists():
            logger.warning(f"Interactions file not found at {file_path}")
            return self._generate_sample_interactions()
        
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_cols = ["user_id", "item_id", "timestamp", "weight"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Loaded {len(df)} interactions from {file_path}")
        return df
    
    def load_items(self) -> pd.DataFrame:
        """Load item metadata from CSV file.
        
        Returns:
            DataFrame with item information including title, tags, text, features.
        """
        file_path = self.data_dir / "items.csv"
        
        if not file_path.exists():
            logger.warning(f"Items file not found at {file_path}")
            return self._generate_sample_items()
        
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_cols = ["item_id", "title"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Loaded {len(df)} items from {file_path}")
        return df
    
    def load_users(self) -> Optional[pd.DataFrame]:
        """Load user metadata from CSV file.
        
        Returns:
            DataFrame with user information or None if file doesn't exist.
        """
        file_path = self.data_dir / "users.csv"
        
        if not file_path.exists():
            logger.info(f"Users file not found at {file_path}, skipping user metadata")
            return None
        
        df = pd.read_csv(file_path)
        
        # Validate required columns
        if "user_id" not in df.columns:
            raise ValueError("Missing required column: user_id")
        
        logger.info(f"Loaded {len(df)} users from {file_path}")
        return df
    
    def filter_data(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Filter interactions based on minimum thresholds.
        
        Args:
            interactions: Raw interactions DataFrame.
            
        Returns:
            Filtered interactions DataFrame.
        """
        initial_count = len(interactions)
        
        # Filter users with minimum interactions
        user_counts = interactions["user_id"].value_counts()
        valid_users = user_counts[user_counts >= self.min_user_interactions].index
        interactions = interactions[interactions["user_id"].isin(valid_users)]
        
        # Filter items with minimum interactions
        item_counts = interactions["item_id"].value_counts()
        valid_items = item_counts[item_counts >= self.min_item_interactions].index
        interactions = interactions[interactions["item_id"].isin(valid_items)]
        
        # Remove users/items that no longer meet thresholds after filtering
        while True:
            user_counts = interactions["user_id"].value_counts()
            item_counts = interactions["item_id"].value_counts()
            
            valid_users = user_counts[user_counts >= self.min_user_interactions].index
            valid_items = item_counts[item_counts >= self.min_item_interactions].index
            
            new_interactions = interactions[
                (interactions["user_id"].isin(valid_users)) &
                (interactions["item_id"].isin(valid_items))
            ]
            
            if len(new_interactions) == len(interactions):
                break
            interactions = new_interactions
        
        final_count = len(interactions)
        logger.info(f"Filtered interactions: {initial_count} -> {final_count}")
        
        return interactions
    
    def create_train_val_test_split(
        self, interactions: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create time-aware train/validation/test splits.
        
        Args:
            interactions: Filtered interactions DataFrame.
            
        Returns:
            Tuple of (train, validation, test) DataFrames.
        """
        # Sort by timestamp to ensure temporal ordering
        interactions = interactions.sort_values("timestamp")
        
        # Split by time: last 20% for test, previous 10% for validation
        test_size = self.config["data"]["test_size"]
        val_size = self.config["data"]["val_size"]
        
        # Calculate split indices
        total_size = len(interactions)
        test_start = int(total_size * (1 - test_size))
        val_start = int(total_size * (1 - test_size - val_size))
        
        train = interactions.iloc[:val_start].copy()
        val = interactions.iloc[val_start:test_start].copy()
        test = interactions.iloc[test_start:].copy()
        
        logger.info(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        return train, val, test
    
    def _generate_sample_interactions(self) -> pd.DataFrame:
        """Generate sample interaction data for demonstration."""
        logger.info("Generating sample interaction data")
        
        np.random.seed(self.random_seed)
        
        # Generate realistic interaction patterns
        n_users = 1000
        n_items = 500
        n_interactions = 10000
        
        # Create user-item pairs with some structure
        users = np.random.choice(range(n_users), n_interactions)
        items = np.random.choice(range(n_items), n_interactions)
        
        # Add some popularity bias
        popular_items = np.random.choice(range(n_items), size=50, replace=False)
        popular_mask = np.isin(items, popular_items)
        items[popular_mask] = np.random.choice(popular_items, np.sum(popular_mask))
        
        # Generate timestamps with some seasonality
        timestamps = np.random.randint(0, 365 * 24 * 60 * 60, n_interactions)  # Within a year
        timestamps = np.sort(timestamps)
        
        # Generate weights (ratings, clicks, etc.)
        weights = np.random.choice([1, 2, 3, 4, 5], n_interactions, p=[0.1, 0.2, 0.3, 0.3, 0.1])
        
        df = pd.DataFrame({
            "user_id": users,
            "item_id": items,
            "timestamp": timestamps,
            "weight": weights
        })
        
        # Remove duplicates
        df = df.drop_duplicates(subset=["user_id", "item_id"])
        
        # Save for future use
        self.data_dir.mkdir(exist_ok=True)
        df.to_csv(self.data_dir / "interactions.csv", index=False)
        
        logger.info(f"Generated and saved {len(df)} sample interactions")
        return df
    
    def _generate_sample_items(self) -> pd.DataFrame:
        """Generate sample item data for demonstration."""
        logger.info("Generating sample item data")
        
        np.random.seed(self.random_seed)
        
        # Get unique items from interactions
        interactions = self.load_interactions()
        unique_items = interactions["item_id"].unique()
        
        # Generate item metadata
        titles = [f"Item_{i}" for i in unique_items]
        
        # Generate tags
        tag_categories = ["action", "comedy", "drama", "sci-fi", "romance", "thriller", "horror", "documentary"]
        tags = [",".join(np.random.choice(tag_categories, size=np.random.randint(1, 4), replace=False)) 
                for _ in unique_items]
        
        # Generate descriptions
        descriptions = [f"Description for item {i}" for i in unique_items]
        
        df = pd.DataFrame({
            "item_id": unique_items,
            "title": titles,
            "tags": tags,
            "text": descriptions,
            "features": ["{}"] * len(unique_items)  # Empty JSON features
        })
        
        # Save for future use
        df.to_csv(self.data_dir / "items.csv", index=False)
        
        logger.info(f"Generated and saved {len(df)} sample items")
        return df
