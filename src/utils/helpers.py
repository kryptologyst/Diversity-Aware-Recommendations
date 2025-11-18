"""Utility functions for diversity-aware recommendations."""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary.
        config_path: Path to save configuration file.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Saved configuration to {config_path}")


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    # Set TensorFlow seed if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    logger.info(f"Set random seeds to {seed}")


def create_user_item_matrix(
    interactions: pd.DataFrame,
    fill_value: float = 0.0
) -> pd.DataFrame:
    """Create user-item interaction matrix.
    
    Args:
        interactions: Interactions DataFrame with user_id, item_id, weight columns.
        fill_value: Value to fill missing interactions.
        
    Returns:
        User-item matrix with users as rows and items as columns.
    """
    matrix = interactions.pivot_table(
        index="user_id",
        columns="item_id", 
        values="weight",
        fill_value=fill_value
    )
    
    logger.info(f"Created user-item matrix: {matrix.shape}")
    return matrix


def get_user_interactions(
    interactions: pd.DataFrame,
    user_id: int
) -> List[int]:
    """Get items interacted with by a user.
    
    Args:
        interactions: Interactions DataFrame.
        user_id: User ID.
        
    Returns:
        List of item IDs the user has interacted with.
    """
    user_interactions = interactions[interactions["user_id"] == user_id]["item_id"].tolist()
    return user_interactions


def get_item_interactions(
    interactions: pd.DataFrame,
    item_id: int
) -> List[int]:
    """Get users who have interacted with an item.
    
    Args:
        interactions: Interactions DataFrame.
        item_id: Item ID.
        
    Returns:
        List of user IDs who have interacted with the item.
    """
    item_interactions = interactions[interactions["item_id"] == item_id]["user_id"].tolist()
    return item_interactions


def calculate_item_popularity(interactions: pd.DataFrame) -> pd.Series:
    """Calculate item popularity scores.
    
    Args:
        interactions: Interactions DataFrame.
        
    Returns:
        Series with item popularity scores (normalized).
    """
    popularity = interactions["item_id"].value_counts()
    popularity = popularity / popularity.max()  # Normalize to [0, 1]
    
    logger.info(f"Calculated popularity for {len(popularity)} items")
    return popularity


def calculate_user_activity(interactions: pd.DataFrame) -> pd.Series:
    """Calculate user activity scores.
    
    Args:
        interactions: Interactions DataFrame.
        
    Returns:
        Series with user activity scores (normalized).
    """
    activity = interactions["user_id"].value_counts()
    activity = activity / activity.max()  # Normalize to [0, 1]
    
    logger.info(f"Calculated activity for {len(activity)} users")
    return activity


def filter_cold_start_users(
    interactions: pd.DataFrame,
    min_interactions: int = 5
) -> pd.DataFrame:
    """Filter out cold start users with few interactions.
    
    Args:
        interactions: Interactions DataFrame.
        min_interactions: Minimum number of interactions required.
        
    Returns:
        Filtered interactions DataFrame.
    """
    user_counts = interactions["user_id"].value_counts()
    active_users = user_counts[user_counts >= min_interactions].index
    
    filtered_interactions = interactions[interactions["user_id"].isin(active_users)]
    
    logger.info(f"Filtered cold start users: {len(interactions)} -> {len(filtered_interactions)} interactions")
    return filtered_interactions


def filter_cold_start_items(
    interactions: pd.DataFrame,
    min_interactions: int = 5
) -> pd.DataFrame:
    """Filter out cold start items with few interactions.
    
    Args:
        interactions: Interactions DataFrame.
        min_interactions: Minimum number of interactions required.
        
    Returns:
        Filtered interactions DataFrame.
    """
    item_counts = interactions["item_id"].value_counts()
    popular_items = item_counts[item_counts >= min_interactions].index
    
    filtered_interactions = interactions[interactions["item_id"].isin(popular_items)]
    
    logger.info(f"Filtered cold start items: {len(interactions)} -> {len(filtered_interactions)} interactions")
    return filtered_interactions


def create_negative_samples(
    interactions: pd.DataFrame,
    n_negative_per_user: int = 100,
    random_seed: int = 42
) -> pd.DataFrame:
    """Create negative samples for implicit feedback.
    
    Args:
        interactions: Positive interactions DataFrame.
        n_negative_per_user: Number of negative samples per user.
        random_seed: Random seed for reproducibility.
        
    Returns:
        DataFrame with negative samples.
    """
    np.random.seed(random_seed)
    
    # Get all users and items
    all_users = interactions["user_id"].unique()
    all_items = interactions["item_id"].unique()
    
    # Get user-item pairs that exist
    existing_pairs = set(zip(interactions["user_id"], interactions["item_id"]))
    
    negative_samples = []
    
    for user_id in all_users:
        # Get items user has interacted with
        user_items = set(interactions[interactions["user_id"] == user_id]["item_id"])
        
        # Sample negative items
        available_items = set(all_items) - user_items
        n_samples = min(n_negative_per_user, len(available_items))
        
        if n_samples > 0:
            negative_items = np.random.choice(
                list(available_items), 
                size=n_samples, 
                replace=False
            )
            
            for item_id in negative_items:
                negative_samples.append({
                    "user_id": user_id,
                    "item_id": item_id,
                    "weight": 0,  # Negative sample
                    "timestamp": np.random.randint(0, 1000000)  # Random timestamp
                })
    
    negative_df = pd.DataFrame(negative_samples)
    
    logger.info(f"Created {len(negative_df)} negative samples")
    return negative_df


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """Format metrics dictionary for display.
    
    Args:
        metrics: Dictionary of metric scores.
        precision: Number of decimal places.
        
    Returns:
        Formatted string representation of metrics.
    """
    lines = []
    for metric_name, value in sorted(metrics.items()):
        lines.append(f"{metric_name}: {value:.{precision}f}")
    
    return "\n".join(lines)


def create_results_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Create a results table from multiple model evaluations.
    
    Args:
        results: Dictionary mapping model names to their metrics.
        
    Returns:
        DataFrame with results table.
    """
    df = pd.DataFrame(results).T
    df = df.round(4)
    
    # Sort columns by metric type
    metric_order = []
    for k in [5, 10, 20]:
        for metric in ["precision", "recall", "ndcg", "hit_rate", "map"]:
            col_name = f"{metric}@{k}"
            if col_name in df.columns:
                metric_order.append(col_name)
    
    # Add diversity metrics
    diversity_metrics = ["diversity", "novelty", "coverage", "popularity_bias"]
    for metric in diversity_metrics:
        if metric in df.columns:
            metric_order.append(metric)
    
    # Reorder columns
    df = df[metric_order]
    
    return df
