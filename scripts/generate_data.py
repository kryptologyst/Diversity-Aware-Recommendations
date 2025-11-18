#!/usr/bin/env python3
"""Generate sample data for diversity-aware recommendations."""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.utils.helpers import setup_logging, load_config, set_random_seeds

logger = logging.getLogger(__name__)


def generate_realistic_interactions(
    n_users: int = 1000,
    n_items: int = 500,
    n_interactions: int = 20000,
    random_seed: int = 42
) -> pd.DataFrame:
    """Generate realistic interaction data with diversity patterns.
    
    Args:
        n_users: Number of users.
        n_items: Number of items.
        n_interactions: Number of interactions to generate.
        random_seed: Random seed for reproducibility.
        
    Returns:
        DataFrame with interactions.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    logger.info(f"Generating {n_interactions} interactions for {n_users} users and {n_items} items")
    
    # Create user segments with different preferences
    user_segments = {
        "mainstream": int(n_users * 0.6),  # 60% mainstream users
        "niche": int(n_users * 0.3),      # 30% niche users  
        "explorer": int(n_users * 0.1)     # 10% explorer users
    }
    
    # Create item categories with different popularity
    item_categories = {
        "popular": int(n_items * 0.2),    # 20% popular items
        "moderate": int(n_items * 0.5),    # 50% moderate items
        "niche": int(n_items * 0.3)        # 30% niche items
    }
    
    # Generate user IDs
    users = list(range(n_users))
    
    # Generate item IDs with categories
    items = []
    item_categories_map = {}
    
    item_id = 0
    for category, count in item_categories.items():
        for _ in range(count):
            items.append(item_id)
            item_categories_map[item_id] = category
            item_id += 1
    
    # Generate interactions based on user segments and item categories
    interactions = []
    
    for _ in range(n_interactions):
        user_id = random.choice(users)
        
        # Determine user segment
        if user_id < user_segments["mainstream"]:
            user_segment = "mainstream"
        elif user_id < user_segments["mainstream"] + user_segments["niche"]:
            user_segment = "niche"
        else:
            user_segment = "explorer"
        
        # Choose item based on user segment
        if user_segment == "mainstream":
            # Prefer popular items
            weights = [0.7, 0.2, 0.1]  # popular, moderate, niche
        elif user_segment == "niche":
            # Prefer niche items
            weights = [0.1, 0.3, 0.6]  # popular, moderate, niche
        else:  # explorer
            # More balanced preferences
            weights = [0.3, 0.4, 0.3]  # popular, moderate, niche
        
        # Sample item category
        category = np.random.choice(
            ["popular", "moderate", "niche"], 
            p=weights
        )
        
        # Choose item from category
        category_items = [item for item, cat in item_categories_map.items() if cat == category]
        item_id = random.choice(category_items)
        
        # Generate interaction weight (rating, click strength, etc.)
        if category == "popular":
            weight = np.random.choice([3, 4, 5], p=[0.2, 0.3, 0.5])
        elif category == "moderate":
            weight = np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3])
        else:  # niche
            weight = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        
        # Generate timestamp (within last year)
        timestamp = random.randint(0, 365 * 24 * 60 * 60)
        
        interactions.append({
            "user_id": user_id,
            "item_id": item_id,
            "weight": weight,
            "timestamp": timestamp
        })
    
    # Convert to DataFrame and sort by timestamp
    df = pd.DataFrame(interactions)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="first")
    
    logger.info(f"Generated {len(df)} unique interactions")
    return df


def generate_item_metadata(items: list, random_seed: int = 42) -> pd.DataFrame:
    """Generate realistic item metadata.
    
    Args:
        items: List of item IDs.
        random_seed: Random seed for reproducibility.
        
    Returns:
        DataFrame with item metadata.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    logger.info(f"Generating metadata for {len(items)} items")
    
    # Define item categories and their characteristics
    categories = {
        "movies": {
            "titles": ["The Dark Knight", "Inception", "Pulp Fiction", "Forrest Gump", "The Matrix"],
            "tags": ["action", "drama", "thriller", "sci-fi", "comedy", "romance", "horror", "documentary"],
            "descriptions": [
                "A gripping tale of heroism and sacrifice",
                "An innovative story that challenges reality",
                "A masterpiece of storytelling and cinematography",
                "A heartwarming journey through life",
                "A groundbreaking film that redefined cinema"
            ]
        },
        "books": {
            "titles": ["1984", "To Kill a Mockingbird", "Pride and Prejudice", "The Great Gatsby", "Dune"],
            "tags": ["fiction", "non-fiction", "mystery", "romance", "sci-fi", "biography", "history", "philosophy"],
            "descriptions": [
                "A timeless classic that explores human nature",
                "A profound work that challenges conventional thinking",
                "An engaging story with memorable characters",
                "A masterpiece of literature and social commentary",
                "An epic tale that spans generations"
            ]
        },
        "music": {
            "titles": ["Bohemian Rhapsody", "Hotel California", "Imagine", "Stairway to Heaven", "Billie Jean"],
            "tags": ["rock", "pop", "jazz", "classical", "electronic", "folk", "blues", "country"],
            "descriptions": [
                "An iconic song that defined a generation",
                "A musical masterpiece with timeless appeal",
                "A powerful anthem that resonates with listeners",
                "A classic track that showcases musical excellence",
                "An innovative piece that pushed creative boundaries"
            ]
        }
    }
    
    # Choose category for each item
    item_metadata = []
    
    for item_id in items:
        # Randomly assign category
        category = random.choice(list(categories.keys()))
        category_data = categories[category]
        
        # Generate title (mix of predefined and generated)
        if random.random() < 0.3:  # 30% predefined titles
            title = random.choice(category_data["titles"])
        else:  # 70% generated titles
            title = f"{category.title()} Item {item_id}"
        
        # Generate tags (2-4 tags per item)
        n_tags = random.randint(2, 4)
        tags = random.sample(category_data["tags"], n_tags)
        tags_str = ",".join(tags)
        
        # Generate description
        description = random.choice(category_data["descriptions"])
        
        # Generate additional features (JSON format)
        features = {
            "category": category,
            "year": random.randint(1990, 2023),
            "rating": round(random.uniform(3.0, 5.0), 1),
            "price": round(random.uniform(5.0, 50.0), 2)
        }
        
        item_metadata.append({
            "item_id": item_id,
            "title": title,
            "tags": tags_str,
            "text": description,
            "features": str(features).replace("'", '"')  # Convert to JSON-like string
        })
    
    df = pd.DataFrame(item_metadata)
    logger.info(f"Generated metadata for {len(df)} items")
    return df


def generate_user_metadata(users: list, random_seed: int = 42) -> pd.DataFrame:
    """Generate realistic user metadata.
    
    Args:
        users: List of user IDs.
        random_seed: Random seed for reproducibility.
        
    Returns:
        DataFrame with user metadata.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    logger.info(f"Generating metadata for {len(users)} users")
    
    user_metadata = []
    
    for user_id in users:
        # Generate user features
        features = {
            "age_group": random.choice(["18-25", "26-35", "36-45", "46-55", "55+"]),
            "gender": random.choice(["M", "F", "Other"]),
            "location": random.choice(["US", "UK", "CA", "AU", "DE", "FR", "JP", "Other"]),
            "preferred_categories": random.sample(
                ["movies", "books", "music"], 
                random.randint(1, 3)
            ),
            "activity_level": random.choice(["low", "medium", "high"])
        }
        
        user_metadata.append({
            "user_id": user_id,
            "features": str(features).replace("'", '"')  # Convert to JSON-like string
        })
    
    df = pd.DataFrame(user_metadata)
    logger.info(f"Generated metadata for {len(df)} users")
    return df


def main():
    """Main function to generate sample data."""
    parser = argparse.ArgumentParser(description="Generate sample data for diversity-aware recommendations")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration file")
    parser.add_argument("--n-users", type=int, default=1000, help="Number of users")
    parser.add_argument("--n-items", type=int, default=500, help="Number of items")
    parser.add_argument("--n-interactions", type=int, default=20000, help="Number of interactions")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    set_random_seeds(args.seed)
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        config = {
            "data": {
                "min_interactions_per_user": 5,
                "min_interactions_per_item": 5,
                "random_seed": args.seed
            }
        }
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Starting data generation...")
    
    # Generate interactions
    interactions = generate_realistic_interactions(
        n_users=args.n_users,
        n_items=args.n_items,
        n_interactions=args.n_interactions,
        random_seed=args.seed
    )
    
    # Generate item metadata
    unique_items = sorted(interactions["item_id"].unique())
    items = generate_item_metadata(unique_items, random_seed=args.seed)
    
    # Generate user metadata
    unique_users = sorted(interactions["user_id"].unique())
    users = generate_user_metadata(unique_users, random_seed=args.seed)
    
    # Save data
    interactions.to_csv(output_dir / "interactions.csv", index=False)
    items.to_csv(output_dir / "items.csv", index=False)
    users.to_csv(output_dir / "users.csv", index=False)
    
    logger.info(f"Data saved to {output_dir}")
    logger.info(f"Generated {len(interactions)} interactions, {len(items)} items, {len(users)} users")
    
    # Print data statistics
    print("\nData Statistics:")
    print(f"Interactions: {len(interactions)}")
    print(f"Users: {len(unique_users)}")
    print(f"Items: {len(unique_items)}")
    print(f"Avg interactions per user: {len(interactions) / len(unique_users):.2f}")
    print(f"Avg interactions per item: {len(interactions) / len(unique_items):.2f}")
    print(f"Sparsity: {1 - len(interactions) / (len(unique_users) * len(unique_items)):.4f}")


if __name__ == "__main__":
    main()
