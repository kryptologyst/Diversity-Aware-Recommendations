#!/usr/bin/env python3
"""Train diversity-aware recommendation models."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.models.recommenders import (
    PopularityRecommender,
    UserKNNRecommender,
    ItemKNNRecommender,
    DiversityAwareRecommender
)
from src.utils.helpers import (
    setup_logging, 
    load_config, 
    set_random_seeds,
    calculate_item_popularity,
    create_user_item_matrix
)

logger = logging.getLogger(__name__)


def train_models(
    config: Dict,
    train_interactions: pd.DataFrame,
    items: pd.DataFrame
) -> Dict[str, object]:
    """Train all configured recommendation models.
    
    Args:
        config: Configuration dictionary.
        train_interactions: Training interactions DataFrame.
        items: Item metadata DataFrame.
        
    Returns:
        Dictionary mapping model names to trained models.
    """
    models = {}
    
    # Train popularity model
    if config["models"]["popularity"]["enabled"]:
        logger.info("Training popularity model...")
        popularity_model = PopularityRecommender(config)
        popularity_model.fit(train_interactions, items)
        models["popularity"] = popularity_model
        logger.info("Popularity model trained")
    
    # Train user-kNN model
    if config["models"]["user_knn"]["enabled"]:
        logger.info("Training user-kNN model...")
        user_knn_model = UserKNNRecommender(config)
        user_knn_model.fit(train_interactions, items)
        models["user_knn"] = user_knn_model
        logger.info("User-kNN model trained")
    
    # Train item-kNN model
    if config["models"]["item_knn"]["enabled"]:
        logger.info("Training item-kNN model...")
        item_knn_model = ItemKNNRecommender(config)
        item_knn_model.fit(train_interactions, items)
        models["item_knn"] = item_knn_model
        logger.info("Item-kNN model trained")
    
    # Train diversity-aware model
    if config["models"]["diversity_aware"]["enabled"]:
        logger.info("Training diversity-aware model...")
        diversity_model = DiversityAwareRecommender(config)
        diversity_model.fit(train_interactions, items)
        models["diversity_aware"] = diversity_model
        logger.info("Diversity-aware model trained")
    
    return models


def save_models(models: Dict[str, object], output_dir: Path) -> None:
    """Save trained models to disk.
    
    Args:
        models: Dictionary of trained models.
        output_dir: Directory to save models.
    """
    output_dir.mkdir(exist_ok=True)
    
    for model_name, model in models.items():
        model_path = output_dir / f"{model_name}_model.pkl"
        
        try:
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {model_name} model to {model_path}")
        except Exception as e:
            logger.warning(f"Failed to save {model_name} model: {e}")


def load_models(model_dir: Path) -> Dict[str, object]:
    """Load trained models from disk.
    
    Args:
        model_dir: Directory containing saved models.
        
    Returns:
        Dictionary mapping model names to loaded models.
    """
    models = {}
    
    if not model_dir.exists():
        logger.warning(f"Model directory not found: {model_dir}")
        return models
    
    try:
        import pickle
        
        for model_file in model_dir.glob("*_model.pkl"):
            model_name = model_file.stem.replace("_model", "")
            
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
                models[model_name] = model
            
            logger.info(f"Loaded {model_name} model from {model_file}")
    
    except Exception as e:
        logger.warning(f"Failed to load models: {e}")
    
    return models


def main():
    """Main function to train models."""
    parser = argparse.ArgumentParser(description="Train diversity-aware recommendation models")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration file")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for models")
    parser.add_argument("--load-existing", action="store_true", help="Load existing models instead of training")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    
    # Load configuration
    config = load_config(args.config)
    set_random_seeds(config["data"]["random_seed"])
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if args.load_existing:
        # Load existing models
        logger.info("Loading existing models...")
        models = load_models(output_dir)
        
        if not models:
            logger.error("No existing models found. Run without --load-existing to train new models.")
            return
        
        logger.info(f"Loaded {len(models)} models: {list(models.keys())}")
        return
    
    # Load data
    logger.info("Loading data...")
    data_loader = DataLoader(config)
    
    interactions = data_loader.load_interactions()
    items = data_loader.load_items()
    
    # Filter data
    interactions = data_loader.filter_data(interactions)
    
    # Create train/val/test splits
    train, val, test = data_loader.create_train_val_test_split(interactions)
    
    logger.info(f"Data loaded - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Train models
    logger.info("Training models...")
    models = train_models(config, train, items)
    
    if not models:
        logger.error("No models were trained. Check configuration.")
        return
    
    # Save models
    logger.info("Saving models...")
    save_models(models, output_dir)
    
    # Save data splits for evaluation
    train.to_csv(data_dir / "train.csv", index=False)
    val.to_csv(data_dir / "val.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    
    logger.info(f"Training completed. {len(models)} models trained and saved.")
    logger.info(f"Models: {list(models.keys())}")


if __name__ == "__main__":
    main()
