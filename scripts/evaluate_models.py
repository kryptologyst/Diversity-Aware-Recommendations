#!/usr/bin/env python3
"""Evaluate diversity-aware recommendation models."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.evaluation.metrics import RecommendationMetrics
from src.models.recommenders import BaseRecommender
from src.utils.helpers import (
    setup_logging,
    load_config,
    set_random_seeds,
    calculate_item_popularity,
    create_results_table,
    format_metrics
)

logger = logging.getLogger(__name__)


def evaluate_models(
    models: Dict[str, BaseRecommender],
    test_interactions: pd.DataFrame,
    items: pd.DataFrame,
    config: Dict
) -> Dict[str, Dict[str, float]]:
    """Evaluate all models on test data.
    
    Args:
        models: Dictionary of trained models.
        test_interactions: Test interactions DataFrame.
        items: Item metadata DataFrame.
        config: Configuration dictionary.
        
    Returns:
        Dictionary mapping model names to their evaluation metrics.
    """
    metrics_calculator = RecommendationMetrics(config)
    
    # Calculate item popularity for novelty and bias metrics
    item_popularity = calculate_item_popularity(test_interactions)
    
    # Get item similarity matrix from diversity-aware model if available
    item_similarity = None
    item_to_idx = None
    
    if "diversity_aware" in models:
        diversity_model = models["diversity_aware"]
        if hasattr(diversity_model, 'item_similarity'):
            item_similarity = diversity_model.item_similarity
            item_to_idx = {
                item_id: idx for idx, item_id in enumerate(diversity_model.base_recommender.user_item_matrix.columns)
            }
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name} model...")
        
        try:
            model_metrics = metrics_calculator.evaluate_model(
                model=model,
                test_interactions=test_interactions,
                item_popularity=item_popularity,
                item_similarity=item_similarity,
                item_to_idx=item_to_idx,
                n_recommendations=config["evaluation"]["num_recommendations"]
            )
            
            results[model_name] = model_metrics
            logger.info(f"{model_name} evaluation completed")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            continue
    
    return results


def print_results(results: Dict[str, Dict[str, float]]) -> None:
    """Print evaluation results in a formatted table.
    
    Args:
        results: Dictionary mapping model names to their metrics.
    """
    if not results:
        logger.warning("No results to display")
        return
    
    # Create results table
    results_df = create_results_table(results)
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(results_df.to_string())
    print("="*80)
    
    # Print best model for each metric
    print("\nBEST MODELS BY METRIC:")
    print("-" * 40)
    
    for metric in results_df.columns:
        if metric in results_df.columns:
            best_model = results_df[metric].idxmax()
            best_score = results_df.loc[best_model, metric]
            print(f"{metric:20s}: {best_model:15s} ({best_score:.4f})")


def save_results(
    results: Dict[str, Dict[str, float]], 
    output_path: Path
) -> None:
    """Save evaluation results to CSV file.
    
    Args:
        results: Dictionary mapping model names to their metrics.
        output_path: Path to save results CSV file.
    """
    if not results:
        logger.warning("No results to save")
        return
    
    results_df = create_results_table(results)
    results_df.to_csv(output_path)
    logger.info(f"Results saved to {output_path}")


def analyze_diversity(results: Dict[str, Dict[str, float]]) -> None:
    """Analyze diversity metrics across models.
    
    Args:
        results: Dictionary mapping model names to their metrics.
    """
    print("\n" + "="*60)
    print("DIVERSITY ANALYSIS")
    print("="*60)
    
    diversity_metrics = ["diversity", "novelty", "coverage", "popularity_bias"]
    
    for metric in diversity_metrics:
        if any(metric in model_results for model_results in results.values()):
            print(f"\n{metric.upper()}:")
            print("-" * 20)
            
            metric_values = {}
            for model_name, model_results in results.items():
                if metric in model_results:
                    metric_values[model_name] = model_results[metric]
            
            # Sort by metric value
            sorted_models = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
            
            for model_name, value in sorted_models:
                print(f"  {model_name:15s}: {value:.4f}")


def main():
    """Main function to evaluate models."""
    parser = argparse.ArgumentParser(description="Evaluate diversity-aware recommendation models")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration file")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--model-dir", type=str, default="models", help="Model directory")
    parser.add_argument("--output-file", type=str, default="results/evaluation_results.csv", help="Output file for results")
    parser.add_argument("--load-models", action="store_true", help="Load models from disk")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    
    # Load configuration
    config = load_config(args.config)
    set_random_seeds(config["data"]["random_seed"])
    
    # Setup paths
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    data_loader = DataLoader(config)
    
    # Load test data
    test_path = data_dir / "test.csv"
    if test_path.exists():
        test_interactions = pd.read_csv(test_path)
        logger.info(f"Loaded test data: {len(test_interactions)} interactions")
    else:
        logger.error(f"Test data not found at {test_path}")
        return
    
    # Load items
    items = data_loader.load_items()
    
    # Load models
    if args.load_models:
        logger.info("Loading models from disk...")
        try:
            import pickle
            models = {}
            
            for model_file in model_dir.glob("*_model.pkl"):
                model_name = model_file.stem.replace("_model", "")
                
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                    models[model_name] = model
                
                logger.info(f"Loaded {model_name} model")
            
            if not models:
                logger.error("No models found in model directory")
                return
                
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return
    else:
        # Train models (this would normally be done separately)
        logger.info("Training models for evaluation...")
        from scripts.train_models import train_models
        
        train_path = data_dir / "train.csv"
        if not train_path.exists():
            logger.error(f"Training data not found at {train_path}")
            return
        
        train_interactions = pd.read_csv(train_path)
        models = train_models(config, train_interactions, items)
    
    # Evaluate models
    logger.info("Evaluating models...")
    results = evaluate_models(models, test_interactions, items, config)
    
    if not results:
        logger.error("No models were evaluated successfully")
        return
    
    # Print results
    print_results(results)
    
    # Analyze diversity
    analyze_diversity(results)
    
    # Save results
    save_results(results, output_path)
    
    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main()
