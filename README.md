# Diversity-Aware Recommendations

Production-ready recommendation system that balances relevance with diversity to provide users with varied and novel suggestions.

## Overview

This project implements diversity-aware recommendation algorithms that go beyond traditional collaborative filtering to ensure users receive both relevant and diverse recommendations. The system addresses the common problem of recommendation systems becoming too narrow and repetitive.

## Key Features

- **Diversity Metrics**: Implements various diversity measures including intra-list diversity, novelty, and coverage
- **MMR Re-ranking**: Maximum Marginal Relevance algorithm for balancing relevance and diversity
- **Multiple Baselines**: Popularity, user-kNN, item-kNN, and matrix factorization models
- **Comprehensive Evaluation**: Precision@K, Recall@K, NDCG@K, plus diversity-specific metrics
- **Interactive Demo**: Streamlit interface for exploring recommendations
- **Production Ready**: Type hints, comprehensive testing, and clean architecture

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Sample Data**:
   ```bash
   python scripts/generate_data.py
   ```

3. **Train Models**:
   ```bash
   python scripts/train_models.py
   ```

4. **Run Evaluation**:
   ```bash
   python scripts/evaluate_models.py
   ```

5. **Launch Demo**:
   ```bash
   streamlit run demo/app.py
   ```

## Project Structure

```
├── src/                    # Source code modules
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Recommendation models
│   ├── evaluation/        # Evaluation metrics and utilities
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── data/                  # Data files (interactions, items, users)
├── notebooks/             # Jupyter notebooks for analysis
├── scripts/               # Training and evaluation scripts
├── tests/                 # Unit tests
├── demo/                  # Streamlit demo application
└── assets/                # Images and other assets
```

## Dataset Schema

### interactions.csv
- `user_id`: Unique user identifier
- `item_id`: Unique item identifier  
- `timestamp`: Interaction timestamp
- `weight`: Interaction strength (rating, click, etc.)

### items.csv
- `item_id`: Unique item identifier
- `title`: Item title/name
- `tags`: Comma-separated tags
- `text`: Item description
- `features`: Additional features (JSON format)

### users.csv (optional)
- `user_id`: Unique user identifier
- `features`: User features (JSON format)

## Models

### Baselines
- **Popularity**: Most popular items globally
- **User-kNN**: User-based collaborative filtering
- **Item-kNN**: Item-based collaborative filtering

### Advanced Models
- **Matrix Factorization**: ALS and BPR implementations
- **Diversity-Aware**: MMR re-ranking with various diversity metrics
- **Hybrid**: Combining collaborative filtering with content features

## Evaluation Metrics

### Relevance Metrics
- Precision@K
- Recall@K
- MAP@K
- NDCG@K
- Hit Rate@K

### Diversity Metrics
- Intra-list Diversity (ILD)
- Novelty
- Coverage
- Popularity Bias
- Calibration

## Configuration

Models and experiments are configured via YAML files in the `configs/` directory. See `configs/default.yaml` for the default configuration.

## Development

### Code Quality
- Type hints throughout
- Google-style docstrings
- Black formatting
- Ruff linting
- Pre-commit hooks

### Testing
```bash
pytest tests/
```

### Adding New Models
1. Create model class in `src/models/`
2. Implement required methods
3. Add configuration in `configs/`
4. Add tests in `tests/models/`

## License

MIT License - see LICENSE file for details.
# Diversity-Aware-Recommendations
