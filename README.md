# BerlinRentML

**Spatially Robust Berlin Rental Price Prediction**

A production-grade end-to-end machine learning system for predicting monthly cold rent (Kaltmiete) for apartments in Berlin, with explicit focus on geographic generalization and leakage prevention.

---

## 🎯 Project Overview

This project demonstrates serious ML engineering practices:

- **Real dataset** from ImmobilienScout24 (Germany's largest real estate platform)
- **Rigorous evaluation** with multiple split strategies (random, grouped, spatial)
- **Leakage prevention** as a first-class requirement
- **Geographic generalization** testing
- **Production-ready API** with FastAPI
- **Comprehensive testing** suite
- **Docker containerization**
- **CI/CD** with GitHub Actions

### Why This Matters

Berlin rental listings are geographically clustered. A naive random train/test split can produce deceptively optimistic results because nearby apartments appear in both sets. This project explicitly addresses spatial leakage and tests whether the model can generalize to unseen geographic regions.

---

## 📊 Dataset

**Source**: [Apartment Rental Offers in Germany](https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany) (Kaggle)

- **Platform**: ImmobilienScout24
- **Scope**: ~250,000+ listings across Germany (filtered to Berlin)
- **Target**: `baseRent` (monthly cold rent in €)
- **Features**: Property characteristics, amenities, location, construction year

See [data/DATASET.md](data/DATASET.md) for details.

---

## 🏗️ Architecture

```
BerlinRentML/
├── data/
│   ├── raw/              # Raw dataset (download via script)
│   └── processed/        # Cleaned and processed data
├── src/berlinrentml/
│   ├── data/             # Data loading, validation, cleaning
│   ├── features/         # Feature engineering, preprocessing
│   ├── modeling/         # Training, evaluation, splitting strategies
│   └── api/              # FastAPI inference service
├── scripts/
│   ├── download_data.py  # Download dataset
│   ├── train.py          # Main training pipeline
│   ├── analyze_errors.py # Error analysis
│   └── explain_model.py  # SHAP explainability
├── tests/                # Unit and integration tests
├── models/               # Saved model artifacts
└── Dockerfile            # API containerization
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd BerlinRentML

# Install dependencies
pip install -e .
pip install -e ".[dev]"  # For development
```

### 2. Download Dataset

```bash
# Option A: Using Kaggle CLI (recommended)
pip install kaggle
# Configure Kaggle API token (see data/DATASET.md)
python scripts/download_data.py

# Option B: Manual download
# 1. Visit https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany
# 2. Download CSV
# 3. Place in data/raw/
```

### 3. Train Model

```bash
python scripts/train.py
```

This will:
- Load and validate data
- Clean and engineer features
- Train multiple models (baselines, linear, Random Forest, LightGBM)
- Evaluate on random AND grouped splits
- Select best model based on geographic generalization
- Save model artifacts to `models/`

### 4. Run API

```bash
python src/berlinrentml/api/main.py
```

API will be available at `http://localhost:8000`

Documentation: `http://localhost:8000/docs`

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## 📈 Evaluation Methodology

### Split Strategies

We implement and compare multiple evaluation strategies:

#### 1. **Random Split** (Baseline)
Standard 80/20 train/test split. Shows what most projects report but may be optimistic.

#### 2. **Grouped Split** (Geographic Awareness)
Groups by postal code before splitting. Ensures no postal code appears in both train and test.

#### 3. **Spatial Holdout** (Geographic Generalization)
Holds out entire districts for testing. Tests whether the model can predict in completely unseen regions.

#### 4. **Temporal Split** (If dates available)
Trains on older listings, tests on newer ones.

### Why This Matters

If Random Split gives dramatically better results than Grouped/Spatial splits, it suggests the model is memorizing location-specific patterns rather than learning transferable relationships between features and rent.

---

## 🛡️ Leakage Prevention

Leakage is treated as a critical ML requirement:

### Investigated Leakage Sources

- ✅ **Target leakage**: No target-derived features
- ✅ **Warm-rent leakage**: Using `baseRent` (cold rent), not `totalRent`
- ✅ **Preprocessing leakage**: All transformers fitted on training data only
- ✅ **Duplicate leakage**: Duplicates removed
- ✅ **Geographic leakage**: Location encoded carefully to avoid overfitting
- ✅ **Temporal leakage**: No future information in features

### Preprocessing Pipeline

```python
from berlinrentml.features.preprocessing import LeakageSafePreprocessor

# Fit on training data ONLY
preprocessor = LeakageSafePreprocessor(numeric_features, categorical_features)
X_train_processed = preprocessor.fit_transform(X_train)

# Apply to test data (using training statistics)
X_test_processed = preprocessor.transform(X_test)
```

---

## 🎯 Model Selection

Models evaluated:
- Mean/Median predictors (baselines)
- Linear regression
- Ridge regression
- Random Forest
- LightGBM
- XGBoost

**Selection Criterion**: Model selected based on **grouped split performance** (geographic generalization), not random split performance.

---

## 📊 Results

(Results will be populated after running training pipeline)

### Performance Metrics

| Split Strategy | MAE (€) | RMSE (€) | R² |
|----------------|---------|----------|-----|
| Random Split   | TBD     | TBD      | TBD |
| Grouped Split  | TBD     | TBD      | TBD |
| Spatial Holdout| TBD     | TBD      | TBD |

### Error Analysis

- Performance by price range
- Performance by apartment size
- Performance by district
- Residual analysis
- Worst predictions analysis

See `scripts/analyze_errors.py` for details.

---

## 🔍 Explainability

SHAP (SHapley Additive exPlanations) analysis:

```bash
python scripts/explain_model.py
```

Provides:
- Global feature importance
- Individual prediction explanations
- Feature contribution analysis

---

## 🌐 API Usage

### Endpoints

#### `GET /health`
Health check

```bash
curl http://localhost:8000/health
```

#### `GET /model/info`
Model metadata

```bash
curl http://localhost:8000/model/info
```

#### `POST /predict`
Make prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "livingSpace": 70,
    "rooms": 3,
    "floor": 2,
    "yearConstructed": 2000,
    "geo_plz": "10115",
    "hasKitchen": true,
    "hasBalcony": true
  }'
```

Response:
```json
{
  "predicted_rent": 1250.50,
  "model_name": "LGBMRegressor",
  "timestamp": "2026-10-04T03:00:00.000Z"
}
```

---

## 🐳 Docker

### Build Image

```bash
docker build -t berlinrentml:latest .
```

### Run Container

```bash
docker run -p 8000:8000 berlinrentml:latest
```

API will be available at `http://localhost:8000`

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=berlinrentml --cov-report=html

# Run specific test file
pytest tests/test_features.py -v
```

### Test Coverage

- ✅ Data validation
- ✅ Data cleaning
- ✅ Feature engineering
- ✅ Preprocessing (leakage prevention)
- ✅ Model training
- ✅ API endpoints
- ✅ Invalid inputs

---

## 🔄 CI/CD

GitHub Actions workflow runs on every push:

- Install dependencies
- Run test suite
- Check code style (Black, Ruff)
- Test Python 3.10 and 3.11

See [.github/workflows/ci.yml](.github/workflows/ci.yml)

---

## 🎓 Key Learnings

### What Makes This a Serious ML Project

1. **Real data with real challenges** (missing values, duplicates, outliers)
2. **Leakage prevention** as first-class requirement
3. **Geographic generalization** explicitly tested
4. **Multiple evaluation strategies** compared
5. **Model selection** based on generalization, not training performance
6. **Production-ready code** (API, Docker, tests, CI)
7. **Transparent limitations** documented

### What This Project Does NOT Do

- ❌ Use fake or generated data
- ❌ Report only random-split performance
- ❌ Ignore spatial clustering
- ❌ Use unnecessary complexity (no Kubernetes, Kafka, etc.)
- ❌ Deep learning without justification
- ❌ Claim "production-ready" without evidence

---

## 📝 Limitations

- **Geographic scope**: Berlin only, may not generalize to other cities
- **Temporal drift**: Rental market changes over time, model may degrade
- **Data currency**: Dataset age affects relevance
- **Luxury properties**: Model may underperform on unusual/luxury properties
- **External factors**: Economic conditions, regulations not captured
- **Feature limitations**: Some relevant factors unavailable (property condition details, neighborhood amenities)

---

## 🔮 Future Improvements

- **Uncertainty quantification**: Implement conformal prediction for prediction intervals
- **Temporal models**: Add time-series features if more recent data available
- **Geographic features**: Add distance to amenities, public transport
- **Ensemble methods**: Combine multiple model types
- **AutoML**: Automated hyperparameter tuning at scale
- **Online learning**: Update model with new listings
- **A/B testing**: Compare model versions in production
- **Fairness analysis**: Check for demographic biases

---

## 📚 Documentation

- [SPEC.md](SPEC.md) - Technical specification
- [PLAN.md](PLAN.md) - Implementation plan
- [data/DATASET.md](data/DATASET.md) - Dataset documentation
- [MODEL_CARD.md](MODEL_CARD.md) - Model card (to be created)

---

## 🤝 Contributing

This is a portfolio/benchmark project. For issues or suggestions:

1. Fork the repository
2. Create a feature branch
3. Run tests (`pytest tests/ -v`)
4. Submit a pull request

---

## 📄 License

MIT License - see LICENSE file

---

## 🙏 Acknowledgments

- **Dataset**: Kaggle user corrieaar for the ImmobilienScout24 dataset
- **Platform**: ImmobilienScout24 for the original listings data

---

## 📞 Contact

For questions or feedback about this project, open an issue on GitHub.

---

**Built with serious ML engineering practices. No shortcuts. No fake data. No deceptive metrics.**
