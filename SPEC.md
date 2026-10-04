# BerlinRentML — Technical Specification

## Project Title
**BerlinRentML: Spatially Robust Berlin Rental Price Prediction**

## Objective
Build a production-grade end-to-end machine learning system for predicting monthly cold rent (€) for apartments in Berlin, with explicit focus on geographic generalization and leakage prevention.

## Problem Statement
Predict the monthly cold rent (Kaltmiete) in euros for residential apartments in Berlin based on property characteristics, location, and available metadata.

## Dataset
**Primary Source**: Kaggle - Apartment Rental Offers in Germany (ImmobilienScout24)
- **URL**: https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany
- **Source**: ImmobilienScout24 (Germany's largest real estate platform)
- **License**: Public dataset (verify specific license on Kaggle)
- **Scope**: ~250,000+ rental listings across Germany
- **Target Subset**: Berlin listings only
- **Collection Method**: Web scraping from public listings

### Expected Schema
Based on typical ImmobilienScout24 data:
- `totalRent` / `baseRent` / `serviceCharge` (€)
- `livingSpace` (m²)
- `rooms` / `numberOfRooms`
- `floor`
- `yearConstructed` / `constructionYear`
- `condition` / `interiorQual`
- `hasKitchen`, `hasBalcony`, `hasGarden`, `hasCellar`
- `heatingType`, `firingTypes`
- `geo_plz` (postal code)
- `geo_bln` (district/neighborhood)
- `regio1` (state - filter for "Berlin")
- `scoutId` (listing ID)
- `date` (listing date if available)
- Geographic coordinates (lat/lon if available)

## Target Variable
**baseRent** (Kaltmiete) - monthly cold rent in euros, excluding utilities and service charges.

If only `totalRent` is available, we'll need to carefully handle the warm-rent vs cold-rent distinction to avoid leakage.

## Critical ML Requirements

### 1. Leakage Prevention (CRITICAL)
Explicitly investigate and prevent:
- **Target leakage**: No features derived from the target
- **Warm-rent leakage**: If using totalRent, carefully separate baseRent and serviceCharge
- **Preprocessing leakage**: Fit all transformers on training data only
- **Duplicate leakage**: Same property listed multiple times
- **Property-level leakage**: Same building/address in train and test
- **Temporal leakage**: Features unavailable at prediction time
- **Geographic leakage**: Coordinates/identifiers that memorize exact locations

### 2. Geographic Generalization (CRITICAL)
Berlin listings are geographically clustered. Evaluation must test:

**Evaluation Strategy**:
1. **Random Split** (baseline): Standard 80/20 split
2. **Grouped Split**: Group by postal code or district before splitting
3. **Spatial Holdout**: Hold out entire geographic regions (e.g., 2-3 districts)
4. **Temporal Holdout**: If dates available, train on older listings, test on newer

**Key Question**: Can the model generalize to new geographic areas it hasn't seen, or is it memorizing location-specific patterns?

### 3. Data Quality
- Handle missing values systematically
- Detect and handle duplicates (same property, repeated listings)
- Identify and handle outliers (luxury properties, data errors)
- Validate categorical encodings
- Check for data entry errors (impossible values)

## Features

### Core Features
- **Size**: living space (m²)
- **Rooms**: number of rooms
- **Location**: district, postal code (encoded carefully to avoid leakage)
- **Building age**: year constructed or derived age
- **Floor**: floor level
- **Condition**: property condition/quality

### Amenity Features
- Kitchen availability
- Balcony/terrace
- Garden access
- Cellar/basement
- Parking

### Heating/Energy
- Heating type
- Energy certificate (if available)

### Engineered Features
- **Price per m²** (derived POST-split only for analysis, NOT as input feature)
- **Building age bins**
- **Size bins**
- **Location aggregations** (district-level statistics, fitted on training only)
- **Distance to city center** (if coordinates available)

### Explicitly EXCLUDED (Leakage Risks)
- Total rent (if predicting base rent)
- Any target-derived statistics
- Exact coordinates (use coarse location encoding)
- Property-specific IDs that could memorize individual properties

## Models

### Baselines
1. **Mean predictor**: Predict mean rent
2. **Median predictor**: Predict median rent
3. **Linear regression**: Simple OLS
4. **Regularized linear**: Ridge/Lasso

### Main Models
1. **Regularized Linear Regression** (Ridge, Lasso, ElasticNet)
2. **Random Forest Regressor**
3. **Gradient Boosting** (one or more of):
   - LightGBM
   - XGBoost
   - CatBoost (good for categorical features)

### Selection Criteria
- **NOT** selected purely on random-split performance
- **Primary criterion**: Geographic generalization performance
- **Secondary**: Interpretability, training time, prediction speed
- **Tertiary**: Random-split performance (as sanity check)

### Hyperparameter Tuning
- Use nested cross-validation or separate validation set
- **Never tune on test set**
- For spatial evaluation, use spatial CV (GroupKFold by location)

## Evaluation

### Metrics
Primary:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)

Secondary:
- **Median Absolute Error**
- **Percentage within €50/€100/€200**
- **Residual distribution analysis**

### Error Analysis
Performance breakdown by:
- **Price ranges**: <€500, €500-€1000, €1000-€1500, >€1500
- **Apartment sizes**: <40m², 40-70m², 70-100m², >100m²
- **Districts/neighborhoods**
- **Geographic regions** (for spatial holdout)
- **Building age categories**

### Spatial Analysis
- **Rent distribution map** across Berlin
- **Error distribution map** (where does the model fail?)
- **Performance by district**
- **Geographic clustering of errors**

### Critical Comparison
Compare:
- Random split performance
- Grouped split performance
- Spatial holdout performance
- Temporal holdout performance (if applicable)

**If random >> spatial**: Investigate why. Is the model memorizing location rather than learning transferable patterns?

## Uncertainty Quantification
Implement one of:
1. **Quantile Regression**: Predict 10th, 50th, 90th percentiles
2. **Conformal Prediction**: Distribution-free prediction intervals
3. **Ensemble variance**: Prediction variance across forest/boosting trees

Do NOT label arbitrary model outputs as "confidence" without statistical validity.

## Explainability
- **Global**: Feature importance (SHAP, permutation importance)
- **Local**: SHAP values for individual predictions
- **Examples**: Show 5-10 example predictions with explanations
- **Caution**: Distinguish correlation from causation

## Architecture

### Components
1. **Data Ingestion**: Download and load raw data
2. **Data Validation**: Schema validation, quality checks
3. **Data Cleaning**: Missing values, duplicates, outliers
4. **Feature Engineering**: Transformations, encodings
5. **Train/Test Splitting**: Multiple strategies (random, grouped, spatial, temporal)
6. **Model Training**: Train multiple models with hyperparameter tuning
7. **Model Evaluation**: Comprehensive metrics and error analysis
8. **Model Selection**: Select best model based on geographic generalization
9. **Model Serialization**: Save final model and preprocessing artifacts
10. **Inference API**: FastAPI service for predictions
11. **Testing**: Unit tests, integration tests, API tests

### Technology Stack
- **Language**: Python 3.10+
- **ML**: scikit-learn, LightGBM/XGBoost/CatBoost, SHAP
- **Data**: pandas, numpy, geopandas (if geographic analysis)
- **API**: FastAPI, Pydantic
- **Testing**: pytest
- **Container**: Docker
- **CI/CD**: GitHub Actions
- **Config**: YAML or TOML
- **Logging**: Python logging module

## API Specification

### Endpoints

#### `GET /health`
Health check
- Returns: `{"status": "healthy"}`

#### `GET /model/info`
Model metadata
- Returns: Model name, version, training date, features, metrics

#### `POST /predict`
Single prediction
- Input: Property features (JSON)
- Output: Predicted rent, uncertainty interval (if available)

#### `POST /predict/batch`
Batch predictions
- Input: List of property features
- Output: List of predictions

### Input Validation
- Pydantic models for request validation
- Range checks (e.g., rooms > 0, livingSpace > 0)
- Required vs optional fields

## Testing Strategy

### Unit Tests
- Data validation functions
- Feature engineering functions
- Preprocessing transformations
- Model loading/saving

### Integration Tests
- End-to-end training pipeline
- End-to-end inference pipeline
- API endpoints

### Model Tests
- Prediction consistency (same input → same output)
- Prediction reasonableness (within expected range)
- Shape tests (correct output dimensions)

### Data Tests
- Schema validation
- Value range checks
- Duplicate detection
- Missing value detection

## Reproducibility

### Requirements
- Pinned dependencies (requirements.txt or poetry.lock)
- Random seeds for all random operations
- Configuration files for all hyperparameters
- Clear instructions for data acquisition
- Model versioning

### Documentation
- README with full setup instructions
- Data acquisition steps
- Training instructions
- Inference instructions
- API usage examples

## Docker

### Container Requirements
- Install dependencies
- Copy model artifacts
- Expose API port
- Health check
- Graceful shutdown

## CI/CD

### GitHub Actions
- Install dependencies
- Run tests
- Lint/type checking (optional but recommended)
- Build Docker image (if on main branch)

## Model Card
Document:
- Intended use
- Out-of-scope use
- Training data characteristics
- Evaluation methodology
- Performance metrics (all splits)
- Known limitations
- Bias/fairness considerations
- Geographic limitations
- Uncertainty information

## Known Limitations
- Model trained on Berlin listings only
- Performance depends on data currency
- May not generalize to luxury/unusual properties
- Geographic generalization varies by district
- Temporal drift: rental market changes over time
- Does not account for external economic factors

## Success Criteria
1. ✅ Real dataset successfully acquired and validated
2. ✅ Leakage analysis performed and documented
3. ✅ Multiple evaluation strategies implemented
4. ✅ Geographic generalization explicitly tested
5. ✅ Final model has measured MAE/RMSE/R² on all splits
6. ✅ Error analysis completed
7. ✅ API functional and tested
8. ✅ Test suite passes
9. ✅ Docker container builds and runs
10. ✅ Documentation complete

## Timeline (Autonomous Execution)
1. **Data Acquisition**: Download dataset, validate schema
2. **EDA**: Understand data, identify issues
3. **Data Cleaning**: Handle missing, duplicates, outliers
4. **Feature Engineering**: Create and validate features
5. **Evaluation Framework**: Implement multiple split strategies
6. **Baseline Models**: Train simple baselines
7. **Advanced Models**: Train and tune ML models
8. **Model Selection**: Choose based on geographic generalization
9. **Error Analysis**: Deep dive into model performance
10. **Explainability**: SHAP analysis
11. **Uncertainty**: Implement quantification
12. **API**: Build FastAPI service
13. **Testing**: Write and run tests
14. **Docker**: Containerize API
15. **Documentation**: README, model card
16. **CI**: GitHub Actions
17. **Review**: Senior ML review
18. **Simplification**: Remove unnecessary complexity
19. **Final Validation**: Run full test suite
20. **Delivery**: Final report

---

**Status**: Specification complete. Proceeding to implementation.
