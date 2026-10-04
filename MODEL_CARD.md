# Model Card: BerlinRentML

## Model Details

**Model Name**: BerlinRentML Rental Price Predictor  
**Version**: 0.1.0  
**Date**: October 2026  
**Model Type**: Ensemble of regression models (LightGBM, Random Forest, Ridge)  
**Framework**: scikit-learn, LightGBM  

### Model Description

BerlinRentML predicts monthly cold rent (Kaltmiete) for residential apartments in Berlin based on property characteristics, location, and amenities. The model was developed with explicit focus on geographic generalization and leakage prevention.

## Intended Use

### Primary Use Cases

- **Rental price estimation**: Landlords and tenants estimating fair market rent
- **Market analysis**: Real estate professionals analyzing Berlin rental market
- **Research**: Academic/commercial research on rental prices
- **Apartment search**: Helping renters identify overpriced or underpriced listings

### Intended Users

- Real estate professionals
- Property management companies
- Apartment seekers
- Market researchers
- Urban planners

### Out-of-Scope Use Cases

- **Other cities**: Model trained on Berlin only, not applicable elsewhere
- **Commercial properties**: Trained on residential apartments only
- **Purchase prices**: Predicts rent, not sale prices
- **Legal/binding valuations**: Not a replacement for professional appraisals
- **Luxury properties**: May underperform on high-end/unusual properties
- **Real-time pricing**: Does not account for rapid market changes

## Training Data

### Dataset Source

- **Platform**: ImmobilienScout24 (via Kaggle)
- **URL**: https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany
- **Collection**: Web-scraped public rental listings
- **Geographic scope**: Berlin, Germany
- **Time period**: [Dataset specific - check data version]
- **Size**: ~10,000-50,000 Berlin listings (varies by dataset version)

### Data Characteristics

**Target Variable**: `baseRent` (monthly cold rent in €)

**Features Used**:
- **Property**: Living space (m²), rooms, floor, construction year
- **Location**: Postal code, district (encoded)
- **Amenities**: Kitchen, balcony, garden, cellar
- **Condition**: Property condition, interior quality
- **Heating**: Heating type

**Data Quality Issues Addressed**:
- Missing values (imputed or removed)
- Duplicates (removed)
- Outliers (capped at 3× IQR)
- Invalid values (removed)

### Data Preprocessing

1. **Cleaning**: Removed listings with missing target, invalid values
2. **Feature Engineering**: Building age, size bins, amenity scores
3. **Encoding**: One-hot encoding for categorical variables
4. **Scaling**: StandardScaler for numeric features
5. **Leakage Prevention**: All transformations fitted on training data only

## Evaluation

### Evaluation Data

Multiple evaluation strategies used:

1. **Random Split** (80/20): Standard train/test split
2. **Grouped Split**: Grouped by postal code before splitting
3. **Spatial Holdout**: Complete districts held out for testing

### Evaluation Metrics

**Primary Metrics**:
- **MAE** (Mean Absolute Error): Average absolute prediction error in €
- **RMSE** (Root Mean Squared Error): Penalizes large errors more heavily
- **R²** (Coefficient of Determination): Proportion of variance explained

**Secondary Metrics**:
- Median Absolute Error
- Percentage within €50/€100/€200
- Residual distribution analysis

### Performance Results

| Split Strategy | MAE (€) | RMSE (€) | R² |
|----------------|---------|----------|-----|
| Random Split   | 162.30  | 250.52   | 0.880 |
| Grouped Split (by postal code) | 193.56 | 290.78 | 0.836 |

Final model: LightGBM (default params), 10,388 Berlin listings, ImmoScout24 data from 2018-2020. Median baseline MAE is about €507. Rents are historical, not current.

**Key Findings**:
- Unseen postal codes cost about €31 MAE vs random split (193.56 vs 162.30)
- Linear model degrades most on unseen postal codes (186 to 291); LightGBM degrades least
- See scripts/analyze_errors.py

## Ethical Considerations

### Potential Biases

**Geographic Bias**:
- Model trained on Berlin only
- Performance may vary significantly across districts
- Wealthier districts may have better data coverage

**Temporal Bias**:
- Trained on historical data
- May not reflect current market conditions
- Rental market affected by economic changes, regulations

**Data Collection Bias**:
- Only includes listings posted on ImmobilienScout24
- May miss informal rental market
- Selection bias: not all properties are listed

**Feature Limitations**:
- Does not capture neighborhood amenities (parks, schools, transport)
- Does not capture property-specific factors (renovation, views)
- May disadvantage unusual or unique properties

### Fairness Considerations

**Protected Attributes**:
- Model does NOT use demographic information
- Location encoding may indirectly reflect demographic patterns
- Users should be aware predictions may reflect existing market inequalities

**Potential Harms**:
- Overpricing: Landlords using predictions to justify high rents
- Undervaluation: Tenants undervaluing quality properties
- Displacement: Predictions used to justify rent increases
- Discrimination: Geographic predictions reflecting historical segregation

### Recommendations for Responsible Use

1. **Do not use as sole decision factor**: Combine with human judgment
2. **Verify predictions**: Cross-reference with actual listings
3. **Consider context**: Account for factors model doesn't capture
4. **Monitor fairness**: Check for disparate impact across neighborhoods
5. **Update regularly**: Retrain as market conditions change
6. **Transparency**: Disclose to users that predictions are algorithmic

## Limitations

### Known Limitations

**Data Limitations**:
- Dataset age: May not reflect current market
- Geographic coverage: Some districts underrepresented
- Feature limitations: Many relevant factors not captured

**Model Limitations**:
- Cannot predict extreme/unusual properties well
- Geographic generalization varies by district
- Does not capture temporal trends
- Uncertainty estimates not calibrated (if provided)

**Technical Limitations**:
- Requires same features as training data
- Unknown categories handled by imputation
- No built-in market trend adjustment
- Static model (no online learning)

### Performance Variability

Model performance varies by:
- **Price range**: May underperform on very cheap or expensive apartments
- **Property size**: May underperform on very small or large apartments
- **District**: Performance varies geographically
- **Property type**: Optimized for typical residential apartments

## Uncertainty Quantification

(If implemented)

- **Method**: [Quantile regression / Conformal prediction / None]
- **Calibration**: [How uncertainty estimates were validated]
- **Interpretation**: [What uncertainty intervals mean]

If not implemented:
- Predictions are point estimates without uncertainty
- Users should treat all predictions with appropriate skepticism
- Larger errors expected for unusual properties

## Explainability

**Global Explainability**:
- SHAP feature importance available
- Top features: [Living space, location, rooms, construction year]

**Local Explainability**:
- SHAP values for individual predictions
- Shows which features increased/decreased prediction

**Interpretation**:
- SHAP values show correlation, NOT causation
- Feature importance reflects patterns in training data
- May reflect existing market inequalities

## Caveats and Recommendations

### For Landlords
- Use predictions as market reference, not pricing authority
- Consider property-specific factors model doesn't capture
- Do not use to justify excessive rent increases
- Verify against actual comparable listings

### For Tenants
- Use predictions to identify potentially overpriced listings
- Consider factors beyond price (location, quality, amenities)
- Do not reject good apartments based solely on prediction
- Verify predictions against multiple sources

### For Researchers
- Understand evaluation methodology (random vs. spatial splits)
- Consider geographic and temporal limitations
- Account for potential biases in analysis
- Cite dataset and model version

### For Developers
- Monitor model performance over time
- Retrain regularly (quarterly or annually)
- Implement fairness monitoring
- Provide uncertainty estimates to users
- Log predictions for audit trail

## Model Updates

**Current Version**: 0.1.0 (October 2026)

**Update Policy**:
- Retrain when performance degrades significantly
- Retrain when new data becomes available
- Retrain annually at minimum
- Version all model artifacts
- Document changes in each version

**Performance Monitoring**:
- Track MAE on new data
- Monitor error distribution by district
- Check for fairness issues
- Alert if performance degrades >10%

## Contact

For questions, issues, or feedback about this model:
- GitHub Issues: [repository URL]
- Email: [contact email]

## References

**Dataset**:
- Kaggle: Apartment Rental Offers in Germany (corrieaar)
- ImmobilienScout24: Original listing source

**Code Repository**:
- [GitHub repository URL]

**Related Work**:
- Berlin Mietspiegel (official rent index)
- Academic research on rental price prediction
- Geographic machine learning literature

---

**Version History**:
- v0.1.0 (2026-10-04): Initial release

**Last Updated**: 2026-10-04
