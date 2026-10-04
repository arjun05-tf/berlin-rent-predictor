# Dataset: Apartment Rental Offers in Germany

## Source
**Kaggle Dataset**: [Apartment Rental Offers in Germany](https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany)

**Original Source**: ImmobilienScout24 (Germany's largest real estate platform)

## License
Check Kaggle dataset page for specific license terms.

## Download Instructions

### Option 1: Kaggle CLI (Recommended)
```bash
# Install kaggle CLI
pip install kaggle

# Configure Kaggle API credentials
# Download kaggle.json from https://www.kaggle.com/settings
# Place in ~/.kaggle/kaggle.json (Linux/Mac) or %USERPROFILE%\.kaggle\kaggle.json (Windows)

# Download dataset
kaggle datasets download -d corrieaar/apartment-rental-offers-in-germany -p data/raw/

# Unzip
unzip data/raw/apartment-rental-offers-in-germany.zip -d data/raw/
```

### Option 2: Manual Download
1. Go to https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany
2. Click "Download" button
3. Extract the CSV file to `data/raw/`
4. Rename to `immo_data.csv` if needed

## Expected Files
- `data/raw/immo_data.csv` (or similar name)

## Dataset Characteristics
- **Size**: ~250,000+ listings across Germany
- **Target Subset**: Berlin only (filter by region/city column)
- **Time Period**: Varies (check dataset version)
- **Format**: CSV

## Schema
Expected columns (verify after download):
- Rent information: `totalRent`, `baseRent`, `serviceCharge`
- Property characteristics: `livingSpace`, `rooms`, `floor`, `yearConstructed`
- Amenities: `hasKitchen`, `hasBalcony`, `hasGarden`, `cellar`
- Location: `geo_plz` (postal code), `geo_bln` (district), `regio1` (state)
- Identifiers: `scoutId`, `date`
- Condition: `condition`, `interiorQual`
- Heating: `heatingType`, `firingTypes`

## Usage
```python
from berlinrentml.data.loader import load_raw_data

# Load and filter for Berlin
df = load_raw_data()
```

## Data Quality Notes
- Check for missing values in critical columns
- Validate warm rent vs cold rent separation
- Check for duplicate listings (same property, multiple postings)
- Verify geographic coverage across Berlin districts
- Check date ranges for temporal analysis
