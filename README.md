<div align="center">

<img src=".github/assets/hero.svg" alt="BerlinRentML: gradient boosting rent estimates for Berlin, scored on unseen postal codes" width="100%">

<br>

![Python](https://img.shields.io/badge/python-3.10%2B-f2b53c?style=flat-square&labelColor=0b0f14)
![Model](https://img.shields.io/badge/model-LightGBM-4ade80?style=flat-square&labelColor=0b0f14)
![Serving](https://img.shields.io/badge/serving-FastAPI-4cc9f0?style=flat-square&labelColor=0b0f14)
![Listings](https://img.shields.io/badge/listings-10%2C388-a78bfa?style=flat-square&labelColor=0b0f14)
![MAE](https://img.shields.io/badge/MAE%20unseen%20PLZ-%E2%82%AC191-e0564f?style=flat-square&labelColor=0b0f14)
![License](https://img.shields.io/badge/license-MIT-8b98a6?style=flat-square&labelColor=0b0f14)

[Quick start](#quick-start) · [Results](#results) · [Limitations](#limitations) · [Model card](MODEL_CARD.md)

</div>

Enter a flat, get a rent estimate. Trained on about 10,000 real ImmoScout24 Berlin listings and served through a Streamlit UI and a FastAPI endpoint, with tests, Docker and CI.

Berlin listings cluster by location, so a random train/test split leaks neighbourhood information and flatters the score. This project also evaluates with a **postal-code grouped split**, where every test postal code is unseen during training.

> The elevation above is not decoration: every block is a real flat from
> `data/processed/berlin_processed.csv`, its height set by that flat's monthly
> rent and its width by its floor area, sampled evenly across the market.
> Redraw it with `python .github/assets/make_assets.py`.

---

## Pipeline

<img src=".github/assets/pipeline.svg" alt="Method: clean, engineer features, split by postal code, fit, calibrate" width="100%">

---

## Quick start

```bash
pip install -e ".[dev,train,ui]"
python scripts/train.py            # trains, compares models, saves models/
python scripts/tune.py             # tuned log-rent model + 90% intervals (overwrites models/)
python scripts/analyze_errors.py   # error by district, size and price range
python scripts/explain_model.py    # SHAP feature importance
pytest                             # run tests
streamlit run app/streamlit_app.py # web UI at http://localhost:8501
```

Or serve the REST API instead of the UI:

```bash
uvicorn berlinrentml.api.main:app --port 8000
```

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"livingSpace": 60, "rooms": 2, "geo_plz": "10115", "geo_bln": "Mitte"}'
```

```json
{"predicted_rent": 1197.21, "interval_low": 837.13, "interval_high": 1711.99, "model_name": "LGBMRegressor", "timestamp": "..."}
```

Same flat in Marzahn (`"geo_plz": "12619", "geo_bln": "Marzahn"`) returns about 613 €. Interactive docs are at `/docs`.

## Results

10,388 Berlin listings after cleaning, 80/20 split. LightGBM wins on both splits.

<img src=".github/assets/leakage.svg" alt="Mean absolute error per model, random split against postal-code grouped split" width="100%">

- **LightGBM:** R² 0.880 on the random split, 0.836 on unseen postal codes. About 62% lower error than the median baseline.
- **Leakage is real:** linear regression goes from €186 to €291 once test postal codes are unseen. The final model is picked by grouped-split MAE, not random-split MAE.
- **Where it fails:** the model is least accurate in expensive central districts and for large flats, which it under-prices.

<img src=".github/assets/districts.svg" alt="Mean absolute error by district on the random test set" width="100%">

## Beyond a baseline

`python scripts/tune.py` goes further than a single fit. It splits by postal code into train, calibration and test sets, so every number is for neighbourhoods the model never saw.

| LightGBM variant | MAE (€) | R² |
|---|---|---|
| Default, raw rent target | 195.6 | 0.835 |
| Log-rent target | 193.3 | 0.832 |
| Log-rent, tuned (40-candidate search, postal-code grouped CV) | **190.7** | 0.831 |

The gains are small, and the README says so on purpose: the limit is the data (asking rents, no address, no photos), not the algorithm.

- **Prediction intervals.** Split-conformal intervals from out-of-fold residuals on unseen postal codes. The 90% interval covered 88.5% of test listings, with a median width of about €615. Coverage is weaker in rare, expensive districts (Grunewald: 59%), which the notebook shows.
- **Experiment tracking.** Each tuning run is logged to MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).
- **Drift monitoring.** The API logs requests to `logs/requests.jsonl`. `python scripts/check_drift.py` computes the Population Stability Index of live inputs against the training data.
- **Notebooks.** [01 · EDA and the leakage trap](notebooks/01_eda_and_leakage.ipynb) and [02 · Modeling and uncertainty](notebooks/02_modeling_and_uncertainty.ipynb), executed with outputs.
- **Reproducibility.** `make all` runs train, tune, analysis and tests. CI also trains on synthetic data, starts the API and calls `/predict` as an end-to-end smoke test.

## What is inside

| Area | What it does |
|---|---|
| Data | Loads the Kaggle CSV, filters to Berlin, maps columns, validates schema and ranges |
| Cleaning | Removes impossible values, caps rent outliers, handles missing data |
| Features | Building age, size and age bins, log size, rooms per m², amenity score |
| Evaluation | Random split and `GroupedSplit` by postal code, MAE, RMSE, R², share within €50/100/200 |
| Models | Mean, median, linear, ridge, random forest, LightGBM, XGBoost, lasso, elastic net |
| Serving | FastAPI with validated requests, `/health`, `/model/info`, `/predict` |
| Quality | pytest suite, GitHub Actions CI, Dockerfile, model card |

## Install

Requires Python 3.10 or newer.

```bash
git clone https://github.com/arjun05-tf/berlin-rent-predictor.git
cd berlin-rent-predictor
pip install -e ".[dev,train,ui]"
```

### Get the data

Download `immo_data.csv` from [Kaggle: Apartment rental offers in Germany](https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany) and place it in `data/raw/`. Or use the CLI:

```bash
kaggle datasets download -d corrieaar/apartment-rental-offers-in-germany -p data/raw --unzip
```

The dataset has 268,850 listings across Germany from 2018 to 2020. The pipeline keeps the 10,406 Berlin rows. The CSV is gitignored. See [data/DATASET.md](data/DATASET.md).

## Common commands

```bash
python scripts/train.py            # full pipeline, saves models/final_model_*.joblib
python scripts/analyze_errors.py   # error by district, size and price range
python scripts/explain_model.py    # SHAP feature importance
pytest                             # run tests
```

### Deploy to Vercel

The repo is Vercel-ready: `public/index.html` is the web form, `api/index.py` serves the FastAPI app as a serverless function (`/api/index?r=predict`, `/api/index?r=health`), and the 1.5 MB model files in `models/` ship with it. Runtime dependencies are pinned in `requirements.txt` to the versions the model was trained with.

```bash
npm i -g vercel
vercel          # preview deploy, then: vercel --prod
```

Or import the GitHub repo at [vercel.com/new](https://vercel.com/new). No build settings are needed. If you retrain, run `python scripts/export_options.py` to refresh the dropdown values, then commit `models/` and `public/options.json`. The Streamlit app (`app/`) is for local use, because Vercel cannot host long-lived Streamlit servers.

### Docker

Train first, because the image copies `models/`.

```bash
docker build -t berlin-rent .
docker run -p 8000:8000 berlin-rent
```

## Project layout

```
src/berlinrentml/
  data/        loading, validation, cleaning
  features/    feature engineering, preprocessing
  modeling/    training, evaluation, split strategies
  api/         FastAPI service
scripts/       train, error analysis, SHAP
tests/         unit and API tests
```

## Limitations

- The data is from 2018 to 2020. Predictions reflect past rents, not today's market.
- Offered rents are asking prices, not signed contracts.
- Postal codes unseen in training fall back on the other features, and accuracy drops (about €31 MAE worse than the random split).
- Intervals under-cover in rare, expensive districts, and the model under-prices large high-end flats.

See [MODEL_CARD.md](MODEL_CARD.md) for intended use and ethical considerations.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Model not loaded` from `/predict` | Run `python scripts/train.py` first |
| `No data file found` | Put `immo_data.csv` in `data/raw/` |
| `UnicodeEncodeError` on Windows | Set `PYTHONUTF8=1` before running scripts |
| Docker build fails at `COPY models/` | Train first so `models/` has the `.joblib` files |

## License

MIT. Data from ImmoScout24 via Kaggle, see the dataset page for its terms.
