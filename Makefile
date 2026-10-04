.PHONY: install data train tune analyze explain test app api docker drift all

install:
	pip install -e ".[dev,train,ui]"

data:
	kaggle datasets download -d corrieaar/apartment-rental-offers-in-germany -p data/raw --unzip

train:
	python scripts/train.py

tune:
	python scripts/tune.py

analyze:
	python scripts/analyze_errors.py
	python scripts/explain_model.py

test:
	pytest

app:
	streamlit run app/streamlit_app.py

api:
	uvicorn berlinrentml.api.main:app --port 8000

drift:
	python scripts/check_drift.py

docker:
	docker build -t berlin-rent .

# train -> tune -> analyze -> test
all: train tune analyze test
