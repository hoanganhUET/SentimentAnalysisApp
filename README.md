# SentimentAnalysisApp

Python project for Amazon product review preprocessing, sentiment scoring, and recommendation experimentation.

## Requirements

- Python 3.10+
- pip

## Initial Setup

1. Create and activate a virtual environment.
2. Install dependencies.
3. Create your .env file from .env.example.

Example (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

## Environment Variables

Configured by src/config/config.py:

- DATA_PATH
- MODEL_PATH
- API_KEY

## Data Pipeline

Download raw dataset:

```powershell
python data/raw/download.py --category Electronics
```

Process raw data into JSONL files:

```powershell
python src/data_pipeline/preprocessor.py --chunk-size 20000
```

Processed outputs are written to:

- data/processed/reviews_clean.jsonl
- data/processed/interactions.jsonl
- data/processed/products_meta.jsonl

## Logging

Logging config lives in src/config/logging.yaml.
Default rotating log file path: logs/app.log.
