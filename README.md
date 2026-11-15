# AIRGUARD INDIA
**Intelligent Air Quality Analytics and AQI Risk Prediction**

A Streamlit analytics dashboard for historical Indian city air-quality observations (2015–2020). It is an offline analytical project, not a live AQI monitoring service.

## Project structure
```text
airguard-india/
├── app.py
├── requirements.txt
├── README.md
├── assets/
│   └── custom.css
└── data/
    └── city_day.csv
```

## Run locally on macOS
```bash
cd airguard-india
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
Streamlit opens the local app in your browser (normally `http://localhost:8501`).

## Dataset
Download **Air Quality Data in India (2015–2020)** by Rohan Rao from Kaggle:
https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

Place `city_day.csv` in `data/`, or upload `city_day.csv` / its ZIP archive using the app's sidebar. The dashboard validates the required `City` and `Date` columns and gracefully omits analyses that need unavailable optional columns.

## Included dashboard sections
- Executive overview and dynamic KPI cards
- City comparisons and AQI category composition
- Pollutant distributions, trends, scatter plots and correlations
- Daily, monthly, yearly and seasonal analysis
- AQI risk category analysis
- Descriptive statistics, IQR outlier screening, PCA and K-Means
- Random Forest classification for the **next available labelled observation per city**
- Interactive data explorer, missing-value audit and CSV export
- Automatically calculated findings

## Methodology notes
- Missing pollutant values are not blindly dropped. Descriptive analyses use available observations; PCA/clustering use median imputation and standardization; the classifier uses a preprocessing pipeline fitted on the training data.
- AQI target labels missing from the source data are excluded from supervised target transitions.
- The model sorts observations by city and date, creates the next available labelled category as the target, and evaluates on a chronological holdout. This is **not necessarily next-calendar-day prediction**.
- The dashboard computes model metrics from the loaded file at runtime. It does not hardcode model scores.
- IQR outliers are flagged for investigation, not automatically removed.
- Correlation is not causation. Results are limited by missing values, historical coverage and unavailable weather/traffic/station metadata.

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `README.md`, `assets/custom.css` (optional), and `data/city_day.csv` if repository size/privacy/licensing permits.
2. Sign in to Streamlit Community Cloud and choose **Create app**.
3. Select the repository, branch, and `app.py` as the main file.
4. Deploy. If the CSV is not committed, upload it through the app's sidebar after deployment.
5. Keep the repository's `requirements.txt` at its root so Cloud can install dependencies.

## Troubleshooting
- **`streamlit: command not found`**: activate `.venv` and run `python3 -m pip install -r requirements.txt`; alternatively use `python3 -m streamlit run app.py`.
- **No dataset found**: put the exact file at `data/city_day.csv` or upload the CSV/ZIP in the sidebar.
- **Missing required columns**: ensure the file is the city-day dataset and has `City` and `Date` columns.
- **Slow PCA/K-Means**: the app caps the analytics sample at 12,000 rows for responsiveness.
- **Model cannot train**: verify that `AQI_Bucket` and pollutant columns exist and that enough labelled observations and target classes remain after filtering.
