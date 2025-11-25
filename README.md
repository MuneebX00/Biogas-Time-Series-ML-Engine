Biogas Production Forecast App (Streamlit)

Overview
This project is a Streamlit web app that trains an XGBoost regression model on historical plant data to forecast biogas production for the next 24–72 hours. It provides interactive controls to tweak model hyperparameters, optionally override exogenous drivers, visualize forecasts with confidence bands, and download results.

Repository contents
- `streamlit_app.py`: Streamlit app with data loading, feature engineering, model training, and forecasting UI.
- `biogas_hourly.csv`: Example dataset expected by the app.
- `anomalies.md` and `dataset.md`: Notes/documentation about data and anomalies (if provided by your project).
- `requirements.txt`: Base Python dependencies (you may add more as needed; see below).
- `venv/`: A Python virtual environment folder (optional to use).

Quick start (Windows PowerShell)
1) Open PowerShell and go to the project folder:
```powershell
cd C:\Users\Oogway\Desktop\final
```
2) (Recommended) Activate the virtual environment:
```powershell
.\n+venv\Scripts\Activate.ps1
```
If activation is blocked, allow scripts for current user:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

3) Install dependencies. The app requires at least: pandas, numpy, scikit-learn, xgboost, altair, streamlit.
```powershell
pip install -r requirements.txt streamlit xgboost scikit-learn altair numpy
```

4) Run the app:
```powershell
streamlit run streamlit_app.py
```
Your browser will open `http://localhost:8501` (or another port if 8501 is busy).

Data expectations
The app expects a CSV with at least these columns:
- `timestamp` (parseable datetime)
- `biogas_m3_hour` (numeric target per hour)

Optional exogenous features improve accuracy if present. The code supports and/or auto-detects many common drivers, including:
- Categorical: `feedstock` (will be category-encoded)
- Boolean flags (0/1 or True/False): `gas_leakage_flag`, `downtime_flag`, `is_holiday`, `rapid_temp_spike_flag`, `overheating_flag`, `solar_heating_flag`, `mixing_flag`, `low_quality_feed_flag`
- Numeric drivers (if present): `feed_volume_kg`, `temp_C`, `pH`, `VFA_mg_L_HAc`, `alkalinity_mg_L_CaCO3`, `methane_pct`, `TS_pct`, `VS_pct`, `CN_ratio`, `OLR_kg_TVS_m3_d`, `feed_quality_index`

How it works
1) Data loading
- The app reads the CSV (path configurable in the UI, default `biogas_hourly.csv`) and converts `timestamp` to datetime, sorting by time.

2) Feature engineering (`prepare_features`)
- Category-encodes `feedstock` if present.
- Casts boolean-like columns to integers (0/1).
- Creates target lags: t-1, t-24, t-168 (week), and a 24-hour rolling mean.
- Adds calendar features: hour, dayofweek, month, year.
- Drops rows with NaNs introduced by lagging/rolling to keep training aligned.

3) Model training (`train_model`)
- Splits chronologically into train/test using `test_size` (no shuffling) to reflect time ordering.
- Trains XGBoost regressor (`objective = reg:squarederror`, `eval_metric = rmse`) with early stopping on a held-out test slice.
- Records the exact training feature order to ensure alignment at prediction time.
- Reports diagnostics (RMSE, MAE, MAPE, R², test size) in the UI.

4) Forecasting (`recursive_forecast`)
- Produces 24/48/72-hour forecasts recursively. For each future hour, it:
  - Builds a base row with seasonal-naive exogenous values (uses value from t-24 if available, else last known).
  - Applies user overrides from the UI (if enabled).
  - Recomputes the same engineered features (lags, rolling mean, calendar fields).
  - Predicts with XGBoost and feeds the prediction back into subsequent steps.
- Confidence intervals are approximated using a normal assumption with the residual standard deviation from the test set and a selectable z-score (0.80/0.90/0.95).

Using the app
- CSV path: Top text input (default `biogas_hourly.csv`).
- Preprocessing expander: Optionally filter out rows flagged as low quality feed.
- Tabs:
  - Forecast: Shows the forecast line with a confidence band, provides a CSV download, and displays recent history.
  - Model settings: Adjust train/test split, XGBoost hyperparameters, and training runtime (rounds, early stopping).
  - Feature overrides: Enable and set custom numeric/boolean exogenous values to run what-if scenarios.

Dependencies
Minimum packages required by the code:
- pandas
- numpy
- scikit-learn
- xgboost
- altair
- streamlit

You can add these to `requirements.txt` or install them as shown in Quick start. If you encounter build issues with `xgboost` on Windows, try upgrading pip and wheel first:
```powershell
python -m pip install --upgrade pip wheel
pip install xgboost
```
GPU/CUDA is not required; CPU wheels are sufficient for this app.

Troubleshooting
- Module not found (e.g., streamlit, xgboost):
  - Ensure your venv is activated and run the install command again.
- Streamlit doesn’t open a browser:
  - Check the console for the local URL and open it manually.
  - If port 8501 is busy, Streamlit will pick another port; look for it in the console.
- CSV not found or parse error:
  - Confirm the path in the top text input and that `timestamp` is a parseable datetime column.
- Altair chart errors:
  - Ensure altair is installed and up to date: `pip install --upgrade altair`.
- Performance/memory:
  - Reduce `num_boost_round` and/or `max_depth` in Model settings; decrease `test_size`.

Repro commands
From the project root in PowerShell:
```powershell
.
venv\Scripts\Activate.ps1
pip install -r requirements.txt streamlit xgboost scikit-learn altair numpy
streamlit run streamlit_app.py
```

Notes
- The app caches CSV loading (`@st.cache_data`) for faster reloads when tweaking settings.
- Forecast download is available via the Download button as `forecast.csv`.








