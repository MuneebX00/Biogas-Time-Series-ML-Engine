import pandas as pd
import numpy as np
import xgboost as xgb
import altair as alt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error


@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp')


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    # Encode categorical
    if 'feedstock' in engineered.columns:
        engineered['feedstock'] = engineered['feedstock'].astype('category').cat.codes
    # Booleans to ints
    for col in ['gas_leakage_flag','downtime_flag','is_holiday','rapid_temp_spike_flag','overheating_flag','solar_heating_flag','mixing_flag','low_quality_feed_flag']:
        if col in engineered.columns:
            engineered[col] = engineered[col].astype(int)
    # Lags and rolling
    engineered['biogas_m3_hour_lag1'] = engineered['biogas_m3_hour'].shift(1)
    engineered['biogas_m3_hour_lag24'] = engineered['biogas_m3_hour'].shift(24)
    engineered['biogas_m3_hour_lag168'] = engineered['biogas_m3_hour'].shift(168)
    engineered['biogas_m3_hour_rolling_mean_24'] = engineered['biogas_m3_hour'].rolling(window=24).mean()
    # Calendar
    engineered['hour'] = engineered['timestamp'].dt.hour
    engineered['dayofweek'] = engineered['timestamp'].dt.dayofweek
    engineered['month'] = engineered['timestamp'].dt.month
    engineered['year'] = engineered['timestamp'].dt.year
    engineered = engineered.dropna()
    return engineered


def train_model(
    df_engineered: pd.DataFrame,
    test_size: float,
    xgb_params: dict,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[xgb.Booster, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    X = df_engineered.drop(columns=['biogas_m3_hour','timestamp'])
    y = df_engineered['biogas_m3_hour']
    # Capture the exact training feature order to reuse at prediction time
    feature_names: list[str] = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False, random_state=xgb_params.get('seed', 42))
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': xgb_params.get('learning_rate', 0.05),
        'max_depth': xgb_params.get('max_depth', 3),
        'min_child_weight': xgb_params.get('min_child_weight', 2),
        'subsample': xgb_params.get('subsample', 0.8),
        'colsample_bytree': xgb_params.get('colsample_bytree', 0.8),
        'seed': xgb_params.get('seed', 42),
        'nthread': -1,
    }
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds, evals=[(dtest,'eval')])
    return model, X_train, y_train, X_test, y_test, feature_names


def recursive_forecast(
    df_full: pd.DataFrame,
    model: xgb.Booster,
    train_feature_names: list[str],
    horizon: int = 24,
    overrides: dict | None = None,
) -> pd.DataFrame:
    df = df_full.copy()
    # Ensure categorical/boolean/object columns are numeric before recursion
    if 'feedstock' in df.columns:
        df['feedstock'] = df['feedstock'].astype('category').cat.codes
    for col in ['gas_leakage_flag','downtime_flag','is_holiday','rapid_temp_spike_flag','overheating_flag','solar_heating_flag','mixing_flag','low_quality_feed_flag']:
        if col in df.columns:
            df[col] = df[col].astype(int)
    for col in df.columns:
        if col not in ['timestamp','biogas_m3_hour'] and df[col].dtype == 'O':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('timestamp')

    last_ts = df['timestamp'].max()
    forecasts = []
    # Determine exogenous columns to seasonally fill (exclude engineered, target, timestamp)
    base_exog_cols = [
        c for c in df.columns
        if c not in [
            'timestamp','biogas_m3_hour','hour','dayofweek','month','year',
            'biogas_m3_hour_lag1','biogas_m3_hour_lag24','biogas_m3_hour_lag168','biogas_m3_hour_rolling_mean_24'
        ]
    ]

    for h in range(1, horizon + 1):
        future_ts = last_ts + pd.Timedelta(hours=h)
        # Base row from last known exogenous values
        row = {
            'timestamp': future_ts,
            'hour': future_ts.hour,
            'dayofweek': future_ts.dayofweek,
            'month': future_ts.month,
            'year': future_ts.year,
        }
        # Seasonal-naive for exogenous: use value from t-24 if available, else last known
        for col in base_exog_cols:
            prev_ts = future_ts - pd.Timedelta(hours=24)
            prev_mask = df['timestamp'] == prev_ts
            if prev_mask.any():
                row[col] = df.loc[prev_mask, col].iloc[-1]
            else:
                row[col] = df[col].iloc[-1]

        # Apply constant overrides (from sidebar) if provided
        if overrides:
            for k, v in overrides.items():
                if k in row:
                    row[k] = v

        # Append and recompute lags/rolling
        tmp = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        tmp['biogas_m3_hour_lag1'] = tmp['biogas_m3_hour'].shift(1)
        tmp['biogas_m3_hour_lag24'] = tmp['biogas_m3_hour'].shift(24)
        tmp['biogas_m3_hour_lag168'] = tmp['biogas_m3_hour'].shift(168)
        tmp['biogas_m3_hour_rolling_mean_24'] = tmp['biogas_m3_hour'].rolling(window=24).mean()
        tmp['hour'] = pd.to_datetime(tmp['timestamp']).dt.hour
        tmp['dayofweek'] = pd.to_datetime(tmp['timestamp']).dt.dayofweek
        tmp['month'] = pd.to_datetime(tmp['timestamp']).dt.month
        tmp['year'] = pd.to_datetime(tmp['timestamp']).dt.year

        # Ensure numeric types for XGBoost at the prediction step
        if 'feedstock' in tmp.columns:
            tmp['feedstock'] = tmp['feedstock'].astype('category').cat.codes
        for c in ['gas_leakage_flag','downtime_flag','is_holiday','rapid_temp_spike_flag','overheating_flag','solar_heating_flag','mixing_flag','low_quality_feed_flag']:
            if c in tmp.columns:
                tmp[c] = tmp[c].astype(int)
        for c in tmp.columns:
            if c not in ['timestamp','biogas_m3_hour'] and tmp[c].dtype == 'O':
                tmp[c] = pd.to_numeric(tmp[c], errors='coerce')
        # Keep NaNs for lags/rolling (XGBoost can handle NaNs); avoid zero-imputation that flattens predictions
        tmp = tmp.ffill()

        features = tmp.drop(columns=['biogas_m3_hour','timestamp']).iloc[[-1]]
        # Reorder and align prediction features to the exact training order
        features = features.reindex(columns=train_feature_names)
        dmat = xgb.DMatrix(features)
        pred = float(model.predict(dmat)[0])

        tmp.at[tmp.index[-1], 'biogas_m3_hour'] = pred
        df = tmp
        forecasts.append({'timestamp': future_ts, 'forecast_biogas_m3_hour': pred})

    return pd.DataFrame(forecasts)

def main():
    st.set_page_config(page_title='Biogas 24h Forecast', layout='wide')
    st.title('Biogas Production Forecast (Next 24 Hours)')

    csv_path = st.text_input('CSV path', value='biogas_hourly.csv')
    df_raw = load_data(csv_path)

    st.caption(f"Records loaded: {len(df_raw):,}")

    # Optional filters
    with st.expander('Preprocessing options'):
        remove_low_quality = st.checkbox('Remove low quality feed rows', value=True)
        if remove_low_quality and 'low_quality_feed_flag' in df_raw.columns:
            df_raw = df_raw[df_raw['low_quality_feed_flag'] == False]

    df_feat = prepare_features(df_raw)

    # Tabs: Forecast | Model settings | Feature overrides
    tab_forecast, tab_settings, tab_features = st.tabs(["Forecast", "Model settings", "Feature overrides"])

    with tab_settings:
        st.subheader('Training/test split')
        test_size = st.slider('Test size (fraction)', min_value=0.1, max_value=0.6, value=0.4, step=0.05)
        st.subheader('XGBoost hyperparameters')
        c1, c2, c3 = st.columns(3)
        with c1:
            learning_rate = st.number_input('learning_rate', min_value=0.005, max_value=0.5, value=0.05, step=0.005, format='%0.3f')
            max_depth = st.number_input('max_depth', min_value=1, max_value=12, value=3, step=1)
        with c2:
            min_child_weight = st.number_input('min_child_weight', min_value=1, max_value=10, value=2, step=1)
            subsample = st.number_input('subsample', min_value=0.1, max_value=1.0, value=0.8, step=0.05, format='%0.2f')
        with c3:
            colsample_bytree = st.number_input('colsample_bytree', min_value=0.1, max_value=1.0, value=0.8, step=0.05, format='%0.2f')
            seed = st.number_input('seed', min_value=0, max_value=99999, value=42, step=1)
        st.subheader('Training runtime')
        num_boost_round = st.number_input('num_boost_round', min_value=100, max_value=5000, value=800, step=50)
        early_stopping_rounds = st.number_input('early_stopping_rounds', min_value=10, max_value=200, value=50, step=5)

    # Defaults if user doesn't touch settings before first render
    if 'test_size' not in locals():
        test_size = 0.4
        learning_rate = 0.05
        max_depth = 3
        min_child_weight = 2
        subsample = 0.8
        colsample_bytree = 0.8
        seed = 42
        num_boost_round = 800
        early_stopping_rounds = 50

    xgb_params = {
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'seed': seed,
    }

    with st.spinner('Training model...'):
        model, X_train, y_train, X_test, y_test, feature_names = train_model(
            df_feat,
            test_size=float(test_size),
            xgb_params=xgb_params,
            num_boost_round=int(num_boost_round),
            early_stopping_rounds=int(early_stopping_rounds),
        )
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    residual_std = float(np.std(y_test.values - y_pred))

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    with st.expander('Model diagnostics', expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric('RMSE', f"{rmse:.3f}")
        c2.metric('MAE', f"{mae:.3f}")
        c3.metric('MAPE', f"{mape:.2%}")
        c4.metric('R2', f"{r2:.3f}")
        c5.metric('n_test', f"{len(y_test):,}")

    # Sidebar: horizon and confidence level
    horizon = st.sidebar.select_slider('Forecast horizon', options=[24, 48, 72], value=24)
    conf_level = st.sidebar.select_slider('Confidence level', options=[0.80, 0.90, 0.95], value=0.95)

    # Feature overrides page content
    overrides: dict | None = None
    with tab_features:
        st.subheader('Custom exogenous overrides')
        use_overrides = st.checkbox('Enable overrides', value=False)
        if use_overrides:
            overrides = {}
            numeric_candidates = [
                'feed_volume_kg','temp_C','pH','VFA_mg_L_HAc','alkalinity_mg_L_CaCO3','methane_pct',
                'TS_pct','VS_pct','CN_ratio','OLR_kg_TVS_m3_d','feed_quality_index'
            ]
            bool_candidates = [
                'solar_heating_flag','mixing_flag','gas_leakage_flag','downtime_flag','is_holiday',
                'rapid_temp_spike_flag','overheating_flag','low_quality_feed_flag'
            ]
            cnum1, cnum2 = st.columns(2)
            # Numeric sliders
            for i, col in enumerate([c for c in numeric_candidates if c in df_raw.columns]):
                col_min = float(df_raw[col].min())
                col_max = float(df_raw[col].max())
                col_default = float(df_raw[col].iloc[-1])
                if col_min == col_max:
                    # widen bounds slightly
                    col_min = col_min - 1.0
                    col_max = col_max + 1.0
                step = max((col_max - col_min) / 100.0, 0.0001)
                with (cnum1 if i % 2 == 0 else cnum2):
                    overrides[col] = st.slider(col, min_value=col_min, max_value=col_max, value=col_default, step=step)
            # Boolean toggles
            cbool1, cbool2 = st.columns(2)
            for j, col in enumerate([c for c in bool_candidates if c in df_raw.columns]):
                default_val = bool(int(df_raw[col].iloc[-1]))
                with (cbool1 if j % 2 == 0 else cbool2):
                    overrides[col] = 1 if st.checkbox(col, value=default_val) else 0
    st.subheader(f'{horizon}-hour Forecast')
    # Build minimal df with necessary columns for recursion (include last known target history)
    df_for_forecast = df_raw[['timestamp','biogas_m3_hour']].copy()
    # Merge back exogenous
    exo_cols = [c for c in df_raw.columns if c not in ['timestamp','biogas_m3_hour']]
    df_for_forecast = df_for_forecast.merge(df_raw[['timestamp'] + exo_cols], on='timestamp', how='left')

    forecast_df = recursive_forecast(df_for_forecast, model, feature_names, horizon=horizon, overrides=overrides)
    # Confidence intervals assuming constant residual variance
    z_map = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.96}
    z = z_map.get(conf_level, 1.96)
    forecast_df['lower'] = (forecast_df['forecast_biogas_m3_hour'] - z * residual_std).clip(lower=0)
    forecast_df['upper'] = forecast_df['forecast_biogas_m3_hour'] + z * residual_std
    # Removed tables per request; chart below visualizes the forecast
    # Plot with confidence band
    base = alt.Chart(forecast_df).encode(x='timestamp:T')
    band = base.mark_area(opacity=0.2).encode(y='lower:Q', y2='upper:Q')
    line = base.mark_line().encode(y='forecast_biogas_m3_hour:Q')
    st.altair_chart(band + line, use_container_width=True)

    # Download forecast CSV
    st.download_button('Download forecast (CSV)', data=forecast_df.to_csv(index=False).encode('utf-8'), file_name='forecast.csv', mime='text/csv')

    # Recent history chart for context
    st.subheader('Recent history (last 72 hours)')
    recent_hours = 72
    recent = df_raw.sort_values('timestamp').tail(recent_hours)
    st.line_chart(recent.set_index('timestamp')['biogas_m3_hour'])

    # Feature importance
    st.subheader('Feature importance')
    try:
        importance = model.get_score(importance_type='gain')
        if importance:
            imp_df = pd.DataFrame({
                'feature': list(importance.keys()),
                'gain': list(importance.values()),
            }).sort_values('gain', ascending=False).head(20)
            st.altair_chart(
                alt.Chart(imp_df).mark_bar().encode(x='gain:Q', y=alt.Y('feature:N', sort='-x')),
                use_container_width=True,
            )
        else:
            st.caption('Importance not available from model.')
    except Exception:
        st.caption('Could not compute feature importance for this model/run.')

    st.info('Run with: streamlit run streamlit_app.py')


if __name__ == '__main__':
    main()



