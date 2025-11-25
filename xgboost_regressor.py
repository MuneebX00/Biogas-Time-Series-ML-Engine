# preprocess the data
# convert timestamp to datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# Load the data into CSV file
df = pd.read_csv('biogas_hourly.csv')
df.head()

df['timestamp'] = pd.to_datetime(df['timestamp'])

# Remove low quality feed flag rows to improve accuracy
df = df[df['low_quality_feed_flag'] == False]


# Sort data by timestamp to ensure correct lag and rolling calculations
df = df.sort_values(by='timestamp')

# Feature Engineering
# Lag Features (e.g., lag of 1, 24, and 168 hours for biogas production)
df['biogas_m3_hour_lag1'] = df['biogas_m3_hour'].shift(1)
df['biogas_m3_hour_lag24'] = df['biogas_m3_hour'].shift(24)
df['biogas_m3_hour_lag168'] = df['biogas_m3_hour'].shift(168)

# Rolling Statistics (e.g., 24-hour rolling mean of biogas production)
df['biogas_m3_hour_rolling_mean_24'] = df['biogas_m3_hour'].rolling(window=24).mean()

# Date/Time Features
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['year'] = df['timestamp'].dt.year


# Assuming 'feedstock' is categorical, encode it
df['feedstock'] = df['feedstock'].astype('category').cat.codes

# Handle boolean flags by converting to int
bool_cols= ['gas_leakage_flag','downtime_flag','is_holiday','rapid_temp_spike_flag','overheating_flag']
for col in bool_cols:
    df[col] = df[col].astype(int)

# Drop rows with NaN values created by lag and rolling features
df.dropna(inplace=True)

# Features and target(assuming predict 'biogas_m3_hour')
X = df.drop(columns=['biogas_m3_hour','timestamp'])
y = df['biogas_m3_hour']

#Split into 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4 , shuffle=False, random_state=42) # shuffle=False for time series

#create xgboost datasets
train_data = xgb.DMatrix(X_train, label=y_train)
test_data = xgb.DMatrix(X_test, label=y_test)

#parameters for xgboost(regression)
params= {
    'objective':"reg:squarederror",
    'eval_metric': 'rmse',
    'n_estimators':2000,
    'learning_rate':0.05,
    'max_depth':3,
    'min_child_weight':2,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'random_state':42,
    'n_jobs':-1,
}

# Train the model
xgb_model = xgb.train(params, train_data, num_boost_round=params['n_estimators'], early_stopping_rounds=50,
                  evals=[(test_data, 'test')])

# Make predictions
y_pred = xgb_model.predict(test_data)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)


print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")