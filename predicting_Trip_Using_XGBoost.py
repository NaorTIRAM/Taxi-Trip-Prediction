import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


# Read the CSV file
data = pd.read_csv('new_data.csv')
holiday = pd.read_csv('holiday_calendar.csv')
# Filter rows where the country code is 'US'
filtered_data = holiday[holiday['Country Code'] == 'US'][['Date']] 

holidays = filtered_data

# Preprocess the data
data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
data['date'] = (data['tpep_pickup_datetime'] - pd.to_datetime('2000-01-01')).dt.days
data['hour'] = data['tpep_pickup_datetime'].dt.hour
data_agg = data.groupby(['date', 'hour']).size().reset_index(name='num_trips')

# Add holiday and weekend features
data_agg['date_datetime'] = pd.to_datetime('2000-01-01') + pd.to_timedelta(data_agg['date'], unit='D')
data_agg['holiday'] = data_agg['date_datetime'].isin(holidays).astype(int)
data_agg['weekend'] = (data_agg['date_datetime'].dt.dayofweek >= 5).astype(int)
data_agg['day_of_week'] = data_agg['date_datetime'].dt.dayofweek

# Prepare data for training and testing
X = data_agg[['date', 'hour', 'holiday', 'weekend', 'day_of_week']]
y = data_agg['num_trips']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the XGBoost model with the best parameters
model = XGBRegressor(random_state=42, colsample_bytree=0.8, learning_rate=0.2, max_depth=9, n_estimators=400, subsample=0.80)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'XGBoost (optimized):')
print(f'  Mean Squared Error: {mse:.4f}')
print(f'  Mean Absolute Error: {mae:.4f}')
print(f'  R-squared Score: {r2:.4f}\n')

# Predict for a specific date and range of hours
def predict_for_date_and_hours(model, scaler, date, start_hour, end_hour):
    input_date = (date - pd.to_datetime('2000-01-01')).days
    input_data = pd.DataFrame({'date': [input_date] * (end_hour - start_hour),
                               'hour': list(range(start_hour, end_hour)),
                               'holiday': [date in holidays] * (end_hour - start_hour),
                               'weekend': [(date.weekday() >= 5)] * (end_hour - start_hour),
                               'day_of_week': [date.weekday()] * (end_hour - start_hour)})
    input_data_scaled = scaler.transform(input_data)
    predictions = model.predict(input_data_scaled)
    total_trips = predictions.sum()
    return total_trips

date = pd.to_datetime('2023-01-02')
start_hour = 10
end_hour = 11
predicted_trips = predict_for_date_and_hours(model, scaler, date, start_hour, end_hour)
print(f'Predicted number of trips for {date} between hours {start_hour} and {end_hour}: {predicted_trips:.2f}')
