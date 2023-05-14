# Taxi-Trip-Prediction
This notebook predicts the number of taxi trips in New York City in specifit time during specific hours, using an XGBoost model
you can download the data from here: https://drive.google.com/file/d/1_Ns1QLRw0XPK2TueqtAmoo8-WKNcEG72/view?usp=sharing
or here: https://www.kaggle.com/datasets/microize/newyork-yellow-taxi-trip-data-2020-2019

This code reads taxi trip data and holiday calendar data from CSV files, preprocesses the data, trains an XGBoost model to predict the number of taxi trips, evaluates the model's performance, and makes predictions for specific dates and hours. Here is a step-by-step explanation of the code:

1.Import necessary libraries: pandas for data manipulation, scikit-learn for preprocessing, train-test splitting, and performance metrics, and XGBoost for the regression model.

2.Read the taxi trip data (new_data.csv) and holiday calendar data (holiday_calendar.csv) using pandas.

3.Filter the holiday calendar data to keep only the rows with country code 'US' and store the 'Date' column in the filtered_data variable.

4.Preprocess the taxi trip data: convert the pickup datetime column to pandas datetime objects, calculate the number of days since January 1, 2000, and extract the hour from the pickup datetime.

5.Aggregate the taxi trip data by date and hour, calculating the total number of trips in each time window.

6.Add holiday, weekend, and day of the week features to the aggregated data.

7.Prepare the data for training and testing by splitting it into input features (X) and target values (y).

8.Scale the input features using a MinMaxScaler.

9.Train an XGBoost model with the best hyperparameters found earlier.

10.Evaluate the trained model using mean squared error, mean absolute error, and R-squared score.

11.Define a function predict_for_date_and_hours() that takes a trained model, scaler, date, and start and end hours as input, and returns the predicted total number of trips in the specified time window.

12. Use the predict_for_date_and_hours() function to make a prediction for a specific date and range of hours. Print the predicted number of trips for the specified time window.


The result i got:
XGBoost (optimized):
  Mean Squared Error: 43278.6065
  Mean Absolute Error: 140.8754
  R-squared Score: 0.9929


