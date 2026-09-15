# ============================================================
# DDoS Prediction Using ARIMA Time-Series Forecasting
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# ============================================================
# 1. LOAD DATASET
# ============================================================

FILE_NAME = "network_traffic.csv"

data = pd.read_csv(FILE_NAME)

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

print("\n================ DATASET INFORMATION ================\n")
print("Columns found:")
print(data.columns.tolist())

print("\nFirst 5 records:")
print(data.head())

# ============================================================
# 2. IDENTIFY TIME AND TRAFFIC COLUMNS
# ============================================================

# Assume first column is time/date
time_column = data.columns[0]

# Find numeric columns
numeric_columns = data.select_dtypes(
    include=[np.number]
).columns.tolist()

if len(numeric_columns) == 0:
    print("\nERROR: No numeric traffic column found.")
    print("Your dataset must contain a numeric network traffic column.")
    exit()

# Select first numeric column as traffic
traffic_column = numeric_columns[0]

print("\nTime column selected   :", time_column)
print("Traffic column selected:", traffic_column)

# ============================================================
# 3. CONVERT TIME COLUMN
# ============================================================

data[time_column] = pd.to_datetime(
    data[time_column],
    errors="coerce"
)

# Remove invalid dates
data = data.dropna(subset=[time_column])

# Sort according to time
data = data.sort_values(time_column)

# Set time as index
data.set_index(time_column, inplace=True)

# ============================================================
# 4. CLEAN TRAFFIC DATA
# ============================================================

data[traffic_column] = pd.to_numeric(
    data[traffic_column],
    errors="coerce"
)

# Remove missing values
traffic = data[traffic_column].dropna()

# Remove negative values
traffic = traffic[traffic >= 0]

print("\nNumber of observations:", len(traffic))

print("\nTraffic statistics:")
print(traffic.describe())

# ============================================================
# 5. CHECK DATA SIZE
# ============================================================

if len(traffic) < 20:
    print("\nERROR: Not enough data for ARIMA forecasting.")
    print("Please provide at least 20 observations.")
    exit()

# ============================================================
# 6. CALCULATE NORMAL TRAFFIC THRESHOLD
# ============================================================

mean_traffic = traffic.mean()
std_traffic = traffic.std()

# 3-sigma threshold
threshold = mean_traffic + (3 * std_traffic)

print("\n================ DDoS THRESHOLD ================\n")

print("Average traffic :", round(mean_traffic, 2))
print("Standard deviation:", round(std_traffic, 2))
print("DDoS threshold  :", round(threshold, 2))

# ============================================================
# 7. BUILD ARIMA MODEL
# ============================================================

print("\n================ ARIMA MODEL ================\n")

# ARIMA(p,d,q)
#
# p = autoregressive component
# d = differencing
# q = moving average component

p = 5
d = 1
q = 2

print("ARIMA order:", (p, d, q))

try:

    model = ARIMA(
        traffic,
        order=(p, d, q)
    )

    model_fit = model.fit()

except Exception as e:

    print("\nARIMA model error:")
    print(e)

    # Simpler model if ARIMA(5,1,2) fails
    print("\nTrying simpler ARIMA(1,1,1)...")

    model = ARIMA(
        traffic,
        order=(1, 1, 1)
    )

    model_fit = model.fit()

# ============================================================
# 8. DISPLAY MODEL SUMMARY
# ============================================================

print("\nARIMA model successfully trained.")

print("\nModel Summary:")
print(model_fit.summary())

# ============================================================
# 9. FORECAST FUTURE NETWORK TRAFFIC
# ============================================================

FORECAST_STEPS = 30

print("\n================ FORECAST ================\n")

forecast = model_fit.forecast(
    steps=FORECAST_STEPS
)

# Convert forecast to positive values
forecast = np.maximum(forecast, 0)

# ============================================================
# 10. CREATE FUTURE TIMESTAMPS
# ============================================================

# Try to determine the time interval
time_difference = traffic.index.to_series().diff().median()

if pd.isna(time_difference):
    time_difference = pd.Timedelta(minutes=1)

future_dates = pd.date_range(
    start=traffic.index[-1] + time_difference,
    periods=FORECAST_STEPS,
    freq=time_difference
)

# ============================================================
# 11. CREATE FORECAST DATAFRAME
# ============================================================

forecast_data = pd.DataFrame({
    "Timestamp": future_dates,
    "Predicted_Traffic": forecast.values
})

print(forecast_data.to_string(index=False))

# ============================================================
# 12. DETECT POTENTIAL DDoS
# ============================================================

forecast_data["DDoS_Status"] = np.where(
    forecast_data["Predicted_Traffic"] > threshold,
    "POTENTIAL DDoS",
    "NORMAL"
)

print("\n================ DDoS PREDICTION ================\n")

for index, row in forecast_data.iterrows():

    timestamp = row["Timestamp"]
    predicted = row["Predicted_Traffic"]
    status = row["DDoS_Status"]

    print(
        f"{timestamp} | "
        f"Predicted Traffic: {predicted:.2f} | "
        f"Status: {status}"
    )

# ============================================================
# 13. COUNT POTENTIAL DDoS EVENTS
# ============================================================

ddos_count = (
    forecast_data["DDoS_Status"] == "POTENTIAL DDoS"
).sum()

print("\n===================================================")
print("                 FINAL RESULT")
print("===================================================")

if ddos_count > 0:

    print("WARNING: Potential DDoS activity predicted!")
    print("Number of high-traffic predictions:", ddos_count)

else:

    print("No potential DDoS activity predicted.")

print("===================================================")

# ============================================================
# 14. SAVE FORECAST RESULTS
# ============================================================

forecast_data.to_csv(
    "ddos_forecast_results.csv",
    index=False
)

print("\nForecast results saved to:")
print("ddos_forecast_results.csv")

# ============================================================
# 15. VISUALIZATION
# ============================================================

plt.figure(figsize=(14, 7))

# Historical traffic
plt.plot(
    traffic.index,
    traffic.values,
    label="Historical Network Traffic"
)

# Forecast
plt.plot(
    forecast_data["Timestamp"],
    forecast_data["Predicted_Traffic"],
    linestyle="--",
    label="ARIMA Forecast"
)

# DDoS threshold
plt.axhline(
    y=threshold,
    linestyle=":",
    label="DDoS Threshold"
)

plt.xlabel("Time")
plt.ylabel("Network Traffic")

plt.title(
    "Network Traffic Forecast and Potential DDoS Detection"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# ============================================================
# END OF PROGRAM
# ============================================================