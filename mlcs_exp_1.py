import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# -------------------------------
# Generate Sample Network Traffic Data
# -------------------------------

# Set random seed for reproducible results
np.random.seed(42)

# Create date range
dates = pd.date_range(start="2024-01-01", periods=120, freq="D")

# Create trend component
trend = np.linspace(1000, 1800, 120)

# Create weekly seasonal component
seasonal = 200 * np.sin(np.arange(120) * 2 * np.pi / 7)

# Create random noise
noise = np.random.normal(0, 50, 120)

# Generate network traffic
traffic = trend + seasonal + noise

# Create DataFrame
data = pd.DataFrame({
    "Date": dates,
    "Network_Traffic": traffic
})

# Convert Date column to datetime
data["Date"] = pd.to_datetime(data["Date"])

# Set Date as index
data.set_index("Date", inplace=True)

# -------------------------------
# Display Dataset
# -------------------------------

print("=" * 50)
print("FIRST FIVE ROWS OF DATASET")
print("=" * 50)
print(data.head())

# -------------------------------
# Time Series Decomposition
# -------------------------------

result = seasonal_decompose(
    data["Network_Traffic"],
    model="additive",
    period=7
)

# -------------------------------
# Display Components
# -------------------------------

print("\n" + "=" * 50)
print("TREND COMPONENT")
print("=" * 50)
print(result.trend.head(10))

print("\n" + "=" * 50)
print("SEASONAL COMPONENT")
print("=" * 50)
print(result.seasonal.head(10))

print("\n" + "=" * 50)
print("RESIDUAL COMPONENT")
print("=" * 50)
print(result.resid.head(10))

# -------------------------------
# Plot Original Time Series
# -------------------------------

plt.figure(figsize=(12, 5))
plt.plot(
    data.index,
    data["Network_Traffic"],
    color="blue",
    linewidth=2,
    label="Network Traffic"
)

plt.title("Original Network Traffic")
plt.xlabel("Date")
plt.ylabel("Traffic")
plt.legend()
plt.grid(True)

plt.savefig("original_network_traffic.png", dpi=300)
plt.close()

# -------------------------------
# Plot Decomposition Components
# -------------------------------

fig = result.plot()
fig.set_size_inches(12, 9)

plt.suptitle(
    "Time Series Decomposition of Network Traffic",
    fontsize=16
)

plt.savefig("time_series_decomposition.png", dpi=300)
plt.close()

print("\n" + "=" * 50)
print("OUTPUT GENERATED SUCCESSFULLY")
print("=" * 50)
print("1. original_network_traffic.png")
print("2. time_series_decomposition.png")
print("=" * 50)
print("Experiment completed successfully.")