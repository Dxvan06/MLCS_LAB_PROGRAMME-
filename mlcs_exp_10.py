
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Network event dataset
data = {
    "duration": [10, 12, 8, 15, 9, 200, 250, 180, 300, 220],
    "packets": [20, 25, 18, 22, 19, 500, 600, 450, 700, 550],
    "bytes": [300, 350, 280, 320, 290, 5000, 6000, 4500, 7000, 5500]
}

df = pd.DataFrame(data)

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(df)

# Apply K-Means clustering
model = KMeans(n_clusters=2, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X)

# Identify abnormal cluster using average byte count
abnormal_cluster = df.groupby("Cluster")["bytes"].mean().idxmax()

df["Category"] = df["Cluster"].apply(
    lambda x: "Abnormal" if x == abnormal_cluster else "Normal"
)

print("NETWORK EVENT CLASSIFICATION")
print(df)
