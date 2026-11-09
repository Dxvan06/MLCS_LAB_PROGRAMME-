
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Wireless sensor network dataset
data = {
    "hour": [10, 11, 12, 9, 14, 2, 3, 1, 4, 2],
    "ip_last": [10, 12, 15, 20, 18, 200, 210, 220, 230, 240],
    "event_type": [
        0, 0, 0, 0, 0, 1, 1, 2, 2, 1
    ],
    "label": [
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

# Select contextual features
X = df[["hour", "ip_last", "event_type"]]
y = df["label"]

# Train Decision Tree model
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X, y)

# Test new wireless sensor events
test_data = pd.DataFrame({
    "hour": [11, 2, 3],
    "ip_last": [15, 210, 230],
    "event_type": [0, 1, 2]
})

predictions = model.predict(test_data)

print("WIRELESS SENSOR EVENT DETECTION")

for i, prediction in enumerate(predictions):
    result = "Malicious" if prediction == 1 else "Normal"

    print("\nEvent:", i + 1)
    print("Timestamp Hour:", test_data.iloc[i]["hour"])
    print("Source IP Last Octet:", test_data.iloc[i]["ip_last"])
    print("Event Type:", test_data.iloc[i]["event_type"])
    print("Classification:", result)
