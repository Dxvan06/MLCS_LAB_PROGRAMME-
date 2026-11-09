
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Sample dataset
data = {
    "special_chars": [0, 1, 0, 5, 4, 0, 6, 1, 5, 0],
    "sql_keywords":  [0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    "input_length":  [10, 15, 20, 80, 70, 12, 90, 18, 85, 25],
    "label":         [0, 0, 0, 1, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Separate features and labels
X = df[["special_chars", "sql_keywords", "input_length"]]
y = df["label"]

# Create and train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Test new input records
test_data = pd.DataFrame({
    "special_chars": [0, 6, 1],
    "sql_keywords": [0, 1, 0],
    "input_length": [15, 100, 20]
})

predictions = model.predict(test_data)

print("MALICIOUS DATA INJECTION DETECTION")

for i, prediction in enumerate(predictions):
    result = "Malicious" if prediction == 1 else "Normal"

    print("\nInput Record:", i + 1)
    print("Special Characters:", test_data.iloc[i, 0])
    print("SQL Keywords:", test_data.iloc[i, 1])
    print("Input Length:", test_data.iloc[i, 2])
    print("Classification:", result)
