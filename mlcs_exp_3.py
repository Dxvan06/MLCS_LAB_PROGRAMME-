# ============================================================
# EXPERIMENT 3
# Cyber-Attack Classification Using Random Forest
# Ensemble Learning Method
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

FILE_NAME = "cyber_attacks.csv"

data = pd.read_csv(FILE_NAME)

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

print("\n================ DATASET INFORMATION ================\n")

print("Dataset shape:", data.shape)

print("\nColumn names:")
print(data.columns.tolist())

print("\nFirst 5 records:")
print(data.head())


# ------------------------------------------------------------
# 2. FIND TARGET COLUMN
# ------------------------------------------------------------

possible_targets = [
    "Label",
    "label",
    "Attack",
    "attack",
    "Class",
    "class",
    "Target",
    "target",
    "Category",
    "category"
]

target_column = None

for column in possible_targets:
    if column in data.columns:
        target_column = column
        break

# If no target is found, use last column
if target_column is None:
    target_column = data.columns[-1]

print("\nTarget column:", target_column)


# ------------------------------------------------------------
# 3. CLEAN DATA
# ------------------------------------------------------------

data = data.replace([np.inf, -np.inf], np.nan)

data = data.dropna()

print("\nDataset after cleaning:", data.shape)


# ------------------------------------------------------------
# 4. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

X = data.drop(columns=[target_column])

y = data[target_column]


# ------------------------------------------------------------
# 5. REMOVE ID / TIME COLUMNS IF PRESENT
# ------------------------------------------------------------

id_columns = [
    "id",
    "ID",
    "Flow ID",
    "Flow_ID",
    "Timestamp",
    "Time"
]

for column in id_columns:

    if column in X.columns:
        X = X.drop(columns=[column])


# ------------------------------------------------------------
# 6. ENCODE CATEGORICAL FEATURES
# ------------------------------------------------------------

print("\nEncoding categorical features...")

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns

for column in categorical_columns:

    encoder = LabelEncoder()

    X[column] = encoder.fit_transform(
        X[column].astype(str)
    )


# ------------------------------------------------------------
# 7. ENCODE TARGET LABELS
# ------------------------------------------------------------

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    y.astype(str)
)

print("\nAttack classes:")

for i, label in enumerate(label_encoder.classes_):

    print(i, "=", label)


# ------------------------------------------------------------
# 8. SPLIT DATASET
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))

print("Testing samples :", len(X_test))


# ------------------------------------------------------------
# 9. CREATE RANDOM FOREST MODEL
# ------------------------------------------------------------

print("\n================ RANDOM FOREST ================\n")

model = RandomForestClassifier(

    n_estimators=100,

    criterion="gini",

    max_depth=None,

    random_state=42,

    n_jobs=-1
)


# ------------------------------------------------------------
# 10. TRAIN MODEL
# ------------------------------------------------------------

print("Training Random Forest...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ------------------------------------------------------------
# 11. MAKE PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(
    X_test
)


# ------------------------------------------------------------
# 12. CALCULATE ACCURACY
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n================ MODEL RESULT ================\n")

print(
    "Random Forest Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ------------------------------------------------------------
# 13. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)


# ------------------------------------------------------------
# 14. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:\n")

print(cm)


# ------------------------------------------------------------
# 15. SAVE CONFUSION MATRIX GRAPH
# ------------------------------------------------------------

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=label_encoder.classes_
)

disp.plot()

plt.title(
    "Random Forest Cyber-Attack Classification"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nConfusion matrix graph saved as: "
    "confusion_matrix.png"
)


# ------------------------------------------------------------
# 16. FEATURE IMPORTANCE
# ------------------------------------------------------------

importance = model.feature_importances_


feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": importance

})


feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False
)


print("\n================ FEATURE IMPORTANCE ================\n")

print(
    feature_importance.head(10).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 17. SAVE FEATURE IMPORTANCE GRAPH
# ------------------------------------------------------------

top_features = feature_importance.head(10)


plt.figure(
    figsize=(10, 6)
)


plt.barh(

    top_features["Feature"],

    top_features["Importance"]

)


plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 10 Features Used for Cyber-Attack Classification"
)


plt.gca().invert_yaxis()


plt.tight_layout()


plt.savefig(

    "feature_importance.png",

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(
    "\nFeature importance graph saved as: "
    "feature_importance.png"
)


# ------------------------------------------------------------
# 18. SAVE PREDICTIONS
# ------------------------------------------------------------

results = pd.DataFrame({

    "Actual":
        label_encoder.inverse_transform(y_test),

    "Predicted":
        label_encoder.inverse_transform(y_pred)

})


results.to_csv(

    "cyber_attack_predictions.csv",

    index=False

)


print(
    "\nPredictions saved to: "
    "cyber_attack_predictions.csv"
)


# ------------------------------------------------------------
# 19. FINAL RESULT
# ------------------------------------------------------------

print("\n===================================================")

print(
    "              FINAL RESULT"
)

print("===================================================")

print(
    "Model: Random Forest Ensemble Classifier"
)

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print(
    "Number of trees:",
    model.n_estimators
)

print("===================================================")

print("\nFiles generated:")

print("1. confusion_matrix.png")

print("2. feature_importance.png")

print("3. cyber_attack_predictions.csv")

print("\nProgram completed successfully!")