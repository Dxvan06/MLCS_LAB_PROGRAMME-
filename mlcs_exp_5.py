# ============================================================
# MCA EXPERIMENT 5
# Apply Support Vector Machine (SVM) to classify URLs
# into Legitimate and Malicious categories
# ============================================================

import os
import pandas as pd
import matplotlib

# Use a non-interactive backend so graphs can be saved
# without display-related warnings
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. FIND DATASET
# ============================================================

# Get the directory where this Python program is located
script_folder = os.path.dirname(
    os.path.abspath(__file__)
)

# Dataset is expected in the same directory
dataset_path = os.path.join(
    script_folder,
    "url_svm_dataset.csv"
)


print("=" * 60)
print("SVM URL CLASSIFICATION")
print("=" * 60)

print("\nDataset path:")
print(dataset_path)


# ============================================================
# 2. LOAD DATASET
# ============================================================

if not os.path.exists(dataset_path):

    print("\nERROR: Dataset not found!")

    print("\nPlease place:")
    print("url_svm_dataset.csv")

    print("\nInside:")
    print(script_folder)

    exit()


df = pd.read_csv(dataset_path)

print("\nDataset successfully loaded!")


# ============================================================
# 3. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 10 Rows:")
print(df.head(10))


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

if "URL" not in df.columns or "Label" not in df.columns:

    print("\nERROR:")
    print("Dataset must contain the following columns:")
    print("URL")
    print("Label")

    exit()


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())


# Remove missing URL and Label values
df = df.dropna(
    subset=["URL", "Label"]
)


# ============================================================
# 6. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

class_counts = df["Label"].value_counts()

print(class_counts)


# ============================================================
# 7. CLASS DISTRIBUTION GRAPH
# ============================================================

plt.figure(figsize=(7, 5))

class_counts.plot(
    kind="bar"
)

plt.title(
    "URL Dataset - Class Distribution"
)

plt.xlabel(
    "URL Class"
)

plt.ylabel(
    "Number of URLs"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

class_graph_path = os.path.join(
    script_folder,
    "url_class_distribution.png"
)

plt.savefig(
    class_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nClass distribution graph saved:"
)

print(class_graph_path)


# ============================================================
# 8. PREPARE DATA
# ============================================================

# Input feature
X = df["URL"].astype(str)

# Target label
y = (
    df["Label"]
    .astype(str)
    .str.lower()
    .str.strip()
)


# ============================================================
# 9. ENCODE LABELS
# ============================================================

# Legitimate = 0
# Malicious  = 1

y = y.map({
    "legitimate": 0,
    "malicious": 1
})


# Remove unknown labels
valid_rows = y.notna()

X = X[valid_rows]
y = y[valid_rows]


# Convert to integer
y = y.astype(int)


print("\n" + "=" * 60)
print("LABEL ENCODING")
print("=" * 60)

print("\n0 = Legitimate")
print("1 = Malicious")


# ============================================================
# 10. DATASET SUMMARY
# ============================================================

print("\nTotal URLs:")
print(len(X))

print("\nLegitimate URLs:")
print(sum(y == 0))

print("\nMalicious URLs:")
print(sum(y == 1))


# ============================================================
# 11. TRAIN TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:")
print(len(X_train))

print("\nTesting samples:")
print(len(X_test))


# ============================================================
# 12. TF-IDF FEATURE EXTRACTION
# ============================================================

print("\n" + "=" * 60)
print("TF-IDF FEATURE EXTRACTION")
print("=" * 60)


# Character-level TF-IDF is suitable for URLs
# because URL character patterns are useful
# for distinguishing legitimate and malicious URLs.

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    min_df=2,
    max_features=50000
)


# Fit only on training data
X_train_tfidf = vectorizer.fit_transform(
    X_train
)


# Transform test data
X_test_tfidf = vectorizer.transform(
    X_test
)


print("\nTraining TF-IDF shape:")
print(X_train_tfidf.shape)

print("\nTesting TF-IDF shape:")
print(X_test_tfidf.shape)


# ============================================================
# 13. CREATE SVM MODEL
# ============================================================

print("\n" + "=" * 60)
print("SVM MODEL TRAINING")
print("=" * 60)


svm_model = SVC(
    kernel="linear",
    C=1.0,
    random_state=42
)


# Train the model
svm_model.fit(
    X_train_tfidf,
    y_train
)


print("\nSVM training completed successfully!")


# ============================================================
# 14. MAKE PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("MODEL PREDICTION")
print("=" * 60)


y_pred = svm_model.predict(
    X_test_tfidf
)


# ============================================================
# 15. CALCULATE PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ============================================================
# 16. DISPLAY PERFORMANCE
# ============================================================

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)


print(
    f"\nAccuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1-Score : {f1 * 100:.2f}%"
)


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)


print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Legitimate",
            "Malicious"
        ],
        zero_division=0
    )
)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)


cm = confusion_matrix(
    y_test,
    y_pred
)


print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 19. CONFUSION MATRIX GRAPH
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Legitimate",
        "Malicious"
    ]
)


fig, ax = plt.subplots(
    figsize=(7, 6)
)


disp.plot(
    ax=ax
)


plt.title(
    "SVM URL Classification - Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

plt.tight_layout()


confusion_graph_path = os.path.join(
    script_folder,
    "svm_confusion_matrix.png"
)


plt.savefig(
    confusion_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "\nConfusion matrix graph saved:"
)

print(confusion_graph_path)


# ============================================================
# 20. CUSTOM URL TESTING
# ============================================================

print("\n" + "=" * 60)
print("CUSTOM URL PREDICTION")
print("=" * 60)


test_urls = [

    "https://www.google.com",

    "https://www.microsoft.com/login",

    "https://www.amazon.com/products",

    "https://www.github.com",

    "https://www.python.org/docs",

    "http://paypal-security.xyz/login/verify-account",

    "http://free-prize.xyz/login.php",

    "http://google-verify.tk/secure/update-password",

    "http://bank-security.top/account/confirm",

    "http://wallet-connect.click/wallet/connect"

]


# Convert custom URLs into TF-IDF
test_features = vectorizer.transform(
    test_urls
)


# Make predictions
custom_predictions = svm_model.predict(
    test_features
)


# ============================================================
# 21. DISPLAY CUSTOM URL RESULTS
# ============================================================

results = []


for url, prediction in zip(
    test_urls,
    custom_predictions
):

    if prediction == 0:

        result = "LEGITIMATE"

    else:

        result = "MALICIOUS"


    print("\nURL:")
    print(url)

    print("Prediction:")
    print(result)


    results.append({
        "URL": url,
        "Prediction": result
    })


# ============================================================
# 22. SAVE CUSTOM PREDICTIONS
# ============================================================

results_df = pd.DataFrame(
    results
)


prediction_path = os.path.join(
    script_folder,
    "svm_url_predictions.csv"
)


results_df.to_csv(
    prediction_path,
    index=False
)


print(
    "\nCustom URL predictions saved:"
)

print(prediction_path)


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL PROJECT SUMMARY")
print("=" * 60)


print(
    f"""
Dataset Size       : {len(X)} URLs

Training Samples   : {len(X_train)}

Testing Samples    : {len(X_test)}

Legitimate URLs    : {sum(y == 0)}

Malicious URLs     : {sum(y == 1)}

Algorithm          : Support Vector Machine (SVM)

Kernel             : Linear

C Parameter        : 1.0

Feature Extraction : Character-level TF-IDF

N-Gram Range       : 2 - 5

Accuracy            : {accuracy * 100:.2f}%

Precision           : {precision * 100:.2f}%

Recall              : {recall * 100:.2f}%

F1-Score            : {f1 * 100:.2f}%
"""
)


print("=" * 60)
print("OUTPUT FILES")
print("=" * 60)

print(
    "\n1. url_class_distribution.png"
)

print(
    "2. svm_confusion_matrix.png"
)

print(
    "3. svm_url_predictions.csv"
)


print("\n" + "=" * 60)
print("PROGRAM COMPLETED SUCCESSFULLY")
print("=" * 60)