# ============================================================
# EXPERIMENT 4
# Logistic Regression for URL Classification
# Legitimate vs Malicious URLs
# ============================================================

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ------------------------------------------------------------
# 1. CREATE / LOAD DATASET
# ------------------------------------------------------------

FILE_NAME = "urls.csv"

try:
    data = pd.read_csv(FILE_NAME)

except FileNotFoundError:

    print("\nDataset not found.")
    print("Creating a sample URL dataset...\n")

    urls = [
        "https://www.google.com",
        "https://www.microsoft.com",
        "https://www.amazon.com",
        "https://www.wikipedia.org",
        "https://www.github.com",
        "https://www.linkedin.com",
        "https://www.apple.com",
        "https://www.ibm.com",
        "https://www.python.org",
        "https://www.amazon.in",

        "http://secure-login-example.com",
        "http://verify-account-example.com",
        "http://bank-login-example.com",
        "http://update-password-example.com",
        "http://free-prize-example.com",
        "http://login-confirm-example.com",
        "http://account-verify-example.com",
        "http://security-alert-example.com",
        "http://payment-update-example.com",
        "http://claim-reward-example.com",

        "https://www.netflix.com",
        "https://www.adobe.com",
        "https://www.nasa.gov",
        "https://www.cisco.com",
        "https://www.oracle.com",
        "https://www.intel.com",
        "https://www.microsoft.com/en-in",
        "https://www.google.com/search",
        "https://github.com/login",
        "https://stackoverflow.com",

        "http://192.168.1.1/login",
        "http://45.23.67.89/verify",
        "http://91.45.23.11/account",
        "http://185.23.44.91/login.php",
        "http://103.45.67.89/update",
        "http://78.34.12.56/security",
        "http://66.45.23.12/bank",
        "http://55.23.89.12/login",
        "http://34.56.78.90/verify",
        "http://23.45.67.89/password"
    ]

    labels = [
        "legitimate",
        "legitimate",
        "legitimate",
        "legitimate",
        "legitimate",
        "legitimate",
        "legitimate",
        "legitimate",
        "legitimate",
        "legitimate",

        "malicious",
        "malicious",
        "malicious",
        "malicious",
        "malicious",
        "malicious",
        "malicious",
        "malicious",
        "malicious",
        "malicious",

        "legitimate",
        "legitimate",
        "legitimate",
        "legitimate",
        "legitimate",
        "legitimate",
        "legitimate",
        "legitimate",
        "legitimate",
        "legitimate",

        "malicious",
        "malicious",
        "malicious",
        "malicious",
        "malicious",
        "malicious",
        "malicious",
        "malicious",
        "malicious",
        "malicious"
    ]

    data = pd.DataFrame({
        "URL": urls,
        "Label": labels
    })

    data.to_csv(FILE_NAME, index=False)

    print("Sample dataset created:", FILE_NAME)


# ------------------------------------------------------------
# 2. DISPLAY DATASET
# ------------------------------------------------------------

print("\n================ DATASET INFORMATION ================\n")

print("Dataset shape:", data.shape)

print("\nColumn names:")
print(data.columns.tolist())

print("\nFirst 10 records:")
print(data.head(10))


# ------------------------------------------------------------
# 3. FIND URL AND LABEL COLUMNS
# ------------------------------------------------------------

url_column = None
label_column = None

for column in data.columns:

    if column.lower() in ["url", "urls", "link", "website"]:
        url_column = column

    if column.lower() in [
        "label",
        "class",
        "target",
        "category"
    ]:
        label_column = column


if url_column is None:
    url_column = data.columns[0]

if label_column is None:
    label_column = data.columns[-1]


print("\nURL column:", url_column)
print("Label column:", label_column)


# ------------------------------------------------------------
# 4. CLEAN DATA
# ------------------------------------------------------------

data = data.dropna()

data[url_column] = data[url_column].astype(str)

data[label_column] = data[label_column].astype(str).str.lower()


# ------------------------------------------------------------
# 5. FEATURE EXTRACTION FUNCTION
# ------------------------------------------------------------

def extract_features(url):

    url = str(url)

    features = {}

    # URL length
    features["URL_Length"] = len(url)

    # Number of dots
    features["Dots"] = url.count(".")

    # Number of hyphens
    features["Hyphens"] = url.count("-")

    # Number of slashes
    features["Slashes"] = url.count("/")

    # Number of digits
    features["Digits"] = sum(
        char.isdigit() for char in url
    )

    # Number of special characters
    features["Special_Characters"] = len(
        re.findall(r"[^a-zA-Z0-9]", url)
    )

    # Presence of @
    features["Has_At"] = int("@" in url)

    # Presence of IP address
    features["Has_IP"] = int(
        bool(
            re.search(
                r"(\d{1,3}\.){3}\d{1,3}",
                url
            )
        )
    )

    # HTTPS usage
    features["Uses_HTTPS"] = int(
        url.lower().startswith("https://")
    )

    # Number of query parameters
    features["Query_Parameters"] = url.count("?")

    return features


# ------------------------------------------------------------
# 6. EXTRACT FEATURES
# ------------------------------------------------------------

feature_data = data[url_column].apply(
    extract_features
)

X = pd.DataFrame(
    feature_data.tolist()
)

print("\n================ EXTRACTED FEATURES ================\n")

print(X.head())


# ------------------------------------------------------------
# 7. CONVERT LABELS
# ------------------------------------------------------------

def convert_label(label):

    if label in [
        "malicious",
        "malware",
        "phishing",
        "bad",
        "1"
    ]:
        return 1

    return 0


y = data[label_column].apply(
    convert_label
)


print("\nLabel distribution:")

print(
    y.value_counts()
)


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
# 9. FEATURE SCALING
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ------------------------------------------------------------
# 10. CREATE LOGISTIC REGRESSION MODEL
# ------------------------------------------------------------

print("\n================ LOGISTIC REGRESSION ================\n")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# ------------------------------------------------------------
# 11. TRAIN MODEL
# ------------------------------------------------------------

print("Training Logistic Regression...")

model.fit(
    X_train_scaled,
    y_train
)

print("Training completed!")


# ------------------------------------------------------------
# 12. PREDICTION
# ------------------------------------------------------------

y_pred = model.predict(
    X_test_scaled
)


# ------------------------------------------------------------
# 13. ACCURACY
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n================ MODEL RESULT ================\n")

print(
    "Logistic Regression Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ------------------------------------------------------------
# 14. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Legitimate",
            "Malicious"
        ]
    )
)


# ------------------------------------------------------------
# 15. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:\n")

print(cm)


# ------------------------------------------------------------
# 16. SAVE CONFUSION MATRIX GRAPH
# ------------------------------------------------------------

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[
        "Legitimate",
        "Malicious"
    ]
)

disp.plot()

plt.title(
    "Logistic Regression URL Classification"
)

plt.tight_layout()

plt.savefig(
    "url_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nConfusion matrix saved as: "
    "url_confusion_matrix.png"
)


# ------------------------------------------------------------
# 17. FEATURE COEFFICIENTS
# ------------------------------------------------------------

coefficients = model.coef_[0]

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Coefficient": coefficients

})


feature_importance["Absolute_Coefficient"] = (
    feature_importance["Coefficient"].abs()
)


feature_importance = feature_importance.sort_values(

    by="Absolute_Coefficient",

    ascending=False
)


print("\n================ FEATURE COEFFICIENTS ================\n")

print(
    feature_importance.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 18. FEATURE COEFFICIENT GRAPH
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.barh(

    feature_importance["Feature"],

    feature_importance["Coefficient"]

)

plt.xlabel(
    "Logistic Regression Coefficient"
)

plt.ylabel(
    "URL Feature"
)

plt.title(
    "URL Features and Logistic Regression Coefficients"
)

plt.axvline(
    0,
    linewidth=1
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "url_feature_coefficients.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nFeature coefficient graph saved as: "
    "url_feature_coefficients.png"
)


# ------------------------------------------------------------
# 19. SAVE PREDICTIONS
# ------------------------------------------------------------

results = pd.DataFrame({

    "URL":
        data.loc[X_test.index, url_column],

    "Actual":
        np.where(
            y_test == 1,
            "Malicious",
            "Legitimate"
        ),

    "Predicted":
        np.where(
            y_pred == 1,
            "Malicious",
            "Legitimate"
        )

})


results.to_csv(
    "url_predictions.csv",
    index=False
)


print(
    "\nPredictions saved to: "
    "url_predictions.csv"
)


# ------------------------------------------------------------
# 20. FINAL RESULT
# ------------------------------------------------------------

print("\n===================================================")

print(
    "              FINAL RESULT"
)

print("===================================================")

print(
    "Model: Logistic Regression"
)

print(
    "Task: URL Classification"
)

print(
    "Classes: Legitimate / Malicious"
)

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("===================================================")

print("\nFiles generated:")

print("1. urls.csv")

print("2. url_confusion_matrix.png")

print("3. url_feature_coefficients.png")

print("4. url_predictions.csv")

print("\nProgram completed successfully!")