# ============================================================
# MCA EXPERIMENT
# Detect Abnormal Patterns in URLs Using Feature Engineering
# ============================================================

import os
import re
import pandas as pd
import matplotlib

# Non-interactive backend for saving graphs
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. LOAD DATASET
# ============================================================

script_folder = os.path.dirname(
    os.path.abspath(__file__)
)

dataset_path = os.path.join(
    script_folder,
    "url_svm_dataset.csv"
)

print("=" * 65)
print("URL ABNORMAL PATTERN DETECTION")
print("USING FEATURE ENGINEERING")
print("=" * 65)

print("\nDataset path:")
print(dataset_path)


if not os.path.exists(dataset_path):

    print("\nERROR: Dataset not found!")
    print("Place url_svm_dataset.csv in:")
    print(script_folder)
    exit()


df = pd.read_csv(dataset_path)

print("\nDataset successfully loaded!")


# ============================================================
# 2. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 65)
print("DATASET INFORMATION")
print("=" * 65)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 URLs:")
print(df.head())


# ============================================================
# 3. URL FEATURE ENGINEERING
# ============================================================

def extract_url_features(url):

    url = str(url)

    features = {}

    # --------------------------------------------------------
    # Length Features
    # --------------------------------------------------------

    features["URL_Length"] = len(url)

    features["Hostname_Length"] = len(
        url.split("/")[2]
    ) if "://" in url else 0


    # --------------------------------------------------------
    # Character Count Features
    # --------------------------------------------------------

    features["Dot_Count"] = url.count(".")

    features["Slash_Count"] = url.count("/")

    features["Hyphen_Count"] = url.count("-")

    features["Underscore_Count"] = url.count("_")

    features["Question_Count"] = url.count("?")

    features["Equal_Count"] = url.count("=")

    features["At_Count"] = url.count("@")

    features["Ampersand_Count"] = url.count("&")

    features["Percent_Count"] = url.count("%")

    features["Digit_Count"] = sum(
        c.isdigit() for c in url
    )


    # --------------------------------------------------------
    # Alphabet Count
    # --------------------------------------------------------

    features["Letter_Count"] = sum(
        c.isalpha() for c in url
    )


    # --------------------------------------------------------
    # Digit Ratio
    # --------------------------------------------------------

    if len(url) > 0:

        features["Digit_Ratio"] = (
            features["Digit_Count"] / len(url)
        )

    else:

        features["Digit_Ratio"] = 0


    # --------------------------------------------------------
    # Special Character Count
    # --------------------------------------------------------

    features["Special_Character_Count"] = sum(
        not c.isalnum() for c in url
    )


    # --------------------------------------------------------
    # Protocol
    # --------------------------------------------------------

    if url.lower().startswith("https://"):

        features["HTTPS"] = 1

    else:

        features["HTTPS"] = 0


    # --------------------------------------------------------
    # Suspicious Characters
    # --------------------------------------------------------

    features["Has_At_Symbol"] = int(
        "@" in url
    )

    features["Has_Double_Slash"] = int(
        "//" in url[8:]
    )


    # --------------------------------------------------------
    # IP Address Detection
    # --------------------------------------------------------

    ip_pattern = (
        r"(?:\d{1,3}\.){3}\d{1,3}"
    )

    features["Has_IP_Address"] = int(
        re.search(ip_pattern, url) is not None
    )


    # --------------------------------------------------------
    # Suspicious Keywords
    # --------------------------------------------------------

    suspicious_words = [
        "login",
        "signin",
        "verify",
        "verification",
        "secure",
        "security",
        "update",
        "confirm",
        "account",
        "password",
        "bank",
        "wallet",
        "free",
        "prize",
        "urgent",
        "billing",
        "recover"
    ]


    url_lower = url.lower()

    keyword_count = 0

    for word in suspicious_words:

        if word in url_lower:

            keyword_count += 1


    features["Suspicious_Keyword_Count"] = (
        keyword_count
    )


    # --------------------------------------------------------
    # Suspicious TLD
    # --------------------------------------------------------

    suspicious_tlds = [
        ".xyz",
        ".tk",
        ".top",
        ".click",
        ".site",
        ".online",
        ".info"
    ]


    features["Suspicious_TLD"] = int(
        any(
            tld in url_lower
            for tld in suspicious_tlds
        )
    )


    # --------------------------------------------------------
    # URL Entropy
    # --------------------------------------------------------

    # Measures randomness of characters
    if len(url) > 0:

        probabilities = [
            url.count(c) / len(url)
            for c in set(url)
        ]

        import math

        entropy = -sum(
            p * math.log2(p)
            for p in probabilities
        )

    else:

        entropy = 0


    features["URL_Entropy"] = entropy


    return features


# ============================================================
# 4. EXTRACT FEATURES FOR ALL URLs
# ============================================================

print("\n" + "=" * 65)
print("EXTRACTING URL FEATURES")
print("=" * 65)


feature_data = []

for url in df["URL"]:

    feature_data.append(
        extract_url_features(url)
    )


features_df = pd.DataFrame(
    feature_data
)


print("\nExtracted Features:")
print(features_df.columns.tolist())


print("\nFeature Dataset Shape:")
print(features_df.shape)


# ============================================================
# 5. DISPLAY FEATURE VALUES
# ============================================================

print("\n" + "=" * 65)
print("FEATURE VALUES")
print("=" * 65)


print(
    features_df.head(10).to_string(
        index=False
    )
)


# ============================================================
# 6. SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    features_df
)


# ============================================================
# 7. ISOLATION FOREST
# ============================================================

print("\n" + "=" * 65)
print("ANOMALY DETECTION")
print("=" * 65)


# Isolation Forest detects unusual observations
# based on their feature patterns.

model = IsolationForest(
    n_estimators=200,
    contamination=0.10,
    random_state=42
)


model.fit(X_scaled)


# Prediction:
#  1  = Normal
# -1  = Anomaly

predictions = model.predict(
    X_scaled
)


# Convert to readable labels

df["Anomaly"] = predictions

df["Pattern"] = df["Anomaly"].map({
    1: "Normal",
    -1: "Abnormal"
})


# ============================================================
# 8. ANOMALY SUMMARY
# ============================================================

normal_count = sum(
    df["Pattern"] == "Normal"
)

abnormal_count = sum(
    df["Pattern"] == "Abnormal"
)


print("\nNormal URLs:")
print(normal_count)

print("\nAbnormal URLs:")
print(abnormal_count)


# ============================================================
# 9. DISPLAY ABNORMAL URLs
# ============================================================

print("\n" + "=" * 65)
print("ABNORMAL URL PATTERNS")
print("=" * 65)


abnormal_urls = df[
    df["Pattern"] == "Abnormal"
]


print(
    abnormal_urls[
        ["URL", "Label", "Pattern"]
    ].to_string(index=False)
)


# ============================================================
# 10. SAVE FEATURE DATASET
# ============================================================

feature_output = pd.concat(
    [
        df[
            ["URL", "Label", "Pattern"]
        ].reset_index(drop=True),

        features_df.reset_index(drop=True)
    ],
    axis=1
)


feature_path = os.path.join(
    script_folder,
    "url_engineered_features.csv"
)


feature_output.to_csv(
    feature_path,
    index=False
)


print("\nFeature dataset saved:")
print(feature_path)


# ============================================================
# 11. ANOMALY RESULTS GRAPH
# ============================================================

pattern_counts = df[
    "Pattern"
].value_counts()


plt.figure(
    figsize=(7, 5)
)


pattern_counts.plot(
    kind="bar"
)


plt.title(
    "Normal vs Abnormal URL Patterns"
)


plt.xlabel(
    "Pattern"
)


plt.ylabel(
    "Number of URLs"
)


plt.xticks(
    rotation=0
)


plt.tight_layout()


anomaly_graph = os.path.join(
    script_folder,
    "url_anomaly_distribution.png"
)


plt.savefig(
    anomaly_graph,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print("\nAnomaly graph saved:")
print(anomaly_graph)


# ============================================================
# 12. URL LENGTH GRAPH
# ============================================================

plt.figure(
    figsize=(9, 5)
)


plt.hist(
    features_df[
        "URL_Length"
    ],
    bins=20
)


plt.title(
    "Distribution of URL Length"
)


plt.xlabel(
    "URL Length"
)


plt.ylabel(
    "Number of URLs"
)


plt.tight_layout()


length_graph = os.path.join(
    script_folder,
    "url_length_distribution.png"
)


plt.savefig(
    length_graph,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print("\nURL length graph saved:")
print(length_graph)


# ============================================================
# 13. SUSPICIOUS KEYWORD GRAPH
# ============================================================

plt.figure(
    figsize=(9, 5)
)


plt.hist(
    features_df[
        "Suspicious_Keyword_Count"
    ],
    bins=range(
        0,
        features_df[
            "Suspicious_Keyword_Count"
        ].max() + 2
    )
)


plt.title(
    "Suspicious Keyword Distribution"
)


plt.xlabel(
    "Number of Suspicious Keywords"
)


plt.ylabel(
    "Number of URLs"
)


plt.tight_layout()


keyword_graph = os.path.join(
    script_folder,
    "suspicious_keyword_distribution.png"
)


plt.savefig(
    keyword_graph,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nSuspicious keyword graph saved:"
)

print(keyword_graph)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)


print(
    f"""
Total URLs              : {len(df)}

Engineered Features     : {len(features_df.columns)}

Normal URLs             : {normal_count}

Abnormal URLs           : {abnormal_count}

Detection Algorithm     : Isolation Forest

Feature Engineering     : URL length, character counts,
                          digit ratio, HTTPS, IP address,
                          suspicious keywords, suspicious TLD,
                          URL entropy and special characters
"""
)


print("=" * 65)
print("OUTPUT FILES")
print("=" * 65)


print(
    "\n1. url_engineered_features.csv"
)

print(
    "2. url_anomaly_distribution.png"
)

print(
    "3. url_length_distribution.png"
)

print(
    "4. suspicious_keyword_distribution.png"
)


print("\n" + "=" * 65)
print("PROGRAM COMPLETED SUCCESSFULLY")
print("=" * 65)