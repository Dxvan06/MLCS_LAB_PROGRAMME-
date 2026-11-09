
import pandas as pd

# Sample email dataset
data = {
    "sender": [
        "admin@google.com",
        "support@paypal.com",
        "security@gmail.com",
        "user@yahoo.com",
        "bank@fakebank.xyz"
    ],
    "reply_to": [
        "admin@google.com",
        "attacker@unknown.com",
        "security@gmail.com",
        "user@yahoo.com",
        "bank@fakebank.xyz"
    ],
    "spf": [
        "pass", "fail", "pass", "pass", "fail"
    ]
}

df = pd.DataFrame(data)

def detect_spoofing(row):
    sender_domain = row["sender"].split("@")[1]
    reply_domain = row["reply_to"].split("@")[1]

    if sender_domain != reply_domain or row["spf"] == "fail":
        return "Suspicious / Spoofed"
    return "Legitimate"

df["Result"] = df.apply(detect_spoofing, axis=1)

print("EMAIL SPOOFING DETECTION")
print(df[["sender", "reply_to", "spf", "Result"]])
