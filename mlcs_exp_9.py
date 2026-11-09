
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample email dataset
emails = [
    "Win a free lottery prize now",
    "Claim your cash reward today",
    "Get a free mobile phone",
    "Congratulations you won money",
    "Meeting scheduled for tomorrow",
    "Please submit your assignment",
    "Project discussion at 10 AM",
    "Your class notes are attached"
]

labels = [
    "Spam", "Spam", "Spam", "Spam",
    "Ham", "Ham", "Ham", "Ham"
]

# Create Naive Bayes model
model = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)

# Train model
model.fit(emails, labels)

# Test messages
test_emails = [
    "Win free money now",
    "Meeting scheduled for today",
    "Claim your prize"
]

predictions = model.predict(test_emails)

print("EMAIL SPAM CLASSIFICATION")

for email, result in zip(test_emails, predictions):
    print("\nEmail:", email)
    print("Classification:", result)
