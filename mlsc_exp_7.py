import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
import matplotlib.pyplot as plt

# Characters used for training
characters = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

images = []
labels = []

# Generate character images
for char in characters:
    for i in range(20):
        image = Image.new("L", (40, 40), 255)
        draw = ImageDraw.Draw(image)
        draw.text((10, 8), char, fill=0)

        image_array = np.array(image).flatten()
        images.append(image_array)
        labels.append(char)

# Convert to NumPy arrays
X = np.array(images)
y = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create and train SVM model
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Predict test data
prediction = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, prediction)
print("Model Accuracy:", accuracy)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, prediction, zero_division=0))

# Generate confusion matrix
cm = confusion_matrix(y_test, prediction, labels=list(characters))

print("\nConfusion Matrix:")
print(cm)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=list(characters)
)

disp.plot(xticks_rotation=90, values_format="d")
plt.title("Confusion Matrix - CAPTCHA Recognition")
plt.tight_layout()
plt.show()

# Test with a simple CAPTCHA string
captcha_text = "A7K9"
predicted_text = ""

for char in captcha_text:
    image = Image.new("L", (40, 40), 255)
    draw = ImageDraw.Draw(image)
    draw.text((10, 8), char, fill=0)

    image_array = np.array(image).flatten()
    result = model.predict([image_array])
    predicted_text += result[0]

# Display results
print("\nActual CAPTCHA:", captcha_text)
print("Predicted CAPTCHA:", predicted_text)