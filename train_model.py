import csv
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


DATA_FILE = "sign_data_normalized.csv"
MODEL_FILE = "signova_normalized_model.pkl"


# ==========================================
# LOAD DATA
# ==========================================

X = []
y = []

with open(DATA_FILE, "r", newline="") as file:

    reader = csv.reader(file)

    for line_number, row in enumerate(reader, start=1):

        # 1 label + 63 features
        if len(row) != 64:
            print(
                f"Skipping bad row {line_number}: "
                f"{len(row)} fields"
            )
            continue

        label = row[0]

        try:
            features = [float(value) for value in row[1:]]
        except ValueError:
            print(f"Skipping invalid row {line_number}")
            continue

        X.append(features)
        y.append(label)


X = np.array(X)
y = np.array(y)


# ==========================================
# DATASET INFORMATION
# ==========================================

print("\n================================")
print("       SIGNOVA DATASET")
print("================================")

print(f"Valid samples: {len(X)}")
print(f"Features per sample: {X.shape[1]}")
print(f"Classes: {np.unique(y)}")

for label in np.unique(y):
    print(
        f"{label}: "
        f"{np.sum(y == label)} samples"
    )


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# TRAIN RANDOM FOREST
# ==========================================

print("\nTraining normalized SIGNOVA model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# ==========================================
# PREDICTION
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n================================")
print("        MODEL RESULTS")
print("================================")

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n================================")
print("    CLASSIFICATION REPORT")
print("================================")

print(
    classification_report(
        y_test,
        predictions
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

labels = model.classes_

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

print("\n================================")
print("       CONFUSION MATRIX")
print("================================")

print(cm)


# ==========================================
# SAVE CONFUSION MATRIX
# ==========================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

display.plot()

plt.title(
    "SIGNOVA Normalized Model - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "signova_normalized_confusion_matrix.png"
)

plt.show()


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    MODEL_FILE
)

print("\n================================")
print("       FILES CREATED")
print("================================")

print(MODEL_FILE)
print("signova_normalized_confusion_matrix.png")

print("\nSIGNOVA normalized training completed!")