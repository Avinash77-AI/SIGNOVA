import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = "sign_data_normalized.csv"
MODEL_FILE = "signova_nn.pkl"

print("Loading SIGNOVA dataset...")

df = pd.read_csv(DATA_FILE)

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

print("Dataset shape:", X.shape)
print("Classes:", sorted(y.unique()))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Neural Network...")

model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

# Test
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print()
print("================================")
print("SIGNOVA Neural Network Results")
print("================================")
print("Accuracy:", accuracy)
print()
print(classification_report(y_test, predictions))

# Save model
joblib.dump(model, MODEL_FILE)

print("================================")
print("Neural Network saved!")
print("Model:", MODEL_FILE)
print("================================")