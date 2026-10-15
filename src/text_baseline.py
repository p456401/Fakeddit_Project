import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# 1. File paths
# -----------------------------
TRAIN_FILE = "data/processed/train_clean.csv"
VAL_FILE = "data/processed/validation_clean.csv"
TEST_FILE = "data/processed/test_clean.csv"

# -----------------------------
# 2. Load datasets
# -----------------------------
print("Loading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

# -----------------------------
# 3. Separate text and labels
# -----------------------------
X_train = train_df["text"].fillna("")
y_train = train_df["label"]

X_val = val_df["text"].fillna("")
y_val = val_df["label"]

X_test = test_df["text"].fillna("")
y_test = test_df["label"]

# -----------------------------
# 4. TF-IDF
# -----------------------------
print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF validation shape:", X_val_tfidf.shape)
print("TF-IDF test shape:", X_test_tfidf.shape)

# -----------------------------
# 5. Logistic Regression
# -----------------------------
print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)

print("Training completed.")

# -----------------------------
# 6. Validation evaluation
# -----------------------------
val_pred = model.predict(X_val_tfidf)

print("\n========== VALIDATION RESULTS ==========")

print("Accuracy :", accuracy_score(y_val, val_pred))
print("Precision:", precision_score(y_val, val_pred, average="binary"))
print("Recall   :", recall_score(y_val, val_pred, average="binary"))
print("F1 Score :", f1_score(y_val, val_pred, average="binary"))

print("\nClassification Report:")
print(classification_report(y_val, val_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_val, val_pred))

# -----------------------------
# 7. Final test evaluation
# -----------------------------
test_pred = model.predict(X_test_tfidf)

print("\n========== TEST RESULTS ==========")

print("Accuracy :", accuracy_score(y_test, test_pred))
print("Precision:", precision_score(y_test, test_pred, average="binary"))
print("Recall   :", recall_score(y_test, test_pred, average="binary"))
print("F1 Score :", f1_score(y_test, test_pred, average="binary"))

print("\nClassification Report:")
print(classification_report(y_test, test_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))

# -----------------------------
# 8. Save model and vectorizer
# -----------------------------
print("\nSaving model...")

joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(model, "models/logistic_regression.pkl")

print("Model saved successfully.")
print("Baseline completed!")