import os
import re
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Paths
DATA_PATH  = "data/Jobs_category_data - Sheet1.csv"
MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Load Data
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   Rows loaded: {len(df)}")
print(f"   Columns    : {list(df.columns)}\n")

# 2. Rename columns to match pipeline
df = df.rename(columns={
    "job-title"  : "Job Title",
    "categories" : "Category",
})

# 3. Consolidate near-duplicate categories
category_map = {
    "Technical"                                : "Information Technology",
    "Digital Innovation & Technology"          : "Information Technology",
    "Innovation and Technology"                : "Information Technology",
    "Engineering & Technical"                  : "Engineering",
    "Industrial / Manufacturing"               : "Manufacturing",
    "Manufacturing & Production"               : "Manufacturing",
    "Human Resources & Talent"                 : "Human Resources",
    "Customer Service & Relations"             : "Customer Service",
    "Finance & Accounts"                       : "Finance",
    "Sales & Business Development"             : "Sales",
    "Customer Projects & Services"             : "Customer Service",
    "Customer Projects, Solutions and Services": "Customer Service",
    "Customer Satisfaction & Quality"          : "Customer Service",
    "Customer Satisfaction and Quality"        : "Customer Service",
    "Marketing and Communication"              : "Marketing",
    "Procurement & Supply Chain"               : "Supply Chain Planning",
    "General Management"                       : "Other / General Management",
}
df["Category"] = df["Category"].replace(category_map)

print(f"   Categories : {df['Category'].nunique()} unique")
print(df["Category"].value_counts().to_string())
print()

# 4. Text Cleaning
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["combined"] = df["Job Title"].apply(clean_text)

print("Sample cleaned input:")
print(f"   {df['combined'].iloc[0][:80]}...\n")

# 5. Features & Labels
X = df["combined"]
y = df["Category"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=42
)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}\n")

# 6. TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# 7. Train Models & Compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=5, random_state=42),
    "Naive Bayes"        : MultinomialNB(alpha=0.5),
}

best_model      = None
best_model_name = ""
best_accuracy   = 0.0

print("Training models...\n")
for name, clf in models.items():
    clf.fit(X_train_vec, y_train)
    preds    = clf.predict(X_test_vec)
    accuracy = accuracy_score(y_test, preds)
    print(f"   {name:<25} Accuracy: {accuracy * 100:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy   = accuracy
        best_model      = clf
        best_model_name = name

print(f"\nBest model: {best_model_name} ({best_accuracy * 100:.2f}%)")
if best_accuracy < 0.70:
    print("   Accuracy is low — consider adding more rows per category.\n")
else:
    print()

# 8. Detailed report for best model
print("Classification Report (Best Model):")
best_preds = best_model.predict(X_test_vec)
print(classification_report(y_test, best_preds))

# 9. Save artifacts
joblib.dump(best_model,  os.path.join(MODELS_DIR, "model.pkl"))
joblib.dump(vectorizer,  os.path.join(MODELS_DIR, "vectorizer.pkl"))
joblib.dump(le,          os.path.join(MODELS_DIR, "label_encoder.pkl"))

print("\nSaved:")
print(f"   {MODELS_DIR}/model.pkl")
print(f"   {MODELS_DIR}/vectorizer.pkl")
print(f"   {MODELS_DIR}/label_encoder.pkl")
print("\nTraining complete. You can now run: streamlit run app/app.py")
