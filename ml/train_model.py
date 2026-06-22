'''Run this ONCE to train and save the model.
Command: python ml/train_model.py'''


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
from sklearn.pipeline import Pipeline

#Paths
DATA_PATH = "data/jobs_sample_200.csv"
MODELS_DIR = "models"

#Load Data
print(" Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   Rows loaded: {len(df)}")
print(f"   Columns    : {list(df.columns)}")
print(f"   Categories : {df['Category'].nunique()} unique\n")

#2. Text Cleaning 
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Combine title (3x weight) + description
df["combined"] = (
    df["Job Title"].apply(clean_text) + " " +
    df["Job Title"].apply(clean_text) + " " +
    df["Job Title"].apply(clean_text) + " " +
    df["Job Description"].apply(clean_text)
)

print("Sample cleaned input:")
print(f"   {df['combined'].iloc[0][:80]}...\n")

#3. Features & Labels 
X = df["combined"]
y = df["Category"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train / Test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=42
)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}\n")

#4. TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True         # apply log normalization to term frequencies
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

#5. Train Models & Compare
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
    print(" Accuracy is low because the dataset is small.")
    print(" Replace data/jobs_sample.csv with a larger dataset (500+ rows)")
    print(" and retrain — accuracy should reach 85–95%.\n")
else:
    print()

# Detailed report for best model
print(" Classification Report (Best Model):")
best_preds = best_model.predict(X_test_vec)
print(classification_report(y_test, best_preds))

#6.Save training data
joblib.dump(best_model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("\n Saved:")
print("   models/model.pkl")
print("   models/vectorizer.pkl")
print("   models/label_encoder.pkl")
print("\nTraining complete. You can now run: streamlit run app/app.py")

