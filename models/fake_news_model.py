import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load data
data = pd.read_csv("data/processed/fake_news_cleaned.csv")
print("Columns:", data.columns)
print("Subjects:", data['subject'].unique())

# Label: 1 for fake, 0 for real
data['label'] = data['subject'].apply(lambda x: 1 if x == 'politicsNews' else 0)

# Features and labels
X = data['text']
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('lr', LogisticRegression())
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
preds = pipeline.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Accuracy: {acc:.2f}")

# Save model
joblib.dump(pipeline, "models/fake_news_model.pkl")
print("✅ Fake news model saved as fake_news_model.pkl")
