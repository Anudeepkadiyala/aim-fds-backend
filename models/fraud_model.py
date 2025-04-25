import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load data
data = pd.read_csv("data/processed/fraud_cleaned.csv")
print("✅ Columns in fraud data:", data.columns)

# Features and label
X = data.drop('Class', axis=1)  # 'Class' is the fraud label (0 = legit, 1 = fraud)
y = data['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"✅ Fraud Detection Accuracy: {acc:.2f}")

# Save model
joblib.dump(model, "models/fraud_model.pkl")
print("✅ Model saved as models/fraud_model.pkl")
