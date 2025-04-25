import joblib

# Load your trained models
hate_model = joblib.load("models/hate_speech_model.pkl")
fake_model = joblib.load("models/fake_news_model.pkl")

print("Hate Speech Model Pipeline Steps:")
print(hate_model.named_steps)

print("\nFake News Model Pipeline Steps:")
print(fake_model.named_steps)
