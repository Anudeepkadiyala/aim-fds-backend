import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ✅ Load the model (make sure path is correct)
model = load_model('models/deepfake/deepfake_model.h5')  # Updated if it's moved inside models folder

# ✅ Use absolute path or correct relative path
test_dir = 'C:/Users/kadiy/Desktop/Projects/AIM-FDS/data/raw/deepfake/Dataset/Test'
categories = ['Fake', 'Real']

for category in categories:
    folder_path = os.path.join(test_dir, category)
    print(f"\nCategory: {category}")

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            # ✅ Resize to 299x299 for Xception
            img = image.load_img(img_path, target_size=(299, 299))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0][0]
            result = "Fake" if prediction > 0.5 else "Real"

            print(f"{img_name} ➤ Predicted: {result} | Confidence: {prediction:.4f}")
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
