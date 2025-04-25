import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Set paths
train_dir = r"C:\Users\kadiy\Desktop\Projects\AIM-FDS\data\raw\deepfake\Dataset\Train"
val_dir = r"C:\Users\kadiy\Desktop\Projects\AIM-FDS\data\raw\deepfake\Dataset\Validation"
test_dir = r"C:\Users\kadiy\Desktop\Projects\AIM-FDS\data\raw\deepfake\Dataset\Test"

# Confirm they exist
print("Train:", os.path.exists(train_dir))
print("Val:", os.path.exists(val_dir))
print("Test:", os.path.exists(test_dir))

# Data preprocessing
img_size = (128, 128)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
val_data = val_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
test_data = test_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Save best model
checkpoint = ModelCheckpoint("deepfake_model.h5", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Train the model
history = model.fit(train_data, validation_data=val_data, epochs=3, callbacks=[checkpoint])

# Evaluate on test data
loss, acc = model.evaluate(test_data)
print(f"Test Accuracy: {acc * 100:.2f}%")
