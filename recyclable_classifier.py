# recyclable_classifier.py
# Author: Davis Ongeri
# Classifies recyclable vs non-recyclable images using CNN + TFLite

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from PIL import Image

# -----------------------------
# Step 1: Dataset Preparation
# -----------------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
SEED = 123

# Path to dataset (same level as this script)
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")

# Load training and validation datasets
train_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("\n📂 Classes found:", train_ds.class_names)

# Prefetching for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# Step 2: Build the CNN Model
# -----------------------------
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# -----------------------------
# Step 3: Train the Model
# -----------------------------
print("\n🧠 Training the model...")
model.fit(train_ds, validation_data=val_ds, epochs=5)

# -----------------------------
# Step 4: Evaluate and Save
# -----------------------------
val_loss, val_acc = model.evaluate(val_ds)
print(f"\n✅ Validation Accuracy: {val_acc:.2%}")

model.save("recyclable_classifier.h5")
print("💾 Saved model as 'recyclable_classifier.h5'")

# -----------------------------
# Step 5: Convert to TFLite
# -----------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("recyclable_classifier.tflite", "wb") as f:
    f.write(tflite_model)
print("📦 Converted and saved model as 'recyclable_classifier.tflite'")

# -----------------------------
# Step 6: Test Inference
# -----------------------------
def run_inference(image_path):
    img = Image.open(image_path).resize(IMG_SIZE)
    input_data = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

    interpreter = tf.lite.Interpreter(model_path="recyclable_classifier.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    prediction = "♻️ Recyclable" if output[0][0] > 0.5 else "❌ Non-Recyclable"
    print(f"\n📷 {os.path.basename(image_path)} → Prediction: {prediction}")

# Example test image (you can change this path)
test_image_path = "C:/Users/user/Documents/plp_learning/AI-ML/wk6 assignment/edge_ai_project/dataset/recyclable/item1.jpeg"
if os.path.exists(test_image_path):
    run_inference(test_image_path)
else:
    print("⚠️ Test image not found. Please check the path.")
