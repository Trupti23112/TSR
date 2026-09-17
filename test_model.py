import tensorflow as tf
import numpy as np
from class_names import CLASS_NAMES

MODEL_PATH = "models/traffic_sign_model.keras"

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")
print("Number of classes:", model.output_shape[-1])

print("\nTraffic sign classes:")
for class_id, name in CLASS_NAMES.items():
    print(f"{class_id}: {name}")