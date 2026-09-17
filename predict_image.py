import tensorflow as tf
import numpy as np
from PIL import Image
from class_names import CLASS_NAMES

# ==============================
# CONFIGURATION
# ==============================

MODEL_PATH = "models/traffic_sign_model.keras"

# Change this to the path of the GTSRB image you want to test
IMAGE_PATH = r"C:\Users\USER\Downloads\archive\Train\14\00014_00025_00005.png"

IMG_SIZE = (64, 64)

# ==============================
# LOAD MODEL
# ==============================

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# ==============================
# LOAD IMAGE
# ==============================

print("Loading image...")

image = Image.open(IMAGE_PATH).convert("RGB")
image = image.resize(IMG_SIZE)

# Convert image to NumPy array
image_array = np.array(image)

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# ==============================
# PREDICTION
# ==============================

predictions = model.predict(image_array, verbose=0)

predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class] * 100

sign_name = CLASS_NAMES[predicted_class]

# ==============================
# RESULT
# ==============================

print("\n==============================")
print("TRAFFIC SIGN PREDICTION")
print("==============================")

print(f"Predicted Class : {predicted_class}")
print(f"Predicted Sign  : {sign_name}")
print(f"Confidence      : {confidence:.2f}%")

print("==============================")