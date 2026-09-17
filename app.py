from flask import Flask, render_template, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from gtts import gTTS
import os
import hashlib

from class_names import CLASS_NAMES

app = Flask(__name__)

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "models/traffic_sign_model.keras"
IMG_SIZE = (64, 64)
CONFIDENCE_THRESHOLD = 70

AUDIO_FOLDER = os.path.join("static", "audio")
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

print("Loading traffic sign model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Traffic sign model loaded successfully!")
print("Classes:", model.output_shape[-1])


# ==========================================
# GENERATE VOICE ALERT
# ==========================================

def generate_voice_alert(sign_name):

    # Make the alert sound natural
    text = f"{sign_name} ahead"

    # Create a safe filename
    file_hash = hashlib.md5(text.encode()).hexdigest()
    filename = f"{file_hash}.mp3"

    filepath = os.path.join(AUDIO_FOLDER, filename)

    # Generate only if it doesn't already exist
    if not os.path.exists(filepath):
        try:
            tts = gTTS(text=text, lang="en")
            tts.save(filepath)
        except Exception as e:
            print("gTTS error:", e)
            return None

    return f"/static/audio/{filename}"


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("camera.html")


# ==========================================
# PREDICTION API
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get image sent from browser
        image_data = request.files["frame"]

        image = Image.open(image_data).convert("RGB")

        # ==================================
        # CENTER DETECTION REGION
        # ==================================

        width, height = image.size

        # Central region where the user should
        # place the traffic sign
        box_width = int(width * 0.60)
        box_height = int(height * 0.60)

        left = (width - box_width) // 2
        top = (height - box_height) // 2

        right = left + box_width
        bottom = top + box_height

        cropped = image.crop(
            (left, top, right, bottom)
        )

        # ==================================
        # PREPARE IMAGE
        # ==================================

        cropped = cropped.resize(IMG_SIZE)

        image_array = np.array(cropped)

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # ==================================
        # CNN PREDICTION
        # ==================================

        predictions = model.predict(
            image_array,
            verbose=0
        )

        predicted_class = int(
            np.argmax(predictions[0])
        )

        confidence = float(
            predictions[0][predicted_class] * 100
        )

        sign_name = CLASS_NAMES[predicted_class]

        # ==================================
        # CONFIDENCE CHECK
        # ==================================

        if confidence < CONFIDENCE_THRESHOLD:

            return jsonify({
                "detected": False,
                "message": "No confident traffic sign detected",
                "confidence": round(confidence, 2)
            })

        # ==================================
        # VOICE ALERT
        # ==================================

        audio_url = generate_voice_alert(
            sign_name
        )

        return jsonify({

            "detected": True,

            "class_id": predicted_class,

            "sign": sign_name,

            "confidence": round(
                confidence,
                2
            ),

            "audio": audio_url,

            "box": {

                "left": left,

                "top": top,

                "width": box_width,

                "height": box_height

            }

        })

    except Exception as e:

        print("Prediction error:", e)

        return jsonify({

            "detected": False,

            "error": str(e)

        }), 500


# ==========================================
# RUN SERVER
# ==========================================

if __name__ == "__main__":

    print("\n======================================")
    print("TRAFFIC SIGN RECOGNITION SYSTEM")
    print("======================================")
    print("Open: http://127.0.0.1:5000")
    print("======================================\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )