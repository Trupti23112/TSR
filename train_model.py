import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# ==============================
# CONFIGURATION
# ==============================

DATASET_PATH = r"C:\Users\USER\Downloads\archive\Train"

IMG_SIZE = (64, 64)
BATCH_SIZE = 64
EPOCHS = 20
NUM_CLASSES = 43

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "traffic_sign_model.keras")

os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# CLASS NAMES
# ==============================

CLASS_NAMES = [str(i) for i in range(NUM_CLASSES)]

# ==============================
# LOAD DATASET
# ==============================

print("Loading GTSRB dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("Dataset loaded successfully.")

# ==============================
# PERFORMANCE OPTIMIZATION
# ==============================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ==============================
# DATA AUGMENTATION
# ==============================

data_augmentation = keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.1)
])

# ==============================
# CNN MODEL
# ==============================

model = keras.Sequential([

    layers.Input(shape=(64, 64, 3)),

    data_augmentation,

    layers.Rescaling(1.0 / 255),

    # Block 1
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(),

    # Block 2
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(),

    # Block 3
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(),

    # Classification
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(NUM_CLASSES, activation="softmax")
])

# ==============================
# COMPILE
# ==============================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# CALLBACKS
# ==============================

callbacks = [

    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True
    ),

    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True
    )
]

# ==============================
# TRAIN
# ==============================

print("\nStarting training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==============================
# FINAL RESULTS
# ==============================

loss, accuracy = model.evaluate(val_ds)

print("\n==============================")
print("TRAINING COMPLETE")
print("==============================")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
print(f"Model saved to: {MODEL_PATH}")
print("==============================")