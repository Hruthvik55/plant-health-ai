import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os

# =========================
# SETTINGS
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

DATASET_PATH = "dataset"
MODEL_PATH = "model/plant_disease_model.h5"

# =========================
# DATA PREPROCESSING
# =========================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="training",
    class_mode="categorical"
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="categorical"
)

# =========================
# CLASS NAMES
# =========================
print("\nDetected Classes:")
print(train_data.class_indices)

# =========================
# BASE MODEL
# =========================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

# =========================
# CUSTOM MODEL
# =========================
model = models.Sequential([
    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.3),

    layers.Dense(train_data.num_classes, activation="softmax")
])

# =========================
# COMPILE
# =========================
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# TRAIN
# =========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# =========================
# SAVE MODEL
# =========================
model.save(MODEL_PATH)

print("\n✅ Model trained successfully!")
print(f"Model saved at: {MODEL_PATH}")