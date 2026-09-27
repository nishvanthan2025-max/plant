import os, json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

DATA_DIR = "data/plantvillage"
MODEL_DIR = "models"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError("Add image class folders inside data/plantvillage first.")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training",
    seed=42, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation",
    seed=42, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
    json.dump(class_names, f)

model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(len(class_names), activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
          callbacks=[EarlyStopping(patience=3, restore_best_weights=True),
                     ModelCheckpoint(os.path.join(MODEL_DIR, "plant_disease_model.keras"), save_best_only=True)])
loss, acc = model.evaluate(val_ds)
print(f"Validation accuracy: {acc*100:.2f}%")
