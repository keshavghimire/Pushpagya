# model.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Directory for student-uploaded images
DATASET_DIR = "student_dataset"

def create_model(num_classes):
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False  

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def train_model(progress):
    global model
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(
        DATASET_DIR, target_size=(224, 224), batch_size=16,
        class_mode="sparse", subset="training"
    )
    val_gen = datagen.flow_from_directory(
        DATASET_DIR, target_size=(224, 224), batch_size=16,
        class_mode="sparse", subset="validation"
    )
    model = create_model(num_classes=len(train_gen.class_indices))

    class ProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, progress):
            super().__init__()
            self.progress = progress
            self.epochs = 10
            self.steps_per_epoch = len(train_gen)
            self.current_epoch = 0

        def on_epoch_begin(self, epoch, logs=None):
            self.current_epoch = epoch
            self.progress(epoch/self.epochs, desc=f"🌷 Level {epoch + 1}/10: Teaching the robot!")

        def on_batch_end(self, batch, logs=None):
            current_step = batch + 1
            self.progress(
                (self.current_epoch * self.steps_per_epoch + current_step) / (self.epochs * self.steps_per_epoch),
                desc=f"🌼 Step {current_step}/{self.steps_per_epoch}: Robot is learning fast!"
            )

    history = model.fit(
        train_gen, 
        validation_data=val_gen, 
        epochs=10,
        callbacks=[ProgressCallback(progress)]
    )
    model.save("student_trained_model.h5")
    return "🌺 Hooray! The robot is super smart now! Let's see what it can do!"

def evaluate_model():
    model = load_model("student_trained_model.h5")
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    val_gen = datagen.flow_from_directory(
        DATASET_DIR, target_size=(224, 224), batch_size=16,
        class_mode="sparse", subset="validation", shuffle=False
    )
    predictions = model.predict(val_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_gen.classes
    cm = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys()))
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=val_gen.class_indices.keys(), yticklabels=val_gen.class_indices.keys())
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()
    return class_report, "confusion_matrix.png"

def predict_unlabeled(img):
    model = load_model("student_trained_model.h5")
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(
        DATASET_DIR, target_size=(224, 224), batch_size=16,
        class_mode="sparse", subset="training"
    )
    class_labels = list(train_gen.class_indices.keys())
    img_array = img_to_array(img)
    img_array = tf.image.resize(img_array, [224, 224]) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)[0]
    top_3_idx = np.argsort(predictions)[-3:][::-1]
    top_3_labels = [class_labels[i] for i in top_3_idx]
    top_3_probs = [round(100 * predictions[i], 2) for i in top_3_idx]
    prediction_text = f"🤖 The robot thinks this picture is:\n"
    for label, prob in zip(top_3_labels, top_3_probs):
        prediction_text += f"• A {label} ({prob}% sure)\n"
    return prediction_text