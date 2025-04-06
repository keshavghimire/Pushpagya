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
import gradio as gr
import os

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

def train_model(progress, user_folder):
    global model
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(
        user_folder, target_size=(224, 224), batch_size=16,
        class_mode="sparse", subset="training"
    )
    val_gen = datagen.flow_from_directory(
        user_folder, target_size=(224, 224), batch_size=16,
        class_mode="sparse", subset="validation"
    )
    print(f"Debug: Training with {train_gen.samples} images in {len(train_gen.class_indices)} classes")
    if train_gen.samples == 0:
        return "🌸 Oops! No images found to train on. Please upload some flower pictures first! 🌟📸", gr.update(visible=False)
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
            self.progress(epoch/self.epochs, desc=f"🌷 Level {epoch + 1}/10: Teaching the robot! 🌸📸")

        def on_batch_end(self, batch, logs=None):
            current_step = batch + 1
            self.progress(
                (self.current_epoch * self.steps_per_epoch + current_step) / (self.epochs * self.steps_per_epoch),
                desc=f"🌼 Step {current_step}/{self.steps_per_epoch}: Robot is learning fast! 🌸📸"
            )

    try:
        history = model.fit(
            train_gen, 
            validation_data=val_gen, 
            epochs=10,
            callbacks=[ProgressCallback(progress)]
        )
        save_path = os.path.abspath(f"{user_folder}/student_trained_model.h5")
        model.save(save_path)
        print(f"Debug: Model saved to {save_path}")
        return ("🌺 Hooray! The robot is super smart now! Click 'Now Test Me!' to continue! 🌸📸", 
                gr.update(visible=True))
    except Exception as e:
        print(f"Error during training: {e}")
        return f"🌸 Oops! Something went wrong while training: {e} 🌟📸", gr.update(visible=False)

def evaluate_model(user_folder):
    model = load_model(os.path.abspath(f"{user_folder}/student_trained_model.h5"))
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    val_gen = datagen.flow_from_directory(
        user_folder, target_size=(224, 224), batch_size=16,
        class_mode="sparse", subset="validation", shuffle=False
    )
    predictions = model.predict(val_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_gen.classes
    cm = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys()))
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=val_gen.class_indices.keys(), yticklabels=val_gen.class_indices.keys())
    plt.xlabel("Predicted 🌸")
    plt.ylabel("Actual 🌸")
    plt.title("Confusion Matrix 🌟📸")
    plt.savefig(os.path.abspath(f"{user_folder}/confusion_matrix.png"))
    plt.close()
    return class_report, f"{user_folder}/confusion_matrix.png 🌸"

def predict_unlabeled(img, user_folder):
    model = load_model(os.path.abspath(f"{user_folder}/student_trained_model.h5"))
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(
        user_folder, target_size=(224, 224), batch_size=16,
        class_mode="sparse", subset="training"
    )
    class_labels = list(train_gen.class_indices.keys())
    img_array = img_to_array(img)
    img_array = tf.image.resize(img_array, [224, 224]) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)[0]
    top_3_idx = np.argsort(predictions)[-3:][::-1]
    top_3_labels = [class_labels[i] for i in top_3_idx]
    top_3_probs = [predictions[i] * 100 for i in top_3_idx]

    # Format the output with HTML, capitalizing flower names
    prediction_text = "<div style='font-family: \"Comic Sans MS\", cursive, sans-serif;'>🤖 The robot thinks this picture 🌸 is:<br>"
    for i, (label, prob) in enumerate(zip(top_3_labels, top_3_probs)):
        label = label.upper()  # Capitalize the flower name
        if i == 0:  # Highest probability (top prediction)
            prediction_text += f"<div style='font-size: 18px; font-weight: bold; color: #d81b60;'>• A {label} ({prob:.2f}% sure) 🌺📸</div>"
        else:  # Lower probabilities
            prediction_text += f"<div style='font-size: 14px; color: #555;'>• A {label} ({prob:.2f}% ) 🌺📸</div>"
    prediction_text += "</div>"
    return prediction_text