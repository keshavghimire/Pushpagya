import os
import shutil
import numpy as np
import tensorflow as tf
import gradio as gr
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import pandas as pd
import seaborn as sns

# Ensure seaborn is installed
try:
    import seaborn as sns
except ImportError:
    os.system("pip install seaborn")

# Directory for student-uploaded images
DATASET_DIR = "student_dataset"
UNLABELED_DIR = "unlabeled_images"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(UNLABELED_DIR, exist_ok=True)

def validate_login(firstname, lastname, grade):
    if not firstname or not lastname or not grade:
        return gr.update(visible=True), gr.update(visible=False), "Please fill in all fields"
    return gr.update(visible=False), gr.update(visible=True), ""


# Create CNN Model
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

# Train Model with Progress
def train_model(progress=gr.Progress()):
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

    # Create a custom callback for progress tracking
    class ProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, progress):
            super().__init__()
            self.progress = progress
            self.epochs = 10
            self.steps_per_epoch = len(train_gen)
            self.current_epoch = 0

        def on_epoch_begin(self, epoch, logs=None):
            self.current_epoch = epoch
            self.progress(epoch/self.epochs, desc=f"Training epoch {epoch + 1}/{self.epochs}")

        def on_batch_end(self, batch, logs=None):
            current_step = batch + 1
            self.progress(
                (self.current_epoch * self.steps_per_epoch + current_step) / (self.epochs * self.steps_per_epoch),
                desc=f"Training batch {current_step}/{self.steps_per_epoch}"
            )

    history = model.fit(
        train_gen, 
        validation_data=val_gen, 
        epochs=10,
        callbacks=[ProgressCallback(progress)]
    )

    model.save("student_trained_model.h5")

    return "✅ Training Complete! AI has learned from your images."

# Generate Confusion Matrix
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

# AI Prediction on Unlabeled Images
def predict_unlabeled(img):
    model = load_model("student_trained_model.h5")
    
    # Get class labels from the training data
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
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions)[-3:][::-1]
    top_3_labels = [class_labels[i] for i in top_3_idx]
    top_3_probs = [round(100 * predictions[i], 2) for i in top_3_idx]
    
    # Create detailed prediction text
    prediction_text = f"Top Predictions:\n"
    for label, prob in zip(top_3_labels, top_3_probs):
        prediction_text += f"• {label}: {prob}%\n"
    
    return prediction_text

# Upload Images & Label Dataset
def upload_images(imgs, label):
    label_dir = os.path.join(DATASET_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    img_count = len(os.listdir(label_dir))  

    for i, img_path in enumerate(imgs):
        img = Image.open(img_path)  
        save_path = os.path.join(label_dir, f"{img_count + i}.jpg")
        img.save(save_path)

    category_summary = []
    for category in os.listdir(DATASET_DIR):
        category_path = os.path.join(DATASET_DIR, category)
        if os.path.isdir(category_path):
            category_summary.append([category, len(os.listdir(category_path))])

    df = pd.DataFrame(category_summary, columns=["Category", "Image Count"])

    return f"Uploaded {len(imgs)} images to '{label}' category!", df

# Upload Unlabeled Images
def upload_unlabeled(img):
    img_path = os.path.join(UNLABELED_DIR, f"{len(os.listdir(UNLABELED_DIR))}.jpg")
    img.save(img_path)
    return "Unlabeled image uploaded! AI will classify it after training."

# Clear Dataset
def clear_dataset():
    shutil.rmtree(DATASET_DIR)
    shutil.rmtree(UNLABELED_DIR)
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(UNLABELED_DIR, exist_ok=True)
    return "Dataset cleared! Start fresh."

# Create Gradio Interfaces
upload_interface = gr.Interface(
    fn=upload_images,
    inputs=[gr.Files(file_types=["image"], label="Upload Multiple Images"), gr.Textbox(label="Label for Images")],
    outputs=["text", gr.Dataframe(headers=["Category", "Image Count"], interactive=False)],
    title="Upload & Label Multiple Images",
    description="Upload multiple images per category and label them. The table below will show how many images are stored in each category."
)

train_interface = gr.Interface(
    fn=train_model,
    inputs=[],
    outputs="text",
    title="🚀 Train AI on Your Labeled Images",
    description="Click to train the AI based on your labeled images. Progress will be shown below.",
    show_progress=True
)

predict_interface = gr.Interface(
    fn=predict_unlabeled,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="AI Classifies Unlabeled Images",
    description="Upload an image and let the AI predict its category!"
)

clear_interface = gr.Interface(
    fn=clear_dataset,
    inputs=[],
    outputs="text",
    title="Reset Dataset",
    description="Clears all uploaded images and starts fresh."
)

# Replace the final interface creation and launch code with this:
def create_app():
    with gr.Blocks() as app:
        # Login Page
        with gr.Group() as login_page:
            with gr.Row():
                # Left side - Project Info
                with gr.Column(scale=1):
                    gr.Markdown("""
                    # 🤖 AI Image Classification Project
                    
                    Welcome to our educational AI project! This interactive tool helps you:
                    
                    * 📸 Upload and label your own images
                    * 🧠 Train an AI model on your dataset
                    * 🔍 Make predictions on new images
                    
                    Perfect for students learning about:
                    * Machine Learning
                    * Image Classification
                    * Neural Networks
                    * Data Science
                    
                    Get started by entering your information!
                    """)
                
                # Right side - Login Form
                with gr.Column(scale=1):
                    gr.Markdown("## Student Information")
                    firstname = gr.Textbox(label="First Name", placeholder="Enter your first name")
                    lastname = gr.Textbox(label="Last Name", placeholder="Enter your last name")
                    grade = gr.Dropdown(
                        choices=["Grade 6", "Grade 7", "Grade 8", "Grade 9", "Grade 10", "Grade 11", "Grade 12"],
                        label="Grade Level"
                    )
                    submit_btn = gr.Button("Start Learning!", variant="primary")
                    error_msg = gr.Markdown()

        # AI Interface (initially hidden)
        with gr.Group(visible=False) as ai_interface:
            tabs = gr.TabbedInterface(
                [upload_interface, train_interface, predict_interface, clear_interface],
                ["Upload & Label", "Train AI", "Classify", "Reset"]
            )

        # Handle submit button click
        submit_btn.click(
            fn=validate_login,
            inputs=[firstname, lastname, grade],
            outputs=[login_page, ai_interface, error_msg]
        )

    return app

# Launch the application
if __name__ == "__main__":
    app = create_app()
    app.launch()
