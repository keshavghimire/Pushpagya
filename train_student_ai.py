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
    print(f"Validating login: firstname={firstname}, lastname={lastname}, grade={grade}")
    
    # Create fun validation messages
    if not firstname:
        return gr.update(visible=True), gr.update(visible=False), "🌸 Oopsie-daisy! Your first name is missing! What should I call you?"
    if not lastname:
        return gr.update(visible=True), gr.update(visible=False), "🌻 Oh no! I need your last name too - how else will I send you flower mail?"
    if not grade:
        return gr.update(visible=True), gr.update(visible=False), "🌺 Whoops! Please tell me what grade you're in so we can start our flower adventure!"
    
    welcome_msg = f"""
    🌟 Welcome, {firstname} {lastname}! 🌟
    
    Awesome! A grade {grade} flower scientist! Get ready for an amazing 
    adventure where YOU get to teach me about beautiful flowers!
    """
    
    print("Validation passed: Switching to AI interface")
    return gr.update(visible=False), gr.update(visible=True), welcome_msg

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

def upload_images(imgs, label):
    print(f"Debug: Entering upload_images with imgs = {imgs}, label = {label}")
    if not imgs or not label:
        return "🌸 Oops! Please upload some pictures and give them a name!", None
    if not isinstance(imgs, (list, tuple)):
        print(f"Error: imgs is not a list or tuple, got {type(imgs)}: {imgs}")
        return "🌸 Error: Invalid upload format. Please upload image files, not folders!", None

    label_dir = os.path.join(DATASET_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    img_count = len(os.listdir(label_dir))  
    valid_imgs = []

    for i, img in enumerate(imgs):
        try:
            img_path = img.name if hasattr(img, 'name') else img
            print(f"Processing image {i}: {img_path}")
            if os.path.isdir(img_path):
                print(f"Skipping directory: {img_path}")
                continue
            img_obj = Image.open(img_path)
            save_path = os.path.join(label_dir, f"{img_count + len(valid_imgs)}.jpg")
            img_obj.save(save_path)
            valid_imgs.append(img_path)
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            continue

    if not valid_imgs:
        return "🌸 Oops! No valid images were uploaded. Please upload image files only!", None

    category_summary = [[category, len(os.listdir(os.path.join(DATASET_DIR, category)))] 
                        for category in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, category))]
    df = pd.DataFrame(category_summary, columns=["Group", "Number of Pictures"])
    return f"🌸 You added {len(valid_imgs)} pictures to the '{label}' group! Great job!", df

def upload_unlabeled(img):
    img_path = os.path.join(UNLABELED_DIR, f"{len(os.listdir(UNLABELED_DIR))}.jpg")
    img.save(img_path)
    return "🌼 Picture added! The robot will guess what it is after learning!"

def clear_inputs():
    print("Debug: Clear button pressed")
    return None, ""

def clear_dataset():
    shutil.rmtree(DATASET_DIR)
    shutil.rmtree(UNLABELED_DIR)
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(UNLABELED_DIR, exist_ok=True)
    return "🌷 All cleared! Let's start a new flower adventure!"

def create_app():
    custom_css = """
    body, html {
        margin: 0;
        padding: 0;
        min-height: 100vh;
        width: 100vw;
        background: linear-gradient(135deg, #ffebee, #fff9c4);
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    .gradio-container {
        min-height: 100vh !important;
        width: 100vw !important;
        margin: 0 !important;
        padding: 10px !important;
        display: flex !important;
        flex-direction: column !important;
    }
    .gr-group {
        max-width: 90vw !important;
        padding: 10px !important;
        border-radius: 15px !important;
        background: rgba(255, 255, 255, 0.85) !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1) !important;
    }
    .step-header {
        color: #d81b60 !important;
        font-size: 20px !important;
        text-align: center !important;
        margin-bottom: 5px !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2) !important;
    }
    .step-desc {
        color: #555 !important;
        font-size: 14px !important;
        text-align: center !important;
        margin-bottom: 10px !important;
    }
    button {
        background-color: #f06292 !important;
        color: white !important;
        border-radius: 15px !important;
        padding: 8px 20px !important;
        font-size: 14px !important;
        transition: transform 0.2s, background-color 0.2s !important;
        cursor: pointer !important;
    }
    button:hover {
        transform: scale(1.05) !important;
        background-color: #ec407a !important;
    }
    .flower-upload, .flower-textbox, .status-box, .flower-table {
        border: 2px solid #ffca28 !important;
        border-radius: 10px !important;
        padding: 5px !important;
        background: rgba(255, 255, 255, 0.9) !important;
        margin-bottom: 10px !important;
    }
    .flower-upload {
        max-height: 150px !important;
    }
    .flower-table {
        max-height: 100px !important;
        overflow-y: auto !important;
    }
    .gr-row {
        align-items: stretch !important;
        margin-bottom: 10px !important;
        flex-wrap: wrap !important;
    }
    .gr-column {
        padding: 5px !important;
        min-width: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important; /* Center content vertically */
    }
    @media (max-width: 768px) {
        .step-header {
            font-size: 16px !important;
        }
        .step-desc {
            font-size: 12px !important;
        }
        button {
            padding: 6px 15px !important;
            font-size: 12px !important;
        }
        .gr-column {
            min-width: 100% !important;
        }
    }
    .custom-input {
        padding: 8px !important;
        border: 2px solid #f8bbd0 !important;
        border-radius: 8px !important;
        font-size: 14px !important;
        margin-bottom: 10px !important;
    }
    .custom-label {
        display: block !important;
        margin-bottom: 5px !important;
        font-weight: bold !important;
        color: #6d4c41 !important;
        font-size: 0.9em !important;
    }
    """

    with gr.Blocks(theme="soft", css=custom_css) as app:
        gr.HTML("""
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const elem = document.documentElement;
                if (elem.requestFullscreen) {
                    elem.requestFullscreen();
                }
            });
        </script>
        """)

        # Login Page
        with gr.Group() as login_page:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div style="
                        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                        padding: 15px;
                        border-radius: 15px;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        border: 2px solid #ffffff;
                        min-height: 300px; /* Ensure consistent height */
                    ">
                        <h1 style="
                            color: #d81b60;
                            text-align: center;
                            font-size: 1.8em;
                            margin-bottom: 10px;
                            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
                        ">
                            🌸 Pushpagya: Flower Recognition Adventure! 🤖
                        </h1>
                        
                        <div style="
                            background: white;
                            padding: 10px;
                            border-radius: 10px;
                            margin-bottom: 10px;
                        ">
                            <p style="font-size: 0.9em; margin: 5px 0;">Hello, flower explorer! I'm Robo the Robot, and I need YOUR help to learn about flowers!</p>
                            
                            <div style="
                                display: flex;
                                align-items: center;
                                margin: 5px 0;
                            ">
                                <span style="font-size: 1.5em; margin-right: 10px;">🌷</span>
                                <span style="font-size: 0.9em;">Pushpagya means <strong>"flower knowledge"</strong> in Sanskrit!</span>
                            </div>
                            
                            <div style="background: #f8f9fa; padding: 10px; border-radius: 8px;">
                                <h3 style="color: #6d4c41; margin: 0 0 5px 0; font-size: 1em;">What You'll Do:</h3>
                                <ul style="padding-left: 15px; margin: 0;">
                                    <li style="margin-bottom: 3px; font-size: 0.85em;">📸 <strong>Add pictures</strong> of flowers like roses or sunflowers</li>
                                    <li style="margin-bottom: 3px; font-size: 0.85em;">🏷️ <strong>Teach me</strong> their names and features</li>
                                    <li style="margin-bottom: 3px; font-size: 0.85em;">🔍 <strong>Test me</strong> with new flowers to see what I've learned</li>
                                </ul>
                            </div>
                            
                            <div style="
                                background: #e3f2fd;
                                padding: 10px;
                                border-radius: 8px;
                                margin-top: 10px;
                            ">
                                <h3 style="color: #1976d2; margin: 0 0 5px 0; font-size: 1em;">Why It's Awesome:</h3>
                                <div style="display: flex; flex-wrap: wrap; gap: 5px;">
                                    <div style="
                                        background: white;
                                        padding: 5px 8px;
                                        border-radius: 15px;
                                        display: inline-flex;
                                        align-items: center;
                                        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                                        font-size: 0.8em;
                                    ">
                                        🌼 Learn botany in a fun way
                                    </div>
                                    <div style="
                                        background: white;
                                        padding: 5px 8px;
                                        border-radius: 15px;
                                        display: inline-flex;
                                        align-items: center;
                                        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                                        font-size: 0.8em;
                                    ">
                                        🤖 See AI learning in action
                                    </div>
                                    <div style="
                                        background: white;
                                        padding: 5px 8px;
                                        border-radius: 15px;
                                        display: inline-flex;
                                        align-items: center;
                                        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                                        font-size: 0.8em;
                                    ">
                                        🌿 Become a nature expert
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <p style="text-align: center; font-style: italic; color: #6d4c41; font-size: 0.85em; margin: 5px 0;">
                            Ready to begin your journey as a flower scientist?
                        </p>
                    </div>
                    """)

                with gr.Column(scale=1):
                    gr.HTML("""
                    <div style="
                       
                      
                       
                      
                    ">
                        <h2 style="
                            padding:15px;
                            color: #e91e63;
                            text-align: center;
                            font-size: 1.5em;
                            margin-bottom: 10px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            gap: 5px;
                        ">
                            <span>🌷</span>
                            <span>Tell Me About You!</span>
                            <span>🌷</span>
                        </h2>
                    </div>
                    """)
                    # Gradio form inputs (replacing HTML inputs)
                    firstname = gr.Textbox(
                        label="Your First Name",
                        placeholder="What's your name?",
                        elem_classes="custom-input"
                    )
                    lastname = gr.Textbox(
                        label="Your Last Name",
                        placeholder="What's your family name?",
                        elem_classes="custom-input"
                    )
                    grade = gr.Dropdown(
                        choices=["Grade 5", "Grade 6", "Grade 7", "Grade 8", 
                                 "Grade 9", "Grade 10", "Grade 11", "Grade 12"],
                        label="What Grade Are You In?",
                        elem_classes="custom-input"
                    )
                    submit_btn = gr.Button("🌸 Let's Start the Flower Adventure! 🌸")
                    error_msg = gr.Markdown(visible=False)

        # AI Interface (initially hidden)
        with gr.Group(visible=False) as ai_interface:
            gr.Markdown("### 🤖 Hi, I'm Robo the Robot! Let's have fun with flowers! 🌺", elem_classes="step-header")
            with gr.Tabs():
                with gr.Tab("Add & Teach"):
                    with gr.Column(scale=1, min_width=0):
                        gr.Markdown(
                            "## 🌷 Step 1: Add Pictures for Robo to Learn! 🌟",
                            elem_classes="step-header"
                        )
                        gr.Markdown(
                            "Pick some flower pictures and name them—like 'Roses' or 'Sunflowers'! Let's grow Robo's brain! 🌼",
                            elem_classes="step-desc"
                        )

                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1, min_width=300):
                                imgs = gr.Files(
                                    file_types=["image"],
                                    label="🌸 Drop Your Flower Pics Here! (Files Only)",
                                    file_count="multiple",
                                    height=150,
                                    elem_classes="flower-upload"
                                )
                                label = gr.Textbox(
                                    label="🌺 Name This Flower Group",
                                    placeholder="e.g., Daisies or Tulips",
                                    elem_classes="flower-textbox"
                                )
                                with gr.Row():
                                    clear_btn = gr.Button("🌿 Clear", variant="secondary")
                                    submit_btn_upload = gr.Button("🌼 Add Pics!", variant="primary")

                            with gr.Column(scale=1, min_width=300):
                                upload_output = gr.Textbox(
                                    label="🌟 Robo's Update",
                                    interactive=False,
                                    lines=3,
                                    elem_classes="status-box"
                                )
                                upload_table = gr.Dataframe(
                                    headers=["Flower Group", "Pic Count"],
                                    interactive=False,
                                    label="🌸 Your Flower Collection",
                                    wrap=True,
                                    elem_classes="flower-table"
                                )

                        gr.Markdown(
                            "## 🌻 Step 2: Teach Robo the Flower Magic! 🚀",
                            elem_classes="step-header"
                        )
                        gr.Markdown(
                            "Press the button to train Robo—it's like giving it a flower superpower! Watch it learn! 🌟",
                            elem_classes="step-desc"
                        )
                        train_btn = gr.Button("🌈 Teach Robo Now!", variant="primary")
                        train_output = gr.Textbox(
                            label="🌼 Robo's Learning Diary",
                            interactive=False,
                            lines=2,
                            elem_classes="status-box"
                        )

                        submit_btn_upload.click(
                            fn=upload_images,
                            inputs=[imgs, label],
                            outputs=[upload_output, upload_table]
                        )
                        clear_btn.click(
                            fn=clear_inputs,
                            inputs=[],
                            outputs=[imgs, label]
                        )
                        train_btn.click(
                            fn=train_model,
                            inputs=[],
                            outputs=train_output
                        )

                with gr.Tab("Guess & Reset"):
                    with gr.Column():
                        gr.Markdown("## 🌺 Guess the Picture!", elem_classes="step-header")
                        gr.Markdown("Upload a picture and see if the robot thinks it's a flower! 🌷", elem_classes="step-desc")
                        with gr.Row():
                            with gr.Column(scale=1):
                                guess_img = gr.Image(type="pil", label="🌸 Upload a Mystery Picture!")
                                guess_btn = gr.Button("Let the Robot Guess! 🤖", variant="primary")
                        guess_output = gr.Textbox(label="Robot's Guess")
                        gr.Markdown("## 🌷 Start Over", elem_classes="step-header")
                        gr.Markdown("Click the button below to clear everything and start a new flower adventure! 🌼", elem_classes="step-desc")
                        reset_btn = gr.Button("Start Over 🧹", variant="primary")
                        reset_output = gr.Textbox(label="Reset Status")

                        guess_btn.click(
                            fn=predict_unlabeled,
                            inputs=guess_img,
                            outputs=guess_output
                        )
                        reset_btn.click(
                            fn=clear_dataset,
                            inputs=[],
                            outputs=reset_output
                        )

        submit_btn.click(
            fn=validate_login,
            inputs=[firstname, lastname, grade],
            outputs=[login_page, ai_interface, error_msg]
        )

    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(inbrowser=True, show_api=False)