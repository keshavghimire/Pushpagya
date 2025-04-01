# utils.py
import os
import shutil
import gradio as gr
from PIL import Image
import pandas as pd

# Directory for student-uploaded images
DATASET_DIR = "student_dataset"
UNLABELED_DIR = "unlabeled_images"

# Create directories if they don't exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(UNLABELED_DIR, exist_ok=True)

def validate_login(firstname, lastname, grade):
    print(f"Validating login: firstname={firstname}, lastname={lastname}, grade={grade}")
    
    # Create fun validation messages
    if not firstname:
        return gr.update(visible=True), gr.update(visible=False), "🌸 Oopsie-daisy! Your first name is missing! What should I call you? 🌟"
    if not lastname:
        return gr.update(visible=True), gr.update(visible=False), "🌻 Oh no! I need your last name too - how else will I send you flower mail? 🌸"
    if not grade:
        return gr.update(visible=True), gr.update(visible=False), "🌺 Whoops! Please tell me what grade you're in so we can start our flower adventure! 📸"
    
    welcome_msg = f"""
    🌟 Welcome, {firstname} {lastname}! 🌟
    
    Awesome! A grade {grade} flower scientist! Get ready for an amazing 
    adventure where YOU get to teach me about beautiful flowers! 🌸📸
    """
    
    print("Validation passed: Switching to AI interface")
    return gr.update(visible=False), gr.update(visible=True), welcome_msg

def upload_images(imgs, label):
    print(f"Debug: Entering upload_images with imgs = {imgs}, label = {label}")
    if not imgs or not label:
        return "🌸 Oops! Please upload some pictures 🌸 and give them a name! 📸", None
    if not isinstance(imgs, (list, tuple)):
        print(f"Error: imgs is not a list or tuple, got {type(imgs)}: {imgs}")
        return "🌸 Error: Invalid upload format. Please upload image files 🌸, not folders! 📸", None

    label_dir = os.path.join(DATASET_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    img_count = len(os.listdir(label_dir))  
    valid_imgs = []

    for i, img in enumerate(imgs):
        try:
            img_path = img.name if hasattr(img, 'name') else img
            print(f"Processing image 🌸 {i}: {img_path}")
            if os.path.isdir(img_path):
                print(f"Skipping directory: {img_path}")
                continue
            img_obj = Image.open(img_path)
            save_path = os.path.join(label_dir, f"{img_count + len(valid_imgs)}.jpg")
            img_obj.save(save_path)
            valid_imgs.append(img_path)
        except Exception as e:
            print(f"Error processing image 🌸 {i}: {e}")
            continue

    if not valid_imgs:
        return "🌸 Oops! No valid images 🌸 were uploaded. Please upload image files only! 📸", None

    category_summary = [[category, len(os.listdir(os.path.join(DATASET_DIR, category)))] 
                        for category in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, category))]
    df = pd.DataFrame(category_summary, columns=["Group 🌸", "Number of Pictures 📸"])
    return f"🌸 You added {len(valid_imgs)} pictures 🌸 to the '{label}' group! Great job! 🌟📸", df

def upload_unlabeled(img):
    img_path = os.path.join(UNLABELED_DIR, f"{len(os.listdir(UNLABELED_DIR))}.jpg")
    img.save(img_path)
    return "🌼 Picture 🌸 added! The robot will guess what it is after learning! 📸"

def clear_inputs():
    print("Debug: Clear button pressed")
    return None, ""

def clear_dataset():
    shutil.rmtree(DATASET_DIR)
    shutil.rmtree(UNLABELED_DIR)
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(UNLABELED_DIR, exist_ok=True)
    return "🌷 All cleared! Let's start a new flower adventure! 🌸📸"