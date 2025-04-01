import os
import shutil
import gradio as gr
from PIL import Image
import pandas as pd

def validate_login(firstname, lastname, grade):
    print(f"Validating login: firstname={firstname}, lastname={lastname}, grade={grade}")
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

def upload_images(imgs, label, user_folder):
    print(f"Debug: Entering upload_images with imgs = {imgs}, label = {label}, user_folder = {user_folder}")
    print(f"Debug: Current working directory: {os.getcwd()}")
    if not imgs or not label:
        print("Debug: No images or label provided")
        return "🌸 Oops! Please upload some pictures 🌸 and give them a name! 📸", None
    if not isinstance(imgs, (list, tuple)):
        print(f"Error: imgs is not a list or tuple, got {type(imgs)}: {imgs}")
        return "🌸 Error: Invalid upload format. Please upload image files 🌸, not folders! 📸", None

    label_dir = os.path.abspath(os.path.join(user_folder, label))
    try:
        os.makedirs(label_dir, exist_ok=True)
        print(f"Debug: Successfully created/using directory: {label_dir}")
    except Exception as e:
        print(f"Error: Failed to create directory {label_dir}: {e}")
        return f"🌸 Error: Could not create folder {label_dir}: {e} 🌟📸", None

    img_count = len(os.listdir(label_dir))
    print(f"Debug: Initial image count in {label_dir}: {img_count}")
    valid_imgs = []

    for i, img in enumerate(imgs):
        print(f"Debug: Processing image {i}: {img}")
        try:
            img_path = img.name if hasattr(img, 'name') else str(img)
            print(f"Debug: Image path: {img_path}")
            if not os.path.exists(img_path):
                print(f"Error: Image file does not exist at {img_path}")
                continue
            if os.path.isdir(img_path):
                print(f"Debug: Skipping directory: {img_path}")
                continue

            img_obj = Image.open(img_path)
            save_path = os.path.abspath(os.path.join(label_dir, f"{img_count + len(valid_imgs)}.jpg"))
            print(f"Debug: Attempting to save image to: {save_path}")
            img_obj.save(save_path, "JPEG")
            if os.path.exists(save_path):
                print(f"Debug: Successfully saved image to: {save_path}")
                valid_imgs.append(img_path)
            else:
                print(f"Error: Save failed - file not found at {save_path}")
        except Exception as e:
            print(f"Error processing image {i} at {img_path}: {e}")
            continue

    if not valid_imgs:
        print("Debug: No valid images were processed")
        return "🌸 Oops! No valid images 🌸 were uploaded. Please upload image files only! 📸", None

    print(f"Debug: Processed {len(valid_imgs)} valid images")
    category_summary = [[category, len(os.listdir(os.path.join(user_folder, category)))] 
                        for category in os.listdir(user_folder) if os.path.isdir(os.path.join(user_folder, category))]
    df = pd.DataFrame(category_summary, columns=["Group 🌸", "Number of Pictures 📸"])
    print(f"Debug: Returning summary: {category_summary}")
    return f"🌸 You added {len(valid_imgs)} pictures 🌸 to the '{label}' group! Great job! 🌟📸", df

def upload_unlabeled(img, user_folder):
    unlabeled_dir = os.path.join(user_folder, "unlabeled_images")
    os.makedirs(unlabeled_dir, exist_ok=True)
    img_path = os.path.join(unlabeled_dir, f"{len(os.listdir(unlabeled_dir))}.jpg")
    img.save(img_path)
    return "🌼 Picture 🌸 added! The robot will guess what it is after learning! 📸"

def clear_inputs():
    print("Debug: Clear button pressed")
    return None, ""

def clear_dataset(user_folder, login_page, ai_interface):
    # No longer deleting the user_folder, just resetting the UI for a new session
    print(f"Debug: Resetting session for {user_folder} without deleting data")
    return "🌷 Session reset! Start a new flower adventure! 🌸📸", gr.update(visible=True), gr.update(visible=False)