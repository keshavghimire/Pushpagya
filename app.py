import gradio as gr
from model import create_model, train_model, evaluate_model, predict_unlabeled
from utils import validate_login, upload_images, upload_unlabeled, clear_inputs, clear_dataset
from templates import LOGIN_PAGE_LEFT_HTML
import os

# Load CSS
with open("styles.css", "r") as f:
    custom_css = f.read()

def create_app():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.getcwd()
    print(f"Debug: Base directory set to {base_dir}")

    with gr.Blocks(theme="soft", css=custom_css, title="Pushpagya") as app:
        gr.HTML("""
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const favicon = document.createElement('link');
                favicon.rel = 'icon';
                favicon.type = 'image/x-icon';
                favicon.href = 'icon.png';
                document.head.appendChild(favicon);
                const elem = document.documentElement;
                if (elem.requestFullscreen) {
                    elem.requestFullscreen();
                }
            });
        </script>
        """)

        # App state
        dataset_state = gr.State(value=None)

        # LOGIN SCREEN
        with gr.Group() as login_page:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML(LOGIN_PAGE_LEFT_HTML)
                with gr.Column(scale=1):
                    gr.HTML("""
                        <h2 style="color: #e91e63; text-align: center; font-size: 1.5em; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; gap: 5px; padding: 20px;">
                            <span>🌷</span>
                            <span>Tell Me About You! 🌸</span>
                            <span>🌷</span>
                        </h2>
                    """)
                    firstname = gr.Textbox(label="Your First Name 🌟", placeholder="What's your name?", elem_classes="custom-input")
                    lastname = gr.Textbox(label="Your Last Name 🌟", placeholder="What's your family name?", elem_classes="custom-input")
                    grade = gr.Dropdown(choices=["Grade 5", "Grade 6", "Grade 7", "Grade 8", "Grade 9", "Grade 10", "Grade 11", "Grade 12"], label="What Grade Are You In? 🌟", elem_classes="custom-input")
                    submit_btn = gr.Button("🌸 Let's Start the Flower Adventure! 🌸📸")
                    error_msg = gr.Markdown(visible=False)

        # AI INTERFACE
        with gr.Group(visible=False) as ai_interface:
            # ADD & TEACH SECTION
            with gr.Group(visible=True) as add_teach_group:
                gr.Markdown("## Add Pictures for Robo to Learn! 🌟📸", elem_classes="step-header", elem_id="step1-header")
                gr.Markdown("Pick some flower pictures 🌸 and name them—like 'Roses' or 'Sunflowers'! Let's grow Robo's brain! 🌼", elem_classes="step-desc")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=300):
                        imgs = gr.Gallery(label="🌸 Drop Your Flower Pics Here! (Files Only) 📸", 
                                         type="filepath", 
                                         height=200, 
                                         elem_classes="flower-upload", 
                                         preview=True, 
                                         object_fit="contain")
                        label = gr.Textbox(label="🌺 Name This Flower Group 🌸", placeholder="e.g., Daisies or Tulips", elem_classes="flower-textbox")
                        with gr.Row():
                            clear_btn = gr.Button("🌿 Clear 🌸", variant="secondary")
                            submit_btn_upload = gr.Button("🌼 Add Pics! 📸", variant="primary")
                    with gr.Column(scale=1, min_width=300):
                        upload_output = gr.Textbox(label="🌟 Robo's Update 🌸", interactive=False, lines=1, elem_classes="status-box")
                        upload_table = gr.Dataframe(headers=["Flower Group 🌸", "Pic Count 📸"], interactive=False, label="🌸 Your Flower Collection 🌟", wrap=True, elem_classes="flower-table")
                gr.Markdown("## Teach Robo the Flower Magic! 🚀📸", elem_classes="step-header")
                gr.Markdown("Press the button to train Robo 🤖—it's like giving it a flower superpower! Watch it learn! 🌟🌸", elem_classes="step-desc")
                train_btn = gr.Button("🌈 Teach Robo Now! 🌸📸", variant="primary")
                train_output = gr.Textbox(label="🌼 Robo's Learning Diary 🌸", interactive=False, lines=2, elem_classes="status-box")
                test_btn = gr.Button("🌟 Now Test Me! 📸", variant="primary", visible=False)

            # GUESS SECTION
            with gr.Group(visible=False) as guess_group:
                gr.Markdown("## 🌺 Guess Time! 📸", elem_classes="step-header")
                guess_img = gr.Image(type="pil", label="🌸 Upload a Mystery Picture! 📸")
                guess_btn = gr.Button("🌺 Guess Now! 📸", variant="primary")
                # Updated: Changed from gr.Textbox to gr.HTML to render HTML output
                guess_output = gr.HTML(label="Robot's Guess 🌟🌸")
                reset_btn = gr.Button("🧹 Reset 🌸", variant="secondary")
                reset_output = gr.Textbox(label="Reset Status 🌟🌸")

            # Event Handlers
            def set_user_folder_and_validate(firstname, lastname, grade):
                user_folder = os.path.join(base_dir, f"user_datasets/{firstname}_{lastname}")
                print(f"Debug: Setting user_folder to {user_folder}")
                os.makedirs(user_folder, exist_ok=True)
                print(f"Debug: Created folder {user_folder}")
                login_result = validate_login(firstname, lastname, grade)
                return (login_result[0], login_result[1], login_result[2], 
                        user_folder,           # Update dataset_state
                        gr.update(value=""),   # Clear firstname
                        gr.update(value=""),   # Clear lastname
                        gr.update(value=None)) # Clear grade

            def reset_everything(user_folder):
                print(f"Debug: Resetting UI for new session, preserving data in {user_folder}")
                return (gr.update(visible=True),    # Show login_page
                        gr.update(visible=False),   # Hide ai_interface
                        gr.update(visible=True),    # Show add_teach_group
                        gr.update(visible=False),   # Hide guess_group
                        "🌷 Ready for a new flower explorer! Data preserved 🌸📸",  # reset_output
                        None,                       # Clear imgs
                        "",                         # Clear label
                        "",                         # Clear upload_output
                        None,                       # Clear upload_table
                        "",                         # Clear train_output
                        gr.update(visible=False),   # Reset test_btn visibility
                        None,                       # Clear guess_img
                        "",                         # Clear guess_output
                        None)                       # Clear dataset_state

            # Login
            submit_btn.click(
                fn=set_user_folder_and_validate,
                inputs=[firstname, lastname, grade],
                outputs=[login_page, ai_interface, error_msg, dataset_state, firstname, lastname, grade]
            )

            # Upload images with dynamic table update and auto-clear
            def handle_upload(imgs, label, user_folder):
                result, table = upload_images(imgs, label, user_folder)
                print(f"Debug: Upload result: {result}")
                if "Error" not in result and "Oops" not in result:
                    return result, table, None, ""  # Clear imgs and label
                return result, table, imgs, label  # Keep inputs if there's an error

            submit_btn_upload.click(
                fn=handle_upload,
                inputs=[imgs, label, dataset_state],
                outputs=[upload_output, upload_table, imgs, label]
            )

            clear_btn.click(
                fn=clear_inputs,
                inputs=[],
                outputs=[imgs, label]
            )

            # Track progress during training
            def handle_train(user_folder):
                result, test_btn_update = train_model(gr.Progress(), user_folder)
                return result, test_btn_update

            train_btn.click(
                fn=handle_train,
                inputs=[dataset_state],
                outputs=[train_output, test_btn]
            )

            test_btn.click(
                fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
                inputs=[],
                outputs=[add_teach_group, guess_group]
            )

            guess_btn.click(
                fn=predict_unlabeled,
                inputs=[guess_img, dataset_state],
                outputs=[guess_output]
            )

            reset_btn.click(
                fn=reset_everything,
                inputs=[dataset_state],
                outputs=[login_page, ai_interface, add_teach_group, guess_group, reset_output,
                        imgs, label, upload_output, upload_table, train_output, test_btn,
                        guess_img, guess_output, dataset_state]
            )

    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(inbrowser=True, show_api=False, share=True)