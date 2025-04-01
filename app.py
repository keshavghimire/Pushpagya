import gradio as gr
from model import create_model, train_model, evaluate_model, predict_unlabeled
from utils import validate_login, upload_images, upload_unlabeled, clear_inputs, clear_dataset
from templates import LOGIN_PAGE_LEFT_HTML
import os

with open("styles.css", "r") as f:
    custom_css = f.read()

current_user_folder = None

def create_app():
    global current_user_folder
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    with gr.Blocks(theme="soft", css=custom_css, title="Flower Explorer 🌸📸") as app:
        gr.HTML("""
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const favicon = document.createElement('link');
                favicon.rel = 'icon';
                favicon.type = 'image/x-icon';
                favicon.href = 'favicon.ico';
                document.head.appendChild(favicon);

                const elem = document.documentElement;
                if (elem.requestFullscreen) {
                    elem.requestFullscreen();
                }
            });
        </script>
        """)

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

        with gr.Group(visible=False) as ai_interface:
            with gr.Group(visible=True) as add_teach_group:
                gr.Markdown("## Add Pictures for Robo to Learn! 🌟📸", elem_classes="step-header", elem_id="step1-header")
                gr.Markdown("Pick some flower pictures 🌸 and name them—like 'Roses' or 'Sunflowers'! Let's grow Robo's brain! 🌼", elem_classes="step-desc")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=300):
                        imgs = gr.Files(file_types=["image"], label="🌸 Drop Your Flower Pics Here! (Files Only) 📸", file_count="multiple", height=150, elem_classes="flower-upload")
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

            with gr.Group(visible=False) as guess_group:
                gr.Markdown("## 🌺 Guess Time! 📸", elem_classes="step-header")
                guess_img = gr.Image(type="pil", label="🌸 Upload a Mystery Picture! 📸")
                guess_btn = gr.Button("🌺 Guess Now! 📸", variant="primary")
                guess_output = gr.Textbox(label="Robot's Guess 🌟🌸")
                reset_btn = gr.Button("🧹 Reset 🌸", variant="secondary")
               

            def set_user_folder_and_validate(firstname, lastname, grade):
                global current_user_folder
                current_user_folder = f"user_datasets/{firstname}_{lastname}"
                print(f"Debug: Set current_user_folder to {current_user_folder}")
                os.makedirs(current_user_folder, exist_ok=True)
                print(f"Debug: Created folder {current_user_folder}")
                login_result = validate_login(firstname, lastname, grade)
                # Clear login fields when switching to AI interface
                return (login_result[0], login_result[1], login_result[2], 
                        gr.update(value=""), gr.update(value=""), gr.update(value=None))

            submit_btn.click(
                fn=set_user_folder_and_validate,
                inputs=[firstname, lastname, grade],
                outputs=[login_page, ai_interface, error_msg, firstname, lastname, grade]
            )

            submit_btn_upload.click(
                fn=lambda imgs, label: upload_images(imgs, label, current_user_folder),
                inputs=[imgs, label],
                outputs=[upload_output, upload_table]
            )
            clear_btn.click(fn=clear_inputs, inputs=[], outputs=[imgs, label])
            train_btn.click(fn=lambda: train_model(gr.Progress(), current_user_folder), inputs=[], outputs=[train_output, test_btn])
            test_btn.click(fn=lambda: (gr.update(visible=False), gr.update(visible=True)), inputs=[], outputs=[add_teach_group, guess_group])
            guess_btn.click(fn=lambda img: predict_unlabeled(img, current_user_folder), inputs=guess_img, outputs=guess_output)
            reset_btn.click(
                fn=lambda: clear_dataset(current_user_folder, login_page, ai_interface),
                inputs=[],
                outputs=[reset_output, login_page, ai_interface]
            )

    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(inbrowser=True, show_api=False)