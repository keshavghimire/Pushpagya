# app.py
import gradio as gr
from model import create_model, train_model, evaluate_model, predict_unlabeled
from utils import validate_login, upload_images, upload_unlabeled, clear_inputs, clear_dataset
from templates import LOGIN_PAGE_LEFT_HTML

# Load custom CSS
with open("styles.css", "r") as f:
    custom_css = f.read()

def create_app():
    with gr.Blocks(
        theme="soft",
        css=custom_css,
        title="Flower Explorer 🌸📸"
    ) as app:
        # Inject favicon and fullscreen script
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

        # Login Page
        with gr.Group() as login_page:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML(LOGIN_PAGE_LEFT_HTML)

                with gr.Column(scale=1):
                    gr.HTML("""
                        <h2 style="
                            color: #e91e63;
                            text-align: center;
                            font-size: 1.5em;
                            margin-bottom: 10px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            gap: 5px;
                            padding: 20px;
                        ">
                            <span>🌷</span>
                            <span>Tell Me About You! 🌸</span>
                            <span>🌷</span>
                        </h2>
                    """)
                    firstname = gr.Textbox(
                        label="Your First Name 🌟",
                        placeholder="What's your name?",
                        elem_classes="custom-input"
                    )
                    lastname = gr.Textbox(
                        label="Your Last Name 🌟",
                        placeholder="What's your family name?",
                        elem_classes="custom-input"
                    )
                    grade = gr.Dropdown(
                        choices=["Grade 5", "Grade 6", "Grade 7", "Grade 8", 
                                 "Grade 9", "Grade 10", "Grade 11", "Grade 12"],
                        label="What Grade Are You In? 🌟",
                        elem_classes="custom-input"
                    )
                    submit_btn = gr.Button("🌸 Let's Start the Flower Adventure! 🌸📸")
                    error_msg = gr.Markdown(visible=False)

        # AI Interface (initially hidden)
        with gr.Group(visible=False) as ai_interface:
            # Add & Teach Section (visible initially, includes greeting)
            with gr.Group(visible=True) as add_teach_group:
                gr.Markdown(
                    "## Add Pictures for Robo to Learn! 🌟📸",
                        elem_classes="step-header",
                        elem_id="step1-header"
                )
                gr.Markdown(
                    "Pick some flower pictures 🌸 and name them—like 'Roses' or 'Sunflowers'! Let's grow Robo's brain! 🌼",
                    elem_classes="step-desc"
                )

                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=300):
                        imgs = gr.Files(
                            file_types=["image"],
                            label="🌸 Drop Your Flower Pics Here! (Files Only) 📸",
                            file_count="multiple",
                            height=150,
                            elem_classes="flower-upload"
                        )
                        label = gr.Textbox(
                            label="🌺 Name This Flower Group 🌸",
                            placeholder="e.g., Daisies or Tulips",
                            elem_classes="flower-textbox"
                        )
                        with gr.Row():
                            clear_btn = gr.Button("🌿 Clear 🌸", variant="secondary")
                            submit_btn_upload = gr.Button("🌼 Add Pics! 📸", variant="primary")

                    with gr.Column(scale=1, min_width=300):
                        upload_output = gr.Textbox(
                            label="🌟 Robo's Update 🌸",
                            interactive=False,
                            lines=1,
                            elem_classes="status-box"
                        )
                        upload_table = gr.Dataframe(
                            headers=["Flower Group 🌸", "Pic Count 📸"],
                            interactive=False,
                            label="🌸 Your Flower Collection 🌟",
                            wrap=True,
                            elem_classes="flower-table"
                        )

                gr.Markdown(
                    "## Teach Robo the Flower Magic! 🚀📸",
                    elem_classes="step-header"
                )
                gr.Markdown(
                    "Press the button to train Robo 🤖—it's like giving it a flower superpower! Watch it learn! 🌟🌸",
                    elem_classes="step-desc"
                )
                train_btn = gr.Button("🌈 Teach Robo Now! 🌸📸", variant="primary")
                train_output = gr.Textbox(
                    label="🌼 Robo's Learning Diary 🌸",
                    interactive=False,
                    lines=2,
                    elem_classes="status-box"
                )
                # "Now Test Me!" button, initially hidden
                test_btn = gr.Button("🌟 Now Test Me! 📸", variant="primary", visible=False)

            # Guess Group (initially hidden, renamed to "Guess")
            with gr.Group(visible=False) as guess_group:
                gr.Markdown(
                    "## 🌺 Guess Time! 📸",
                    elem_classes="step-header"
                )
                guess_img = gr.Image(type="pil", label="🌸 Upload a Mystery Picture! 📸")
                guess_btn = gr.Button("🌺 Guess Now! 📸", variant="primary")
                guess_output = gr.Textbox(label="Robot's Guess 🌟🌸")
                # New Reset/Logout button
                reset_btn = gr.Button("🧹 Reset 🌸", variant="secondary")
                reset_output = gr.Textbox(label="Reset Status 🌟🌸")

            # Event handlers
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
            # Train button shows "Now Test Me!" button after training
            train_btn.click(
                fn=lambda: train_model(gr.Progress()),
                inputs=[],
                outputs=[train_output, test_btn]
            )
            # "Now Test Me!" button hides Add & Teach and shows Guess
            def switch_to_guess():
                return gr.update(visible=False), gr.update(visible=True)
            
            test_btn.click(
                fn=switch_to_guess,
                inputs=[],
                outputs=[add_teach_group, guess_group]
            )
            # Guess button functionality (only guessing)
            guess_btn.click(
                fn=predict_unlabeled,
                inputs=guess_img,
                outputs=guess_output
            )
            # Reset button functionality
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