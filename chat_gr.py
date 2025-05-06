import gradio as gr
import json
import os
from datetime import datetime

FEEDBACK_FILE = "feedback.json"

MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4", "custom-model-1"]  # extend as needed

def save_feedback(conversation, feedback_text):
    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "conversation": conversation,
        "feedback": feedback_text,
    }
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            all_feedback = json.load(f)
    else:
        all_feedback = []
    all_feedback.append(data)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(all_feedback, f, indent=2)

# Stubbed LLM response — replace with your model call
def respond(message, chat_history, model_name):
    # Here you'd use `model_name` to pick your LLM
    response = f"[{model_name}] Echo: {message}"
    chat_history = chat_history + [(message, response)]
    return chat_history, chat_history

with gr.Blocks() as demo:
    state = gr.State([])

    # Layout: Chat on left, controls on right
    with gr.Row():
        with gr.Column(scale=3):
            # Increased height for a longer chat window
            chatbot = gr.Chatbot(label="Chat with LLM", height=600)
            msg = gr.Textbox(placeholder="Type here…", label="Your Message")
        with gr.Column(scale=1, min_width=200):
            model_dropdown = gr.Dropdown(choices=MODEL_OPTIONS, value=MODEL_OPTIONS[0], label="Select Model")
            like = gr.Button("👍 Like")
            dislike = gr.Button("👎 Dislike")
            feedback_box = gr.Textbox(label="What went wrong?", visible=False)
            feedback_submit = gr.Button("Submit Feedback", visible=False)

    # Send message on Enter, clear textbox, passing selected model
    msg.submit(
        fn=lambda message, chat, mdl: (*respond(message, chat, mdl), ""),
        inputs=[msg, state, model_dropdown],
        outputs=[chatbot, state, msg]
    )

    # Toggle feedback UI
    dislike.click(
        fn=lambda: (gr.update(visible=True), gr.update(visible=True)),
        inputs=None,
        outputs=[feedback_box, feedback_submit]
    )
    like.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=False)),
        inputs=None,
        outputs=[feedback_box, feedback_submit]
    )

    # Submit and save feedback
    feedback_submit.click(
        fn=lambda feedback, chat: (
            "",                                # clears the feedback textbox
            gr.update(visible=False),          # hides feedback_box
            gr.update(visible=False),          # hides feedback_submit
            save_feedback(chat, feedback)      # side-effect: write JSON
        ),
        inputs=[feedback_box, state],
        outputs=[feedback_box, feedback_submit]
    )

    demo.launch()
