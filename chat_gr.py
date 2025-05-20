import gradio as gr
import json
import os
from datetime import datetime
import torch
from transformers import TextIteratorStreamer, AutoTokenizer, AutoModelForCausalLM
from threading import Thread
import re
import copy
from sft.tokenizer import get_tokenizer
from webtool.webtool import remove_think, search_tool, StopWordCriteria
import random

model = None
tokenizer = None

GEN_CONFIG = dict(
    max_new_tokens=1024,
    temperature=0.6,
    top_p=0.95,
    # repetition_penalty=1.1,
    do_sample=True
)

def load_model():
    global model
    global tokenizer

    def get_latest_checkpoint(base_directory):
        checkpoint_dirs = []
        # List all directories in the base directory
        for dir_name in os.listdir(base_directory):
            if re.match(r"checkpoint-\d+", dir_name):  # Match pattern "checkpoint-N"
                checkpoint_dirs.append(dir_name)
        if not checkpoint_dirs:
            return None  # No checkpoints found
        # Sort directories based on numerical value
        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))
        return os.path.join(base_directory, latest_checkpoint)

    SIZE = ["135M", "360M"][1]
    MODEL_PATH = get_latest_checkpoint(
        f"/Users/ohi/Documents/GitHub/SmolThink/weights/SmolThink-{SIZE}-codeact"
    )
    # MODEL_PATH = "/Users/ohi/Documents/GitHub/PersonalAssistant/weights/SmolThink-360M-sft-websearch/checkpoint-2029"
    print(MODEL_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map='mps',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False,
        tie_word_embeddings=True,
    )

    # model = torch.compile(model, mode='reduce-overhead', fullgraph=True)
    tokenizer = get_tokenizer(MODEL_PATH)
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model.generation_config.eos_token_id = tokenizer.eos_token_id
    print("Model loaded")
    return


# def save_feedback(conversation, feedback_text):
#     FEEDBACK_FILE = "feedback.json"
#     data = {
#         "timestamp": datetime.utcnow().isoformat(),
#         "conversation": conversation,
#         "feedback": feedback_text,
#     }
#     if os.path.exists(FEEDBACK_FILE):
#         with open(FEEDBACK_FILE, "r") as f:
#             all_feedback = json.load(f)
#     else:
#         all_feedback = []
#     all_feedback.append(data)
#     with open(FEEDBACK_FILE, "w") as f:
#         json.dump(all_feedback, f, indent=2)


SYS_PROMPTS = [
    # {
    #   "content": "You are a helpful assistant assigned with the task of problem-solving. To achieve this, you will be using an interactive coding environment equipped with a variety of tool functions to assist you throughout the process.\n\nAt each turn, you should first provide your step-by-step thinking for solving the task. After that, you have two options:\n\n1) Interact with a Python programming environment and receive the corresponding output. Your code should be enclosed using \"<execute>\" tag, for example: <execute> print(\"Hello World!\") <\/execute>.\n2) Directly provide a solution that adheres to the required format for the given task. Your solution should be enclosed using \"<solution>\" tag, for example: The answer is <solution> A <\/solution>.\n\nYou have 5 chances to interact with the environment or propose a solution. You can only propose a solution 2 times.\n\nTool function available (already imported in <execute> environment):\n[1] wikipedia_search(query: str) -> str\nThe Wikipedia Search tool provides access to a vast collection of articles covering a wide range of topics.\nCan query specific keywords or topics to retrieve accurate and comprehensive information.",
    #   "role": "system"
    # },
    {
      "content": "You are a helpful assistant assigned with the task of problem-solving. To achieve this, you will be using an interactive coding environment equipped with a variety of tool functions to assist you throughout the process.\n\nAt each turn, you should first provide your step-by-step thinking for solving the task. After that, you have two options:\n\n1) Interact with a Python programming environment and receive the corresponding output. Your code should be enclosed using \"<execute>\" tag, for example: <execute> print(\"Hello World!\") <\/execute>.\n2) Directly provide a solution that adheres to the required format for the given task. Your solution should be enclosed using \"<solution>\" tag, for example: The answer is <solution> A <\/solution>.\n\nYou have 5 chances to interact with the environment or propose a solution. You can only propose a solution 2 times.\n\nTool function available (already imported in <execute> environment):\n[1] web_search(query: str) -> str\nThe Web Search tool provides access to a vast collection of information covering a wide range of topics from the web.\nCan query specific keywords or topics to retrieve accurate and comprehensive information.",
      "role": "system"
    },
    {
        "content": "You are a helpful AI assistant.",
        "role": "system"
    }
]

def generate(messages, temp, top_p, use_wiki, lead=None):
    # gens = ["Hmm,", "Okay", "The user is asking", "Let's think,"]
    # lead = f"<think>\nOkay,"
    # del messages[0] # System message deleted
    global model
    global tokenizer
    load_model()

    if lead:
        print("USING LEAD")
        lead = "For your question, I would use Google search:\n<execute>"

    if use_wiki:
        messages = [SYS_PROMPTS[0]] + messages
    else:
        messages = [SYS_PROMPTS[1]] + messages
    
    message = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # message = remove_think(message)
    if lead:
        message += lead
    
    stopping_criteria = StopWordCriteria(
        tokenizer=tokenizer, prompts=[message], stop_words=["</execute>", "<\/execute>", "<\/solution>"]
    )
    
    print("-----------")
    print(message)
    input_ids = tokenizer(message, return_tensors='pt')

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        input_ids=input_ids["input_ids"].to("mps"),
        attention_mask=input_ids["attention_mask"].to("mps"),
        streamer=streamer,
        max_new_tokens=1024,
        temperature=temp,
        top_p=top_p,
        # repetition_penalty=1.1,
        do_sample=True,
        stopping_criteria=[stopping_criteria],
    )

    gen = ""
    if lead:
        print(lead, end="", flush=True)
        gen += lead
        yield lead

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        print(new_text, end="", flush=True)
        gen += new_text
        yield new_text
    

    match = re.search(r'google_search\((["\'])(.*?)\1\)', gen)
    print("Search:", match, flush=True)
    # if match and match.group(2):
    #     result, urls, contents = search_tool(
    #         match.group(2), 
    #         trim=512,
    #         max_results=2,
    #     )
    #     result = f"Observation:\n\n{result}"
    #     print(result)
    
    del model, tokenizer


# Stubbed LLM response — replace with your model call
def respond(message, chat_history, temp, top_p, use_wiki, lead):
    # Here you'd use `model_name` to pick your LLM
    chat_history.append({'role': 'user', 'content': message})
    new_chat_history = copy.deepcopy(chat_history)
    new_chat_history.append({"role": "assistant", "content": ""})
    for tokens in generate(chat_history, temp, top_p, use_wiki, lead):
        new_chat_history[-1]['content'] += tokens
        yield new_chat_history, new_chat_history, ''


with gr.Blocks() as demo:
    # global GEN_CONFIG
    state = gr.State([])

    # Add editing state in Gradio: https://github.com/gradio-app/gradio/issues/7919

    # Layout: Chat on left, controls on right
    with gr.Row():
        with gr.Column(scale=4):
            # Increased height for a longer chat window
            chatbot = gr.Chatbot(
                label="Chat with LLM",
                height=600,
                type='messages',
                resizable=True,
                show_copy_button=True,
                sanitize_html=False,
                allow_tags=['execute']
                # allow_tags=['think', 'answer'],
            )
            msg = gr.Textbox(placeholder="Type here…", label="Your Message", submit_btn=True)
        with gr.Column(scale=1):
            gr.Markdown(value="## LLM Variables")
            # model_dropdown = gr.Dropdown(choices=MODEL_OPTIONS, value=MODEL_OPTIONS[0], label="Select Model")
            temperature = gr.Slider(label="Temperature", minimum=0.01, maximum=0.99, value=0.7, interactive=True, visible=True)
            top_p = gr.Slider(label="Top P", minimum=0.01, maximum=0.9999, value=0.95, interactive=True, visible=True)
            

            gr.Markdown(value="## Tool Options")
            wiki_search = gr.Checkbox(label="Google search", value=True, interactive=True, visible=True)
            force_search = gr.Checkbox(label="Force Search", value=False, interactive=True, visible=True)

            # like = gr.Button("👍 Like")
            # dislike = gr.Button("👎 Dislike")
            # feedback_box = gr.Textbox(label="What went wrong?", visible=False)
            # feedback_submit = gr.Button("Submit Feedback", visible=False)

    # Send message on Enter, clear textbox, passing selected model
    msg.submit(
        fn=respond,
        inputs=[msg, state, temperature, top_p, wiki_search, force_search],
        outputs=[chatbot, state, msg]
    )

    chatbot.clear(fn = lambda: [], outputs = state)


    def force_search_state(state):
        if state:
            return gr.update(visible=True)
        return gr.update(visible=False)

    wiki_search.input(
        fn = force_search_state,
        inputs = [wiki_search],
        outputs = [force_search]
    )

    # def set_val(k, v):
    #     GEN_CONFIG[k] = v

    # top_p.input(
    #     fn = lambda v: float(v),
    #     inputs = [top_p],
    #     outputs = [GEN_CONFIG['top_p']]
    # )

    # temperature.input(
    #     fn = lambda v: float(v),
    #     inputs = [temperature],
    #     outputs = [GEN_CONFIG['temperature']]
    # )

    # Toggle feedback UI
    # dislike.click(
    #     fn=lambda: (gr.update(visible=True), gr.update(visible=True)),
    #     inputs=None,
    #     outputs=[feedback_box, feedback_submit]
    # )

    # like.click(
    #     fn=lambda: (gr.update(visible=False), gr.update(visible=False)),
    #     inputs=None,
    #     outputs=[feedback_box, feedback_submit]
    # )

    # Submit and save feedback
    # feedback_submit.click(
    #     fn=lambda feedback, chat: (
    #         "",                                # clears the feedback textbox
    #         gr.update(visible=False),          # hides feedback_box
    #         gr.update(visible=False),          # hides feedback_submit
    #         save_feedback(chat, feedback)      # side-effect: write JSON
    #     ),
    #     inputs=[feedback_box, state],
    #     outputs=[feedback_box, feedback_submit]
    # )

    demo.launch()
