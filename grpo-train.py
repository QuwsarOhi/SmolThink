# %%
# Reference:
# https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb
# https://www.philschmid.de/mini-deepseek-r1
# https://huggingface.co/blog/open-r1/update-1
# https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb#scrollTo=cXk993X6C2ZZ

#import torch._dynamo
#torch._dynamo.config.suppress_errors = True

import os, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from datasets import load_dataset, Dataset, concatenate_datasets
import peft

# from safetensors.torch import load_model, save_model

import random
import re
import json
import ast
from copy import deepcopy

from typing import Optional
from jinja2 import Template
from transformers.utils import get_json_schema


def get_latest_checkpoint(base_directory):
    checkpoint_dirs = []
    
    # List all directories in the base directory
    for dir_name in os.listdir(base_directory):
        if re.match(r'checkpoint-\d+', dir_name):  # Match pattern "checkpoint-N"
            checkpoint_dirs.append(dir_name)
    
    if not checkpoint_dirs:
        return None  # No checkpoints found
    
    # Sort directories based on numerical value
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))
    
    return os.path.join(base_directory, latest_checkpoint)

# %%
SIZE = "360M"
MODEL_PATH = f"HuggingFaceTB/SmolLM2-{SIZE}-Instruct"
FILE_PATH = get_latest_checkpoint("/Users/ohi/Documents/GitHub/PersonalAssistant/weights/SmolThink-360M-sft/")
# LORA_PATH = os.path.join(LORA_PATH, "think_lora")

model = AutoModelForCausalLM.from_pretrained(
    # MODEL_PATH,
    FILE_PATH,
    device_map="mps",
    low_cpu_mem_usage=True,
    # attn_implementation='sdpa', 'flash_attention_2',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_cache=False,
    tie_word_embeddings=True,
)

# model = peft.PeftModel.from_pretrained(
#     model,
#     LORA_PATH,
#     is_trainable=True, # 👈 here,
# )

# model = model.merge_and_unload().eval().type(torch.bfloat16)

# Gradient checkpointing - Could take more memory in MPS
model.gradient_checkpointing_enable(dict(use_reentrant=False))

# Sanity check
# non_lora_param = 0
# lora_param = 0
# lora_layers = 0
# for name, param in model.named_parameters():
#     # if ...:  # some check on name (ex. if 'lora' in name)
#         # param.requires_grad = False
#     # print(name, param.requires_grad)
#     if 'lora' in name:
#         # param.requires_grad = True
#         assert param.requires_grad == True, f"{name} is not trainable"
#         lora_param += param.numel()
#         lora_layers += 1
#     else:
#         assert param.requires_grad == False
#         non_lora_param += param.numel()

# # print("LoRA adapter added.")
# print(f"Total LoRA params: {lora_param} ({(lora_param/non_lora_param)*100:.2f} %) = ({(lora_param+non_lora_param)/1e6:.2f} million)")
# print(f"Total LoRA layers: {lora_layers}")
# print(f"Approx LoRA size: {lora_param * 2e-6:.2f} mb")
print(f"Model took {model.get_memory_footprint()/1e9:.2f} GB of space (with buffer)")
print(model)

# %%
chat_template = """{%- if tools %}
    {{- '<|endoftext|><|im_start|>system\\n' }}
        {%- if messages[0]['role'] == 'system' %}
            {- messages[0]['content'] }}
        {%- else %}
            {{- 'You are a helpful AI assistant named SmolThink.' }}
        {%- endif %}
    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> tags:\\n<tools>\" }}
    {%- for tool in tools %}
        {{- \"\\n\" }}
            {{- tool | tojson }}
    {%- endfor %}
    {{- \"\\n</tools>\\n\\nYou first think/plan inside <think></think> tags.\\nThen for each function call, return a json object with function name and arguments within <tool_call></tool_call> tags.<|im_end|>\\n\" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|endoftext|><|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
    {%- else %}
        {{- '<|endoftext|><|im_start|>system\\nYou are a helpful AI assistant named SmolThink. First plan/reason/code/validate inside <think></think> tag and provide final answer to user query inside <answer></answer> tag.<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == \"assistant\" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\\n<tool_call>\\n{\"name\": \"' }}
            {{- tool_call.name }}
            {{- '\", \"arguments\": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == \"tool\" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- message.content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n<think>\\n' }}
    {%- if tools %}
        {{- 'I have access to ' }}{% for tool in tools %}'{{ tool.function.name }}'{% if not loop.last %}, {% endif %}{% endfor %}
        {{- ' as tools. Let\\'s evaluate each of them to and then identify the best tool based on given context:' }}
    {% endif %}
{%- endif %}"""

SIZE = "360M"
MODEL_PATH = f"HuggingFaceTB/SmolLM2-{SIZE}-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    add_bos_token=True,
    add_eos_token=True,
)
tokenizer.chat_template = chat_template
tokenizer.pad_token = tokenizer.eos_token
streamer = TextStreamer(tokenizer, skip_prompt=False)
tokenizer.pad_token = tokenizer.eos_token

# %%
from random import Random

random = Random(12)

def generate_data():
    ops = ["*", "/", "+", "-"]
    op = random.choice(ops)

    a = random.randint(1, 100)
    b = random.randint(1, 100)
    while op == '/' and a%b != 0:
        b = random.randint(1, 1000)
        if a % b != 0: continue

    prompt = [{"role": "user", "content": f"Do the following calculation:\n{a} {op} {b} = ?"}]
    ans = eval(f"{a} {op} {b}")
    return {
        'prompt': tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False),
        'ans': str(ans)
    }

math_dataset = [generate_data() for _ in range(1500)]
math_dataset = Dataset.from_list(math_dataset)

# %%
from copy import deepcopy

def tool_call_process(data):
    new_data = {
        'prompt': '',
        'valid': False,
        'tool': None,
        'tool_call': None
    }
    tool_def = None
    try:
        tool_def = json.loads(data['tools'])
        new_data['tool'] = data['tools']
        # print(new_data['tool'], flush=True)
    except Exception as E:
        # print("Error in tool:", E)
        return new_data

    # print(tool_def)

    seq = []
    for s in json.loads(data['conversation']):
        if s['role'] == 'user':
            seq.append(s)
        elif s['role'] == 'tool call':
            tool_call = deepcopy(s['content'])
            if tool_call:
                tool_name = tool_call['name']
                new_data['tool_call'] = str(tool_call)
                tidx = -1
                for idx, tool in enumerate(tool_def):
                    if tool['name'] == tool_name:
                        tidx = idx
                if tidx == -1:
                    return new_data

                try:
                    for (k, v) in tool_call['arguments'].items():
                        if k not in tool_def[tidx].get('arguments'):
                            # print(f"{k} not in {new_data['tool']}", flush=True)
                            return new_data
                except Exception as E:
                    return new_data
            else:
                new_data['tool_call'] = "[]"
        else:
            break
    
    new_data['valid'] = True
    tool_def = [{"type": "function", "function": e} for e in tool_def]
    new_data['prompt'] = tokenizer.apply_chat_template(seq, tools=tool_def, tokenize=False, add_generation_prompt=False)
    new_data['ans'] = ''
    # print(json.dumps(new_data, indent=2), flush=True)
    return new_data

dataset = load_dataset("BitAgent/tool_calling_shuffle")['train']
col_names = dataset.column_names
dataset = dataset.select(range(1500))
dataset = dataset.map(tool_call_process)
dataset = dataset.remove_columns(col_names)
dataset = dataset.filter(lambda x: x['valid'])

print(dataset)
# print("---", flush=True)

# %%
dataset = concatenate_datasets([dataset, math_dataset])
dataset = dataset.shuffle(999)

# for i in range(5):
#     print(dataset[i])

# %%
question = "Do the math: '(9 * 2 + 33) / 2'"

response = '''<think>
Sure, I can help with that. Let me see what I have available. The tools provided include two functions: 'retrieve_payment_status' and 'calculate'. 

First, 'calculate' requires an expression, which in this case is 9 multiplied by 2 plus 33 divided by 2. Since the user has already given a number as input, I don't need to ask for more information here. So I'll use 'calculate' without any arguments because it will handle both numbers directly.

Next, 'retrieve_payment_status' doesn't require any parameters since the user hasn't mentioned anything about payment status. It's just asking for the status of their recent transaction. Therefore, I won't make any changes to its function call.

So, the next step is to call 'calculate' with the expression 9 * 2 + 33 / 2. This should give the user the desired result efficiently.
</think>
<tool_call>
{'name': 'calculate', 'arguments': {'expression': '(9 * 2 + 33) / 2'}}
</tool_call>
<tool_call>
{'name': 'calculate', 'arguments': {'expression': '(9 * 2 + 33) / 2'}}
</tool_call>'''

tool_call = "{'name': 'calculate', 'arguments': {'expression': '(9 * 2 + 33) / 2'}}"


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    responses = [completion[0]["content"] for completion in completions]
    print(responses)
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    print(matches)
    return [0.5 if match else 0.0 for match in matches]


def tool_call_reward(pred, ground):
    score = 0.
    ground_tools = [g['name'] for g in ground]
    for p in pred:
        if p['name'] in ground_tools:
            score += 0.25
        if p in ground:
            score += 0.25
    pred_tools = [p['name'] for p in pred]
    for g in ground:
        if g['name'] in pred_tools:
            score += 0.25
        if g in pred:
            score += 0.25
    return score


def validate_format(text):
    """
    Validate if the text strictly follows the pattern:
    <think> ... </think><tool_call> ... </tool_call>
    
    Returns True if the string matches the pattern, False otherwise.
    """
    pattern = re.compile(r'^\s*<think>.*?</think>\s*<tool_call>.*?</tool_call>\s*$', re.DOTALL)
    return bool(pattern.match(text))


# def correctness_reward_func(prompts, completions, tool_call, **kwargs) -> list[float]:
#     # for i in range(1):
#     #     print("-----Question-----", flush=True)
#     #     print(prompts[i], flush=True)
#     #     print("-----Generation-----", flush=True)
#     #     print(completions[i], flush=True)

#     score = []
#     for gen in completions:
#         tag = "</tool_call>\n"
#         gen = gen[:gen.find(tag)+len(tag)]
#         if validate_format(gen):
#             score.append(0.5)
#         else:
#             score.append(0.0)
        
#     print("Correctness Score:", score, flush=True)
#     return score


from ast import literal_eval
def tool_parse(tool_call:str):
    ret = None
    try:
        ret = literal_eval(tool_call)
    except Exception:
        pass

    _tool_call = tool_call.replace("'", '"')
    ret = json.loads(_tool_call)
    return ret


def tool_call_score(prompts, completions, tool_call, answers=[], **kwargs):
    # Compile regex to capture content inside <tool_call> tags, allowing for whitespace/newlines.
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    matches = [pattern.findall(text) for text in completions]
    calls = []
    score = [0 for _ in range(len(prompts))]
    for i, match in enumerate(matches):
        if not(len(answers) > i and answers[i]): continue
        try: 
            gen_tool_calls = match[0].strip() #.replace("'", '"')
            ground_tool_calls = tool_call[i].strip() #.replace("'", '"')
            print(f" Gen: {gen_tool_calls}\nGrnd: {ground_tool_calls}")
            gen_tool_calls = tool_parse(gen_tool_calls)
            ground_tool_calls = tool_parse(ground_tool_calls)

            if gen_tool_calls and gen_tool_calls == ground_tool_calls:
                score[i] = 1.0 + len(gen_tool_calls['arguments'])*0.25
            elif gen_tool_calls == ground_tool_calls:
                score[i] = 1.0
            elif gen_tool_calls['name'] == ground_tool_calls['name']:
                s = 0.5
                for k, v in gen_tool_calls['arguments'].items():
                    if ground_tool_calls['arguments'].get(k, None) == v:
                        s += 0.25
                    elif ground_tool_calls['arguments'].get(k, None) != None:
                        s += 0.1
                score[i] = s
            else:
                score[i] = 0.25
        except Exception as E:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(E, "|", "line:", exc_tb.tb_lineno)
            score[i] = 0.0

    print("Tool-call Score:", score)
    return score

def math_eval(prompts, completions, tool_call, answers=[], **kwargs):
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    score = [0 for _ in range(len(prompts))]
    for i, ans in enumerate(answers):
        if len(ans) > i and answers[i]:
            try:
                # Getting last three lines
                gen_ans = '\n'.join(pattern.findall(completions[i]).split('\n')[-3:])
                gen_ans = gen_ans.split()
                if ans in gen_ans:
                    score[i] = 2.0
            except:
                pass
    print("Math Score:", score)
    return score

# print(soft_format_reward_func([[{"content": response}]]))
print(tool_call_score([question], [response], [tool_call], []))

# # Example usage:
# text_valid = """<think>
# This is some content inside think.
# </think>
# <tool_call>
# [{'name': 'calculate', 'arguments': {'expression': '9 * 2'}}]
# </tool_call>"""

# text_invalid = """<think>
# This is some content inside think.
# </think>
# Extra content that should not be here.
# <tool_call>
# [{'name': 'calculate', 'arguments': {'expression': '9 * 2'}}]
# </tool_call>"""

# print(validate_format(text_valid))    # Expected output: True
# print(validate_format(text_invalid))  # Expected output: False

# %%
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig
import transformers, gc

class MpsCacheClearCallback(transformers.TrainerCallback):
    def __clearmem(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        # Note: Clearing model gradients
        # for param in model.parameters():
            # param.grad = None
        # print("\nMEMORY CLEARED\n")
    #def on_step_begin(self, *args, **kwargs):      self.__clearmem()
    def on_step_end(self, *args, **kwargs):        self.__clearmem()
    #def on_substep_end(self, *args, **kwargs):     self.__clearmem()
    #def on_evaluate(self, *args, **kwargs):        self.__clearmem()
    #def on_optimizer_step(self, *args, **kwargs):  self.__clearmem()
    #def on_predict(self, *args, **kwargs):         self.__clearmem()
    #def on_prediction_step(self, *args, **kwargs): self.__clearmem()
gc.collect()


training_args = GRPOConfig(
    use_vllm = False,
    learning_rate = 5e-4,
    # adam_beta1 = 0.9,
    # adam_beta2 = 0.99,
    weight_decay = 0.2,
    warmup_ratio = 30 * (100/len(dataset)), #0.1,
    logging_steps=5,
    max_steps=len(dataset),
    save_steps = 10,
    save_total_limit=1,
    ds3_gather_for_generation=False,
    lr_scheduler_type = "cosine",
    # Memory reduction
    optim = "adafactor",    # adamw_torch
    # Memory reduction
    bf16 = True,
    bf16_full_eval=True,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    # Memory reduction
    torch_empty_cache_steps=1,
    num_generations = 2, # Decrease if out of memory
    max_prompt_length = 512,
    max_completion_length = 400,
    temperature=0.9,
    # top_k=15,   # default is 50
    # repetition_penalty = 1.1, # default is 1
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = f"/Users/ohi/Documents/GitHub/PersonalAssistant/weights/SmolLM2-{SIZE}-grpo",
    # Memory reduction
    dataloader_pin_memory=False,
    # Gradient checkpointing - could take more memory in MPS
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
    # torch_compile=True,
    # include_tokens_per_second=True
)

model.config.use_cache = False
model.generation_config.do_sample = True
model.generation_config.temperature = 0.9
model.generation_config.top_k = 20
# model.generation_config.eos_token_id = tokenizer.eos_token_id

trainer = GRPOTrainer(
    model=model,
    # processing_class = tokenizer,
    reward_funcs = [
        # correctness_reward_func,
        tool_call_score,
        math_eval
        # xmlcount_reward_func,
        # soft_format_reward_func,
        # strict_format_reward_func,
        # #int_reward_func,
        # correctness_reward_func,
        # reason_len_reward,
    ],
    args=training_args,
    train_dataset=dataset,
    callbacks=[MpsCacheClearCallback()]
    # peft_config=peft_config, #get_peft_config(model_config),
)

try:
    trainer.train(resume_from_checkpoint=True)
except ValueError as E:
    print("No checkpoint found")
    trainer.train(resume_from_checkpoint=False)

# %%



