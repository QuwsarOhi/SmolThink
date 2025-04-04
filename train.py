# %%
# Reference:
# * https://huggingface.co/agents-course/notebooks/blob/main/bonus-unit1/bonus-unit1.ipynb
# * https://colab.research.google.com/#scrollTo=29da85c8-33bf-4864-aed7-733cbe703512&fileId=https%3A//huggingface.co/agents-course/notebooks/blob/main/bonus-unit1/bonus-unit1.ipynb


# Dataset:
# * https://huggingface.co/datasets/XeTute/Open-Coding-Thoughts
# * https://huggingface.co/datasets/UWNSL/Mix-Large_large_0.2_small_0.8
# * https://huggingface.co/datasets/Jofthomas/hermes-function-calling-thinking-V1
# * https://huggingface.co/datasets/AymanTarig/function-calling-v0.2-with-r1-cot

# Tied lm_head & embed_tokens:
# * https://github.com/huggingface/peft/issues/1750
# * https://github.com/huggingface/peft/pull/2025
# * https://github.com/huggingface/peft/issues/2018

# LoRA vs Full FT
# * https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from datasets import load_dataset, concatenate_datasets,  load_from_disk, Dataset
import peft

# from safetensors.torch import load_model, save_model

import random
import re
import json
import ast
from copy import deepcopy
from enum import Enum

from typing import Optional
from jinja2 import Template
from transformers.utils import get_json_schema
from webtool.webtool import webtool_def


os.environ['TOKENIZERS_PARALLELISM'] = 'false'
lora_r = None # 32
SIZE = "360M" #"135M"
# MODEL_PATH = f"HuggingFaceTB/SmolLM2-{SIZE}-Instruct"
MODEL_PATH = "quwsarohi/SmolThink"
SAVE_PATH = f"weights/SmolThink-{SIZE}-sft-websearch"

# MODEL_PATH = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
# SAVE_PATH = "SmolThink-Qwen-sft"

# LORA_PATH = None
# dataset = load_from_disk("/Users/ohi/Documents/GitHub/PersonalAssistant/datasets/merged_dataset")
dataset = None

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
{%- endif %}"""

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    add_bos_token=True,
    add_eos_token=True,
)
tokenizer.chat_template = chat_template
tokenizer.pad_token = tokenizer.unk_token
streamer = TextStreamer(tokenizer, skip_prompt=True)

print(tokenizer.apply_chat_template([
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I am fine"}
], tokenize=False))

tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_payment_status",
            "description": "Get payment status of a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id.",
                    }
                },
                "required": ["transaction_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_payment_date",
            "description": "Get payment date of a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id.",
                    }
                },
                "required": ["transaction_id"],
            },
        },
    }
]
# print("\n-----\n")
# print(tokenizer.apply_chat_template([
#     {"role": "user", "content": "How are you?"},
#     {"role": "assistant", "content": "<tool_call>[retrieve_payment_date(12)]</tool_call>"},
#     {"role": "tool", "content": "12/12/12"},
#     {"role": "assistant", "content": "12/12/12"}
# ], tools=tools, tokenize=False))

# %%
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    # "/Users/ohi/Documents/GitHub/PersonalAssistant/weights/checkpoint-22400",
    device_map="cpu",
    low_cpu_mem_usage=True,
    attn_implementation='eager', # 'sdpa'
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_cache=False,
    tie_word_embeddings=True,
)

# Gradient checkpointing - Could take more memory in MPS
model.gradient_checkpointing_enable(dict(use_reentrant=False))
# model.gradient_checkpointing_disable()

# model.resize_token_embeddings(len(tokenizer))
model = model.to('mps')
print(f"Model took {model.get_memory_footprint()/1e9:.2f} GB of space (with buffer)")

# %%
# PEFT ref: https://huggingface.co/docs/transformers/en/peft

# r: rank dimension for LoRA update matrices (smaller = more compression)
# lora_alpha: scaling factor for LoRA layers (higher = stronger adaptation)
# lora_dropout: dropout probability for LoRA layers (helps prevent overfitting)

if lora_r:
    SAVE_PATH += f'-r{lora_r}'
    peft_config = peft.LoraConfig(
        r=lora_r,                   # 64
        lora_alpha=2*lora_r,        # alpha = 4 * r | 2 * r
        lora_dropout=0.1,
        target_modules='all-linear',
        modules_to_save = [
            "embed_tokens", 
            "lm_head"
        ],
        # use_rslora=True,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian",
        inference_mode=False,
    )
    model = peft.get_peft_model(model, peft_config, adapter_name="smolthink", autocast_adapter_dtype=False)

    # Sanity check
    non_lora_param = 0
    lora_param = 0
    lora_layers = 0
    for name, param in model.named_parameters():
        if 'lora' in name:
            # param.requires_grad = True
            assert param.requires_grad == True, f"{name} is not trainable"
            lora_param += param.numel()
            lora_layers += 1
        else:
            # if not param.requires_grad:
            #     print(f"{name} is trainable")
            non_lora_param += param.numel()

        # if 'lm_head' in name:
        #     print("lm_head ->", name, ":", param.requires_grad)
        # if 'embed_tokens' in name:
        #     print("embed_tokens ->", name, ":", param.requires_grad)


    def into_million(val):
        return f"{val / 1000 / 1000 :.2f} million"

    # print("LoRA adapter added.")
    print(f"Total LoRA params: {into_million(lora_param)} ({(lora_param/non_lora_param)*100:.2f} %) = {into_million(lora_param)}")
    print(f"Total LoRA layers: {lora_layers}")
    print(f"Approx size: {lora_param * 2e-6:.2f} mb")

# %%
print(f"Model took {model.get_memory_footprint()/1e9:.2f} GB of space (with buffer)")
if lora_r:
    print("Are LoRA weight of embed_tokens and lm_head same?", torch.equal(model.base_model.model.model.embed_tokens.modules_to_save["smolthink"].weight, model.base_model.model.lm_head.modules_to_save["smolthink"].weight))
    model.base_model.model.model.embed_tokens.modules_to_save["smolthink"].weight = model.base_model.model.lm_head.modules_to_save["smolthink"].weight
    print("LoRA embed_tokens and lm_head sharing the same memory?", model.base_model.model.model.embed_tokens.modules_to_save["smolthink"].weight.data.data_ptr() == model.base_model.model.lm_head.modules_to_save["smolthink"].weight.data.data_ptr())
    print("Do model embed_tokens and lm_head sharing same memory?", model.base_model.model.model.embed_tokens.original_module.weight.data.data_ptr() == model.base_model.model.lm_head.original_module.weight.data.data_ptr())
else:
    print("Are LoRA weight of embed_tokens and lm_head same?", torch.equal(model.model.embed_tokens.weight, model.lm_head.weight))
    print("Do model embed_tokens and lm_head sharing same memory?", model.model.embed_tokens.weight.data.data_ptr() == model.lm_head.weight.data.data_ptr())

# %%
def length_filter(data, limit):
    # if data['thought_len'] + data['answer_len'] > 896:
        # return False
    return 0 < data['thought_len'] <= limit and 0 < data['answer_len']

if not dataset:
    def openthought_code(data):
        thought_len, answer_len = 0, 0

        reply = data['output']
        thought = re.findall(r"<thoughts>(.*?)</thoughts>", reply, re.DOTALL)
        thought = ''.join(thought).strip()
        thought_len += len(thought.split())
        
        end_tag = "</thoughts>"
        answer = reply[reply.find(end_tag)+len(end_tag):]
        answer = answer.strip()
        answer_len += len(answer.split()) #len(tokenizer.encode(answer))

        if end_tag not in reply:
            answer_len = 0

        final_answer = f"<think>\n{thought}\n</think>\n<answer>\n{answer}\n</answer>"
        final_answer = final_answer.replace("<thoughts>", "").replace("</thoughts>", "")
        
        output_data = {
            'thought_len': thought_len,
            'answer_len': answer_len,
            'conversations': [
                {"role": "user", 'content': data['input']},
                {"role": "assistant", 'content': final_answer}
            ]
        }

        return output_data

    openthought_dataset = load_dataset("XeTute/Open-Coding-Thoughts")['train']
    print("Dataset length:", len(openthought_dataset))
    openthought_dataset = openthought_dataset.map(openthought_code)
    openthought_dataset = openthought_dataset.map(lambda x: {"conversations": tokenizer.apply_chat_template(x['conversations'], tools=None, tokenize=False)})
    openthought_dataset = openthought_dataset.select(range(50))
    print("OpenThought dataset length (after filter):", len(openthought_dataset))
    # print(openthought_dataset[0]['conversations'])

# %%
if not dataset:
    def r1distillsft_conv(data):
        thought_len, answer_len = 0, 0
        for idx, conv in enumerate(data['reannotated_messages']):
            # print(conv)
            role = conv['role']
            if role == 'assistant':
                reply = data['reannotated_messages'][idx]['content']
                # print(reply)
                thought = re.findall(r"<think>(.*?)</think>", reply, re.DOTALL)
                thought = ''.join(thought).strip()
                thought_len += len(thought.split()) #len(tokenizer.encode(thought))

                end_tag = "</think>"
                if end_tag in reply:
                    answer = reply[reply.find(end_tag)+len(end_tag):]
                    answer = answer.strip()
                else:
                    answer = ''
                if thought.lower() == answer.lower():
                    answer = ''
                # print("Think:", thought)
                # print("Answer:", answer)
                # print("----")
                answer_len += len(answer.split()) #len(tokenizer.encode(answer))
                data['reannotated_messages'][idx]['content'] = f"<think>\n{thought}\n</think>\n<answer>\n{answer}\n</answer>"

        if 'system' in data:
            del data['system']
        data['thought_len'] = thought_len
        data['answer_len'] = answer_len
        return data

    r1_dataset = load_dataset("ServiceNow-AI/R1-Distill-SFT", "v1")['train']
    r1_dataset.shuffle(123)
    r1_dataset = r1_dataset.select(range(50_000, 53_000)) # Prev: 90_000
    r1_dataset = r1_dataset.map(r1distillsft_conv)
    r1_dataset = r1_dataset.filter(lambda x: length_filter(x, 256))
    delete_keys = list(r1_dataset.column_names)
    r1_dataset = r1_dataset.map(lambda x: {"conversations": tokenizer.apply_chat_template(x['reannotated_messages'], tools=None, tokenize=False)})
    r1_dataset = r1_dataset.remove_columns(delete_keys)
    r1_dataset = r1_dataset.select(range(100))
    print("R1-distill dataset length (after filter):", len(r1_dataset))

# %%
if not dataset:
# if True:
    def extract_tag(input_str, tag):
        tool_def = re.findall(f"<{tag}>(.*?)</{tag}>", input_str, re.DOTALL)
        tool_def = map(str.strip, tool_def)
        tool_def = filter(lambda x: len(x) > 0, tool_def)
        return list(tool_def)

    def hermes_fc_thinking(raw_data):
        data = deepcopy(raw_data['conversations'])
        seq = []
        tool_def = None
        tool_names = None
        for d in data:
            if d['role'] == 'system':
                tool_def = extract_tag(d['content'], 'tools')
                if len(tool_def) != 0:
                    try:
                        tool_def = ast.literal_eval(tool_def[0])
                        tool_names = [tool['function']['name'] for tool in tool_def]
                        continue
                    except Exception as E:
                        return {"conversations": ""}
                else:
                    return {"conversations": ""}

            seq.append({})
            seq[-1]['role'] = {"human": "user", "model": "assistant", "system": "system", "tool": "tool"}[d['role']]
            seq[-1]['content'] = d['content']
            if seq[-1]['role'] == 'assistant':
                seq[-1]['content'] = seq[-1]['content'].replace('<think>', '<think>\n')
                seq[-1]['content'] = seq[-1]['content'].replace('</think>', '</think>\n')
                # seq[-1]['content'] = seq[-1]['content'].replace('<tool_call>\n', '<tool_call>\n[')
                # seq[-1]['content'] = seq[-1]['content'].replace('\n</tool_call>', ']\n</tool_call>')
                tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", seq[-1]['content'], re.DOTALL)
                seq[-1]['tool-call'] = []
                if tool_calls:
                    # print(tool_calls, tool_def)
                    for tool_call in tool_calls:
                        try:
                            tool_call = json.loads(tool_call.strip().replace("'", '"'))
                            if tool_call['name'] not in tool_names:
                                raise NotImplementedError
                            seq[-1]['tool_call'] = tool_call
                        except Exception as E:
                            return {"conversations": ""}
                
                if "<think>" not in seq[-1]['content']:
                    seq[-1]['content'] = f"<think>\n</think>\n<answer>\n{seq[-1]['content']}\n</answer>"
            if seq[-1]['role'] == 'tool':
                seq[-1]['content'] = seq[-1]['content'].replace("<tool_response>", "")
                seq[-1]['content'] = seq[-1]['content'].replace("</tool_response>", "")
                seq[-1]['content'] = seq[-1]['content'].strip()
            # seq[-1]['content'] = d['value']
        
        random.shuffle(tool_def)
        ret = tokenizer.apply_chat_template(seq, tools=tool_def, tokenize=False, add_generation_prompt=False) #+ "<tool_call>\n"
        return {"conversations": ret}

    fc_dataset = load_dataset("Jofthomas/hermes-function-calling-thinking-V1")['train']
    # fc_dataset = fc_dataset.select(range(len(fc_dataset)//2))
    fc_dataset = fc_dataset.map(hermes_fc_thinking)
    fc_dataset = fc_dataset.filter(lambda x: len(x['conversations']) > 0)
    fc_dataset = fc_dataset.select(range(50))
    print("Function calling dataset length (after filter):", len(fc_dataset))

# %%
if not dataset:
    def generalreason_conv(data):
        history = None
        data['empty'] = 'true'
        if 'prev_message' in data:
            history = data['prev_message']
        if not history:
            history = []

        if history and history[0]['role'] == 'system':
            del history[0]
        
        for idx, h in history:
            if history[idx]['role'] == 'assistant':
                history[idx]['content'] = f"<think>\n</think>\n<answer>\n{history[idx]['content']}\n</answer>"
        
        if data['model_reasoning']:
            data['empty'] = 'false'
            think = f"<think>\n{data['model_reasoning'].strip()}\n</think>"
        else:
            think = "<think>\n</think>"
        answer = f"<answer>\n{data['model_answer'].strip()}\n</answer>"

        history.append({'role': 'user', 'content': data['question']})
        history.append({'role': 'assistant', 'content': think+"\n"+answer})

        data['history'] = history
        return data

    def get_ascii(data_str):
        try:
            data_str.encode('ascii')
        except Exception as E:
            return False
        return True

    from collections import Counter
    genreason_dataset = load_dataset("GeneralReasoning/GeneralThought-195K")['train']
    genreason_dataset = genreason_dataset.filter(lambda x: x['question_license'] in ['MIT', 'Apache-2.0'])
    # genreason_dataset = genreason_dataset.filter(lambda x: x['task'] in ['Open Conversations', 'Explanation'])
    # genreason_dataset = genreason_dataset.filter(lambda x: get_ascii(x['question']) if x['question'] else False)
    genreason_dataset = genreason_dataset.filter(lambda x: get_ascii(x['model_answer']) if x['model_answer'] else False)
    genreason_dataset = genreason_dataset.filter(lambda x: len(x['model_reasoning'].strip().split()) < 256 if x['model_reasoning'] else True)
    # genreason_dataset = genreason_dataset.filter(lambda x: len(x['question'].strip().split()) < 256 if x['question'] else False)

    genreason_dataset = genreason_dataset.map(generalreason_conv)
    delete_keys = list(genreason_dataset.column_names)
    genreason_dataset = genreason_dataset.map(lambda x: {"conversations": tokenizer.apply_chat_template(x['history'], tools=None, tokenize=False)})
    # print(Counter(genreason_dataset['empty']))
    genreason_dataset = genreason_dataset.remove_columns(delete_keys)
    genreason_dataset = genreason_dataset.select(range(50))
    print("General reason dataset length:", len(genreason_dataset))


# %%
if not dataset:
    def process(data):
        for idx, message in enumerate(data['messages']):
            if message['role'] != 'assistant': continue
            content = message['content']
            tag = "</think>"
            pos = content.find(tag)
            answer = content[pos+len(tag):].strip()
            data['messages'][idx]['content'] = content[:pos].strip() + f"\n</think>\n<answer>\n{answer}\n</answer>"
        return data

    # Login using e.g. `huggingface-cli login` to access this dataset
    codeforces_cot = load_dataset("open-r1/codeforces-cots", "solutions_py_decontaminated")['train']#.select(range(500))
    codeforces_cot = codeforces_cot.filter(lambda x: len(str(x['messages'])) < 8000)
    delete_keys = list(codeforces_cot.column_names)
    codeforces_cot = codeforces_cot.map(process)
    codeforces_cot = codeforces_cot.map(lambda x: {"conversations": tokenizer.apply_chat_template(x['messages'], tools=None, tokenize=False)})
    codeforces_cot = codeforces_cot.remove_columns(delete_keys)
    codeforces_cot = codeforces_cot.select(range(50))
    print("Codeforces CoT dataset length:", len(codeforces_cot))

# %%
if not dataset:
    def process(data):
        seq = [
            {'role': 'user', 'content': data['question']}, 
            {'role': 'assistant', 'content': f"<think>\n</think>\n<tool_call>\n{{'name': 'web_search', 'arguments': {{'search_str': '{data['search_str']}'}}}}</tool_call>"},
            {'role': 'tool', 'content': data['search_results'] + f"\n\nUser question: {data['question']}"},
            {'role': 'assistant', 'content': f"<think>{data['think'].strip()}</think>\n<answer>\n{data['answer'].strip()}\n</answer>"}
        ]

        # data["messages"] = seq
        data["conversations"] = tokenizer.apply_chat_template(seq, tools=[webtool_def], tokenize=False)
        return data

    websearch_data = []
    with open("/Users/ohi/Documents/GitHub/PersonalAssistant/datagen/search_data.jsonl", "r") as f:
        for line in f:
            websearch_data.append(json.loads(line))
    websearch_data = Dataset.from_list(websearch_data)
    delete_keys = list(websearch_data.column_names)
    websearch_data = websearch_data.map(process)
    websearch_data = websearch_data.remove_columns(delete_keys)
    print("WebSearch dataset length:", len(websearch_data))


# %%
if not dataset:
    # if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)
    with open(os.path.join(SAVE_PATH, 'dataset_example.log'), "w") as f:
        for k, d in [
            ("OpenThought", openthought_dataset), 
            ("R1", r1_dataset), 
            ("Function Calling", fc_dataset), 
            ("General Reason", genreason_dataset), 
            ("Codeforce CoT", codeforces_cot), 
            ("WebSearch", websearch_data)
        ]:
            f.write(f"\n{k} length: {len(d)}\n")
            f.write(f"{k}\n{'-'*20}\n{d[0]['conversations']}\n{'='*20}\n\n")

    dataset = concatenate_datasets([openthought_dataset, r1_dataset, fc_dataset, genreason_dataset, codeforces_cot, websearch_data])
    dataset = dataset.shuffle(12)
    # dataset.save_to_disk("/Users/ohi/Documents/GitHub/PersonalAssistant/datasets/merged_dataset")
    del openthought_dataset, r1_dataset, fc_dataset, genreason_dataset, codeforces_cot, websearch_data


# %%
from tqdm import tqdm

class DatasetGen_v1(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.cache = None
        self.cache_idx = -1
        self.cache_len = 0
        self.indices = []
        self._get_len()

    def _get_len(self):
        print("Computing dataset length")
        for idx in tqdm(range(len(self.dataset))):
            self.gen(idx)
            for i in range(self.cache_len):
                self.indices.append((idx, i))
        print("Total length of data:", len(self.indices))
    

    def __len__(self):
        return len(self.indices)
    
    def gen(self, idx):
        self.cache = self.dataset[idx]['conversations'].rstrip()
        self.cache = self.tokenizer(
            self.cache,
            max_length=CONTEXT_LEN,
            truncation=True,
            return_overflowing_tokens=True, # Return the overflowing tokens
            stride=CONTEXT_LEN // 8,
            padding='max_length'
        )
        self.cache_idx = idx
        self.cache_len = len(self.cache['input_ids'])
    
    def __getitem__(self, idx):
        p, q = self.indices[idx]
        if self.cache_idx != p:
            self.gen(p)
        
        input_ids = self.cache['input_ids'][q]
        attention_mask = self.cache['attention_mask'][q]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


# %%
DS_LEN = len(dataset)
CONTEXT_LEN = 1024 * 3 #832 # 1024
TEST_DS_LEN = 50# 250

print("Total dataset len:", DS_LEN)
train_ds = DatasetGen_v1(
    dataset=dataset.select(range(0, DS_LEN-TEST_DS_LEN)), 
    tokenizer=tokenizer
)
test_ds = DatasetGen_v1(
    dataset=dataset.select(range(int(DS_LEN-TEST_DS_LEN), DS_LEN)), 
    tokenizer=tokenizer
)

# print(tokenizer.decode(train_ds[0]['input_ids']))
# print(json.dumps(train_ds.detokenize(0), indent=2))
# print(train_ds.detokenize(99)['input'])
# print(train_ds[0].keys())

# %%
from transformers import (
    # DataCollatorForSeq2Seq,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)

# Train on completion only
# Ref: https://huggingface.co/docs/trl/en/sft_trainer#train-on-completions-only

data_collator = DataCollatorForLanguageModeling(
    # model = model,
    tokenizer = tokenizer,
    mlm=False,
    # max_length = CONTEXT_LEN,
    # pad_to_multiple_of = 2,
    # padding = 'max_length'
)

SAVE_STEPS = 400
training_args = TrainingArguments(
    output_dir=SAVE_PATH,
    # SmolLM2 SFT learning rate: 3.0 * 10-4
    learning_rate = 5e-5,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.2,
    warmup_ratio = 0.1,
    max_grad_norm=0.1,
    logging_steps=20,
    max_steps=len(train_ds),
    save_steps = SAVE_STEPS, #200 // (CONTEXT_LEN // 512),
    save_total_limit=3,
    lr_scheduler_type = "cosine",
    # Memory reduction
    optim = "adamw_torch",    # adamw_torch, adafactor
    # Memory reduction
    bf16 = True,
    bf16_full_eval=True,
    per_device_train_batch_size = 1,
    per_device_eval_batch_size = 1,
    gradient_accumulation_steps = 1, #2, # Increase to 4 for smoother training
    torch_empty_cache_steps=SAVE_STEPS,
    num_train_epochs=1,
    logging_strategy='steps',
    eval_strategy='steps',
    eval_steps= SAVE_STEPS, #200 // (CONTEXT_LEN // 512),
    save_strategy='steps', #'steps', 'no', 'best',
    push_to_hub=False,
    report_to="none",
    dataloader_pin_memory=True,
    # dataloader_num_workers=1,
    # Gradient checkpointing - reduces memory in MPS
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# %%
import transformers, gc

class MpsCacheClearCallback(transformers.TrainerCallback):
    def __clearmem(self):
        gc.collect()
        torch.mps.empty_cache()
        gc.collect()
        # print("\nMEMORY CLEARED\n")
    #def on_step_begin(self, *args, **kwargs):      self.__clearmem()
    def on_step_end(self, *args, **kwargs):        self.__clearmem()
    #def on_substep_end(self, *args, **kwargs):     self.__clearmem()
    #def on_evaluate(self, *args, **kwargs):        self.__clearmem()
    #def on_optimizer_step(self, *args, **kwargs):  self.__clearmem()
    #def on_predict(self, *args, **kwargs):         self.__clearmem()
    #def on_prediction_step(self, *args, **kwargs): self.__clearmem()
gc.collect()

class WeightTieCallback(transformers.TrainerCallback):
    # def on_init_end(self, *args, **kwargs):
    def on_train_begin(self, *args, **kwargs):
        model.base_model.model.model.embed_tokens.modules_to_save["smolthink"].weight = model.base_model.model.lm_head.modules_to_save["smolthink"].weight
        print("------ Weight tied ------")

# %%
trainer = Trainer(
    model = model,
    processing_class = tokenizer,
    args = training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=data_collator,
    # callbacks=[MpsCacheClearCallback()]
    # callbacks=[WeightTieCallback()]
)

print("Model save path:", SAVE_PATH)
model.config.use_cache = False
try:
    trainer.train(resume_from_checkpoint=True)
except ValueError as E:
    print("No checkpoint found")
    trainer.train(resume_from_checkpoint=False)
