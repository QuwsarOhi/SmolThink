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

import ast
import json
import os

# from safetensors.torch import load_model, save_model
import random
import re
import gc
from copy import deepcopy
from tqdm import tqdm

import peft
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TextStreamer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    DataCollatorWithFlattening
)

import transformers
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
# from webtool.webtool import webtool_def


os.environ["TOKENIZERS_PARALLELISM"] = "false"
lora_r = 32
LORA_NAME = "smolfc"
SIZE = "360M"

MODEL_PATH = f"/Users/ohi/Documents/models/SmolLM2-{SIZE}-Instruct" #f"HuggingFaceTB/SmolLM2-{SIZE}-Instruct"
SAVE_PATH = f"weights/SmolThink-{SIZE}-sft"

CONTEXT_LEN = 1024 * 1
CONTEXT_STRIDE = 2

TEST_DS_LEN = 200
SAVE_STEPS = 400

dataset = load_from_disk(f"datasets/dataset_ctx{CONTEXT_LEN}_cot256")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="cpu",
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_cache=False,
    tie_word_embeddings=True,
).to("mps")


## Gradient checkpointing
# model.gradient_checkpointing_enable(dict(use_reentrant=False))
model.gradient_checkpointing_disable()

chat_template = """<empty_output>{%- if tools %}
    {{- '<|im_start|>system\\n' }}
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
        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
    {%- else %}
        {{- '<|im_start|>system\\nYou are a helpful AI assistant named SmolThink. First plan/reason/code/validate inside <think></think> tag and provide final answer to user query inside <answer></answer> tag.<|im_end|>\\n' }}
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

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="cpu",
    low_cpu_mem_usage=True,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_cache=True,
    tie_word_embeddings=True,
).to("mps")


## Gradient checkpointing
# model.gradient_checkpointing_enable(dict(use_reentrant=False))
model.gradient_checkpointing_disable()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    # add_bos_token=True,
    # add_eos_token=True,
)
tokenizer.chat_template = chat_template
tokenizer.bos_token = "<empty_output>"
tokenizer.eos_token = "<|im_end|>"
tokenizer.pad_token = "<|endoftext|>"
tokenizer.unk_token = "<|endoftext|>"
# tokenizer.padding_side = "left"
# tokenizer.truncation_side = "left"


# https://stackoverflow.com/questions/69609401/suppress-huggingface-logging-warning-setting-pad-token-id-to-eos-token-id
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id

assert tokenizer.bos_token_id == 16
assert tokenizer.eos_token_id == 2
assert tokenizer.pad_token_id == 0
assert tokenizer.unk_token_id == 0

streamer = TextStreamer(tokenizer, skip_prompt=True)

if lora_r:
    SAVE_PATH += f"-r{lora_r}"
    peft_config = peft.LoraConfig(
        r=lora_r,  # 64
        lora_alpha=2 * lora_r,  # alpha = 4 * r | 2 * r
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], # "all-linear",
        modules_to_save=None, #["embed_tokens", "lm_head"],
        # use_rslora=True,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian",
        inference_mode=False,
    )

    model = peft.get_peft_model(
        model, peft_config, adapter_name=LORA_NAME, autocast_adapter_dtype=True
    )

    # Sanity check
    non_lora_param = 0
    lora_param = 0
    lora_layers = 0
    for name, param in model.named_parameters():
        if "lora" in name:
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
        return f"{val / 1000 / 1000:.2f} million"

    # print("LoRA adapter added.")
    print(
        f"Total LoRA params: {into_million(lora_param)} ({(lora_param / non_lora_param) * 100:.2f} %) = {into_million(lora_param)}"
    )
    print(f"Total LoRA layers: {lora_layers}")
    print(f"Approx size: {lora_param * 2e-6:.2f} mb")


print(f"Model took {model.get_memory_footprint() / 1e9:.2f} GB of space (with buffer)")

# if lora_r:
#     print(
#         "Are LoRA weight of embed_tokens and lm_head same?",
#         torch.equal(
#             model.base_model.model.model.embed_tokens.modules_to_save[
#                 LORA_NAME
#             ].weight,
#             model.base_model.model.lm_head.modules_to_save[LORA_NAME].weight,
#         ),
#     )
#     model.base_model.model.model.embed_tokens.modules_to_save[
#         LORA_NAME
#     ].weight = model.base_model.model.lm_head.modules_to_save[LORA_NAME].weight
#     print(
#         "LoRA embed_tokens and lm_head sharing the same memory?",
#         model.base_model.model.model.embed_tokens.modules_to_save[
#             LORA_NAME
#         ].weight.data.data_ptr()
#         == model.base_model.model.lm_head.modules_to_save[
#             LORA_NAME
#         ].weight.data.data_ptr(),
#     )
#     print(
#         "Do model embed_tokens and lm_head sharing same memory?",
#         model.base_model.model.model.embed_tokens.original_module.weight.data.data_ptr()
#         == model.base_model.model.lm_head.original_module.weight.data.data_ptr(),
#     )
# else:
#     print(
#         "Are LoRA weight of embed_tokens and lm_head same?",
#         torch.equal(model.model.embed_tokens.weight, model.lm_head.weight),
#     )
#     print(
#         "Do model embed_tokens and lm_head sharing same memory?",
#         model.model.embed_tokens.weight.data.data_ptr()
#         == model.lm_head.weight.data.data_ptr(),
#     )


# print(model)
# import sys
# sys.exit()

# print(
#     tokenizer.apply_chat_template(
#         [
#             {"role": "user", "content": "How are you?"},
#             {"role": "assistant", "content": "I am fine"},
#         ],
#         tokenize=False,
#     )
# )

# SYS_TEMPLATE = """You are an expert in composing functions. You are given a question and a set of possible functions. 
# Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
# If none of the functions can be used, point it out and refuse to answer. 
# If the given question lacks the parameters required by the function, also point it out.

# You have access to the following tools:
# <tools>{tools}</tools>

# The output MUST strictly adhere to the following format, and NO other text MUST be included.
# The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make the tool calls an empty list '[]'.
# <tool_call>[
# {{"name": "func_name1", "arguments": {{"argument1": "value1", "argument2": "value2"}}}},
# ... (more tool calls as required)
# ]</tool_call>"""


# def tool_shuffle(tool):
#     if not isinstance(tool, list):
#         tool = [tool]
#     if tool:
#         assert not isinstance(tool[0], str), f"Tool type should not be a str: type-{type(tool[0])}"
#     random.shuffle(tool)
#     reps = [str(tool)]
#     for c in [None, 1, 2, 3, 4]:
#         reps.append(json.dumps(tool, indent=c))
#     return random.choice(reps)


# def tool_call_process(data):
#     new_data = {
#         'prompt': '',
#         'valid': False,
#         'tool': '',
#         'tool_call': '',
#         'source': 'HuggingFaceTB/smoltalk/apigen-80k'
#     }
#     tool_def = None
#     pattern = re.compile(r'<tools>(.*?)</tools>', re.DOTALL)
#     tool_def = pattern.match(data['messages'][0]['content'])

#     try:
#         content = data['messages'][0]['content']
#         tool_def = re.findall(r"<tools>(.*?)</tools>", content, re.DOTALL)[0]
#         tool_def = json.loads(tool_def)
#         new_data['tool'] = json.dumps(tool_def)
#     except:
#         return new_data

#     seq = [{"role": "system", "content": SYS_TEMPLATE.format(tools=tool_shuffle(tool_def))}]
    
#     for s in data['messages']:
#         if s['role'] == 'system': continue
#         if s['role'] == 'user':
#             seq.append(s)
#         elif s['role'] == 'assistant':
#             tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", s['content'], re.DOTALL)
#             if tool_calls:
#                 tool_calls = json.loads(tool_calls[0])
#                 tool_calls = json.dumps(tool_calls, indent=None) #random.choice([0, 2, 4]))
#                 new_data['tool_call'] = tool_calls
#                 seq.append({"role": "assistant", "content": f"<tool_call>{tool_calls}</tool_call>"})   
#             else:
#                 # s['content'] = f"s['content']
#                 seq.append(s)
#         else:
#             ValueError(f"Unknown role: {s['role']}")
    
#     # new_data['valid'] = True
#     tool_def = [{"type": "function", "function": e} for e in tool_def]
#     new_data['conversations'] = tokenizer.apply_chat_template(seq, tokenize=False, add_generation_prompt=False)
#     return new_data

# dataset = load_dataset("HuggingFaceTB/smoltalk", "apigen-80k")['train'] #['train']#.select(range(30))#.select(range(2))
# dataset = dataset.map(tool_call_process)
# # smoltalk_fc_dataset = smoltalk_fc_dataset.remove_columns(smoltalk_fc_dataset.column_names)
# # smoltalk_fc_dataset = smoltalk_fc_dataset.filter(lambda x: x['valid'] == True)
# print("Smoltalk function calling dataset length (after filter):", len(dataset))

# print("-"*20)
# print(dataset[0]['conversations'])
# print("-"*20)

# import sys
# sys.exit()

## ----- Prompt Template Debugging ------
# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "retrieve_payment_status",
#             "description": "Get payment status of a transaction",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "transaction_id": {
#                         "type": "string",
#                         "description": "The transaction id.",
#                     }
#                 },
#                 "required": ["transaction_id"],
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "retrieve_payment_date",
#             "description": "Get payment date of a transaction",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "transaction_id": {
#                         "type": "string",
#                         "description": "The transaction id.",
#                     }
#                 },
#                 "required": ["transaction_id"],
#             },
#         },
#     },
# ]
# print("\n-----\n")
# print(tokenizer.apply_chat_template([
#     {"role": "user", "content": "How are you?"},
#     {"role": "assistant", "content": "<tool_call>[retrieve_payment_date(12)]</tool_call>"},
#     {"role": "tool", "content": "12/12/12"},
#     {"role": "assistant", "content": "12/12/12"}
# ], tools=tools, tokenize=False))


# PEFT ref: https://huggingface.co/docs/transformers/en/peft
# r: rank dimension for LoRA update matrices (smaller = more compression)
# lora_alpha: scaling factor for LoRA layers (higher = stronger adaptation)
# lora_dropout: dropout probability for LoRA layers (helps prevent overfitting)


class DatasetGen_v1(torch.utils.data.Dataset):
    ''' 
    Memory optimized dataset. Only converts into tokens when necessary. 
    Context stride helps the LLM to go through the sequence
    '''
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.cache = None
        self.cache_idx = -1
        self.cache_len = 0
        self.indices = [(i, 0) for i in range(len(self.dataset))]
        # self._get_len()

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
        self.cache = self.dataset[idx]["conversations"].rstrip()
        self.cache = self.tokenizer(
            self.cache,
            max_length=CONTEXT_LEN,
            truncation=True,
            return_overflowing_tokens=True,  # Return the overflowing tokens
            stride=CONTEXT_LEN // CONTEXT_STRIDE,
            padding="max_length",
        )
        self.cache_idx = idx
        self.cache_len = len(self.cache["input_ids"])

    def __getitem__(self, idx):
        p, q = self.indices[idx]
        if self.cache_idx != p:
            self.gen(p)

        input_ids = self.cache["input_ids"][q]
        attention_mask = self.cache["attention_mask"][q]
        labels = [-100 if t == self.tokenizer.pad_token_id else t for t in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


DS_LEN = len(dataset)
print("Total dataset len:", DS_LEN)
# dataset = dataset.train_test_split(test_size=TEST_DS_LEN/DS_LEN)

train_ds = DatasetGen_v1(
    dataset=dataset.select(range(0, DS_LEN - TEST_DS_LEN)), tokenizer=tokenizer
)

test_ds = DatasetGen_v1(
    dataset=dataset.select(range(int(DS_LEN - TEST_DS_LEN), DS_LEN)),
    tokenizer=tokenizer,
)

# train_ds
# print(tokenizer.decode(train_ds[0]['input_ids']))
# print(json.dumps(train_ds.detokenize(0), indent=2))
# print(train_ds.detokenize(99)['input'])
# print(train_ds[0].keys())
# )

# Train on completion only
# Ref: https://huggingface.co/docs/trl/en/sft_trainer#train-on-completions-only

# data_collator = DataCollatorForLanguageModeling(
#     # model = model,
#     tokenizer=tokenizer,
#     mlm=False,
#     # max_length = CONTEXT_LEN,
#     # pad_to_multiple_of = 2,
#     # padding = 'max_length'
# )

# model = torch.compile(model, mode='max-autotune') #mode='default/reduce-overhead/max-autotune')

training_args = TrainingArguments(
    output_dir=SAVE_PATH,
    # SmolLM2 SFT learning rate: 3.0 * 10-4
    learning_rate=1e-4, #5e-5,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.01, # Increased from 0.2 -> 0.3
    warmup_ratio= 500 / len(train_ds), #0.1,
    max_grad_norm=1.0,
    logging_steps=20,
    max_steps=len(train_ds),
    save_steps=SAVE_STEPS,  # 200 // (CONTEXT_LEN // 512),
    save_total_limit=3,
    lr_scheduler_type="cosine",
    # Memory reduction
    optim="adamw_torch",  # adamw_torch, adafactor
    # Memory reduction
    bf16=True,
    bf16_full_eval=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,  # 2, # Increase to 4 for smoother training
    torch_empty_cache_steps=SAVE_STEPS,
    num_train_epochs=1,
    logging_strategy="steps",
    eval_strategy="steps",
    eval_steps=SAVE_STEPS,  # 200 // (CONTEXT_LEN // 512),
    save_strategy="steps",  #'steps', 'no', 'best',
    push_to_hub=False,
    report_to="none",
    dataloader_pin_memory=True,
    # torch_compile=True,
    # torch_compile_backend='aot_eager'
    # dataloader_num_workers=1,
    # Gradient checkpointing - reduces memory in MPS
    # gradient_checkpointing=True,
    # gradient_checkpointing_kwargs={"use_reentrant": False},
)


class MpsCacheClearCallback(transformers.TrainerCallback):
    def __clearmem(self):
        gc.collect()
        torch.mps.empty_cache()
        gc.collect()
        # print("\nMEMORY CLEARED\n")

    # def on_step_begin(self, *args, **kwargs):      self.__clearmem()
    def on_step_end(self, *args, **kwargs):
        self.__clearmem()

    # def on_substep_end(self, *args, **kwargs):     self.__clearmem()
    # def on_evaluate(self, *args, **kwargs):        self.__clearmem()
    # def on_optimizer_step(self, *args, **kwargs):  self.__clearmem()
    # def on_predict(self, *args, **kwargs):         self.__clearmem()
    # def on_prediction_step(self, *args, **kwargs): self.__clearmem()


trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    # packing=True,
    # Data packing: https://huggingface.co/blog/packing-with-FA2
    data_collator=None, #DataCollatorWithFlattening(), #data_collator, # DataCollatorWithFlattening()
    # callbacks=[MpsCacheClearCallback()]
    # callbacks=[WeightTieCallback()]
)

print("Model save path:", SAVE_PATH)
model.config.use_cache = True
try:
    trainer.train(resume_from_checkpoint=True)
except ValueError:
    print("No checkpoint found")
    trainer.train(resume_from_checkpoint=False)
