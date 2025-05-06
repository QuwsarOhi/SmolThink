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
from webtool.webtool import webtool_def
from sft.tokenizer import get_tokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"
lora_r = None  # 32
SIZE = "135M"
REASONING_LEN = 386

MODEL_PATH = f"/Users/ohi/Documents/models/SmolLM2-{SIZE}-Instruct"
SAVE_PATH = f"weights/SmolThink-{SIZE}-sft"
CONTEXT_LEN = 832
CONTEXT_STRIDE = 2
TEST_DS_LEN = 200
SAVE_STEPS = 400 #1000

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="mps",
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
    torch_dtype=torch.float32,
    trust_remote_code=True,
    use_cache=True,
    tie_word_embeddings=True,
)

## Gradient checkpointing
# model.gradient_checkpointing_enable(dict(use_reentrant=False))
model.gradient_checkpointing_disable()

tokenizer = get_tokenizer(MODEL_PATH)

streamer = TextStreamer(tokenizer, skip_prompt=True)

print(
    tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I am fine"},
        ],
        tokenize=False,
    )
)

## ----- Prompt Template Debugging ------
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
    },
]
print("\n-----\n")
print(tokenizer.apply_chat_template([
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "<tool_call>[retrieve_payment_date(12)]</tool_call>"},
    {"role": "tool", "content": "12/12/12"},
    {"role": "assistant", "content": "12/12/12"}
], tools=tools, tokenize=False))


# Only if adding new tokens
# model.resize_token_embeddings(len(tokenizer))

# PEFT ref: https://huggingface.co/docs/transformers/en/peft
# r: rank dimension for LoRA update matrices (smaller = more compression)
# lora_alpha: scaling factor for LoRA layers (higher = stronger adaptation)
# lora_dropout: dropout probability for LoRA layers (helps prevent overfitting)

if lora_r:
    SAVE_PATH += f"-r{lora_r}"
    peft_config = peft.LoraConfig(
        r=lora_r,  # 64
        lora_alpha=2 * lora_r,  # alpha = 4 * r | 2 * r
        lora_dropout=0.1,
        target_modules="all-linear",
        modules_to_save=["embed_tokens", "lm_head"],
        # use_rslora=True,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian",
        inference_mode=False,
    )
    model = peft.get_peft_model(
        model, peft_config, adapter_name="smolthink", autocast_adapter_dtype=False
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

if lora_r:
    print(
        "Are LoRA weight of embed_tokens and lm_head same?",
        torch.equal(
            model.base_model.model.model.embed_tokens.modules_to_save[
                "smolthink"
            ].weight,
            model.base_model.model.lm_head.modules_to_save["smolthink"].weight,
        ),
    )
    model.base_model.model.model.embed_tokens.modules_to_save[
        "smolthink"
    ].weight = model.base_model.model.lm_head.modules_to_save["smolthink"].weight
    print(
        "LoRA embed_tokens and lm_head sharing the same memory?",
        model.base_model.model.model.embed_tokens.modules_to_save[
            "smolthink"
        ].weight.data.data_ptr()
        == model.base_model.model.lm_head.modules_to_save[
            "smolthink"
        ].weight.data.data_ptr(),
    )
    print(
        "Do model embed_tokens and lm_head sharing same memory?",
        model.base_model.model.model.embed_tokens.original_module.weight.data.data_ptr()
        == model.base_model.model.lm_head.original_module.weight.data.data_ptr(),
    )
else:
    print(
        "Are LoRA weight of embed_tokens and lm_head same?",
        torch.equal(model.model.embed_tokens.weight, model.lm_head.weight),
    )
    print(
        "Do model embed_tokens and lm_head sharing same memory?",
        model.model.embed_tokens.weight.data.data_ptr()
        == model.lm_head.weight.data.data_ptr(),
    )


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
            return_overflowing_tokens=False,  # Return the overflowing tokens
            # stride=CONTEXT_LEN // CONTEXT_STRIDE,
            padding="max_length",
        )
        self.cache["input_ids"] = [self.cache["input_ids"]]
        self.cache["attention_mask"] = [self.cache["attention_mask"]]
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


dataset = load_dataset("json", data_files=f"datasets/dataset_cot{REASONING_LEN}.jsonl", split='train')
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
    learning_rate=5e-5,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.3, # Increased from 0.2 -> 0.3
    warmup_ratio= 500 / len(train_ds), #0.1,
    max_grad_norm=0.1,
    logging_steps=20,
    max_steps=len(train_ds),
    save_steps=SAVE_STEPS,  # 200 // (CONTEXT_LEN // 512),
    save_total_limit=3,
    lr_scheduler_type="cosine",
    # Memory reduction
    optim="adamw_torch",  # adamw_torch, adafactor
    # Memory reduction
    # bf16=True,
    # bf16_full_eval=True,
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
    # torch_compile_backend='reduce_overhead' #'aot_eager'
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


class WeightTieCallback(transformers.TrainerCallback):
    # def on_init_end(self, *args, **kwargs):
    def on_train_begin(self, *args, **kwargs):
        model.base_model.model.model.embed_tokens.modules_to_save[
            "smolthink"
        ].weight = model.base_model.model.lm_head.modules_to_save["smolthink"].weight
        print("------ Weight tied ------")



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

# trainer.train(resume_from_checkpoint=False)
