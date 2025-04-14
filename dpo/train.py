# Resources:
# https://www.philschmid.de/dpo-align-llms-in-2024-with-trl#2-create-and-prepare-the-dataset
# https://www.philschmid.de/rl-with-llms-in-2025-dpo
# https://huggingface.co/docs/trl/dpo_trainer
# http://ethen8181.github.io/machine-learning/deep_learning/llm/rlhf/dpo.html
# https://medium.com/@ufuk.birbiri/llm-fine-tuning-with-direct-preference-optimization-dpo-with-code-12ed92259215
# Online DRO: https://huggingface.co/docs/trl/online_dpo_trainer

# %%
import torch
from trl import DPOTrainer
from datasets import load_dataset
import sys

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments
)

SIZE = "360M"
MODEL_PATH = f"HuggingFaceTB/SmolLM2-{SIZE}-Instruct"
SAVE_STEPS = 20

# %%
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

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)
# Doing this could be risky
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left' # to prevent errors with FA
tokenizer.truncation_side = 'left' # to prevent cutting off last generation

# %%
# Load dataset from the hub
dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train")
dataset = dataset.shuffle().select(range(3))


# %%

def create_triplet(data, tokenizer):
    def get_last_reply(data, index=-1):
        if data[index]["role"] == "assistant":
            return [data[index]]
        else:
            return get_last_reply(data, index-1)
    
    prompt_messages = data["chosen"][:-1]
    chosen = tokenizer.apply_chat_template(
        [{"role": "system", "content": ""}] + \
        data['chosen'], tokenize=False
    ).replace("<|im_start|>system\n<|im_end|>\n", "")

    rejected = chosen = tokenizer.apply_chat_template(
        [{"role": "system", "content": ""}] + \
        data['rejected'], tokenize=False
    ).replace("<|im_start|>system\n<|im_end|>\n", "")

    return {
        "prompt": tokenizer.apply_chat_template(prompt_messages, tokenize=False),
        "chosen": chosen,
        "rejected": rejected
    }

train_ds = dataset.map(create_triplet, fn_kwargs={"tokenizer": tokenizer})
train_ds = train_ds.train_test_split(test_size=0.05)

print(train_ds['train'][0]["prompt"])
print("-+"*10)
print(train_ds['train'][0]["chosen"])
print("-+"*10)
print(train_ds['train'][0]["rejected"])
sys.exit()


# %%
 
args = TrainingArguments(
    output_dir="weights/test-dpo",
    
    learning_rate= 1e-6,
    lr_scheduler_type="cosine",
    adam_beta1=0.9,
    adam_beta2=0.95,
    optim="adamw_torch",    # adamw_torch, adafactor
    weight_decay=0.01,
    max_grad_norm=1.0,      # Reduce to 0.1 if NaN
    warmup_ratio= 0.1,

    num_train_epochs=1,
    logging_strategy="steps",
    eval_strategy="steps",
    eval_steps=SAVE_STEPS,
    save_strategy="steps",  #'steps', 'no', 'best',
    logging_steps=20,
    max_steps=len(train_ds),
    save_steps=SAVE_STEPS,
    save_total_limit=3,

    # Memory reduction
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    bf16_full_eval=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    torch_empty_cache_steps=SAVE_STEPS,
    push_to_hub=False,
    report_to="none",
    dataloader_pin_memory=True,

    # Speedups
    torch_compile=True,
    torch_compile_backend='aot_eager'
)


trainer = DPOTrainer(
    model,
    ref_model=None, # set to none since we use peft
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_length=1024,
    max_prompt_length=512,
    beta=0.5,                   # The beta factor in DPO loss. Higher beta means less divergence
    loss_type="sigmoid",        # The loss type for DPO
)

# start training, the model will be automatically saved to the hub and the output directory
trainer.train()
