# Resources:
# https://www.philschmid.de/dpo-align-llms-in-2024-with-trl#2-create-and-prepare-the-dataset
# https://www.philschmid.de/rl-with-llms-in-2025-dpo
# https://huggingface.co/docs/trl/dpo_trainer
# http://ethen8181.github.io/machine-learning/deep_learning/llm/rlhf/dpo.html
# https://medium.com/@ufuk.birbiri/llm-fine-tuning-with-direct-preference-optimization-dpo-with-code-12ed92259215

# %%
import torch
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    # TextStreamer,
    # DataCollatorForLanguageModeling,
    # Trainer,
    # TrainingArguments,
    # DataCollatorWithFlattening
)

SIZE = "360M"
MODEL_PATH = f"HuggingFaceTB/SmolLM2-{SIZE}-Instruct"

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
dataset = dataset.shuffle().select(range(100))


# %%

print(dataset[0])

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

ds = dataset.map(create_triplet, fn_kwargs={"tokenizer": tokenizer})
ds = ds.train_test_split(test_size=0.05)

print(ds['train'][0]["prompt"])
print("-+"*10)
print(ds['train'][0]["chosen"])
print("-+"*10)
print(ds['train'][0]["rejected"])


# the maximum length of the prompt 
prompt_length = 1024
# the maximum length of the prompt + chosen or rejected response
max_seq_length = 1512


# %%

from transformers import TrainingArguments
 
args = TrainingArguments(
    output_dir="doplhin-dpo",               # directory to save and repository id
    num_train_epochs=1,                     # number of training epochs
    per_device_train_batch_size=12,         # batch size per device during training
    per_device_eval_batch_size=4,           # batch size for evaluation
    gradient_accumulation_steps=1,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    learning_rate=5e-5,                     # 10x higher LR than QLoRA paper
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.1,                       # warmup ratio based on QLoRA paper
    lr_scheduler_type="cosine",             # use cosine learning rate scheduler
    logging_steps=25,                       # log every 25 steps
    save_steps=500,                         # when to save checkpoint
    save_total_limit=2,                     # limit the total amount of checkpoints
    evaluation_strategy="steps",            # evaluate every 1000 steps
    eval_steps=700,                         # when to evaluate
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    push_to_hub=False,                      # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
)
 
dpo_args = {
    "beta": 0.1,                            # The beta factor in DPO loss. Higher beta means less divergence
    "loss_type": "sigmoid"                  # The loss type for DPO.
}


from trl import DPOTrainer
 
trainer = DPOTrainer(
    model,
    ref_model=None, # set to none since we use peft
    peft_config=peft_config,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_length=max_seq_length,
    max_prompt_length=prompt_length,
    beta=dpo_args["beta"],
    loss_type=dpo_args["loss_type"],
)

# start training, the model will be automatically saved to the hub and the output directory
trainer.train()
